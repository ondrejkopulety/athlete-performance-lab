"""
strava_map.py  –  doplnění odkazů na aktivity na Stravě
=======================================================

Deduplikace (`src/ingestion/dedup.py`) umí ke Garmin jízdě přiřadit Strava ID
jen tehdy, když je na disku i odpovídající Strava FIT soubor. `data/fit/
strava_originals/` je ale zmrazený hromadný export – jízdy nahrané po něm
(k srpnu 2026 jich bylo ~150) žádný Strava protějšek na disku nemají a
`activities.strava_id` u nich zůstane `NULL`.

Tenhle krok to dořeší bez stahování FIT: přes Strava REST API si vytáhne
*seznam* aktivit (id + čas startu) a spáruje ho s naší tabulkou `activities`
podle času startu (okno `STRAVA_MATCH_WINDOW_MIN`). Potřebuje jen OAuth
(`STRAVA_CLIENT_ID` / `_SECRET` / `_REFRESH_TOKEN` v `.env`), ne session cookie –
ta je nutná až na stahování originálních souborů (`strava_client.py`).

Selhání není fatální: stejně jako SYNC se jen zaloguje a pipeline pokračuje.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from config.settings import STRAVA_API_ACTIVITIES, STRAVA_MATCH_WINDOW_MIN
from src.db.models import Activity
from src.ingestion.strava_auth import StravaAuthError, refresh_access_token

log = logging.getLogger("strava_map")

_HTTP_TIMEOUT = 30
_PER_PAGE = 200

__all__ = ["StravaAuthError", "MapResult", "map_strava_ids", "fetch_strava_activities"]


@dataclass
class MapResult:
    db_activities: int = 0          # kolik jízd v DB mělo použitelný start_time
    strava_activities: int = 0      # kolik aktivit vrátila Strava
    already_linked: int = 0         # už měly strava_id (a nebyl --force)
    matched: int = 0               # nově spárováno v tomhle běhu
    updated: int = 0               # skutečně zapsáno do DB (matched mínus shody a dry-run)
    unmatched: int = 0             # jízdy bez protějšku ve Stravě
    ambiguous: int = 0             # víc kandidátů ve stejném okně – vzat nejbližší
    dry_run: bool = False
    samples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "[dry-run] " if self.dry_run else ""
        return (
            f"{head}{self.strava_activities} aktivit ze Stravy, "
            f"{self.db_activities} jízd v DB: {self.updated} nově s odkazem, "
            f"{self.already_linked} už mělo, {self.unmatched} bez protějšku"
            + (f", {self.ambiguous} nejednoznačných" if self.ambiguous else "")
        )


# ═══════════════════════════════════════════════════════════════════════════
# Načtení seznamu aktivit
# ═══════════════════════════════════════════════════════════════════════════

def _parse_start(value: str | None) -> datetime | None:
    """Strava `start_date` ('2026-08-30T13:34:25Z') → naivní UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class StravaActivity:
    id: int
    start: datetime
    name: str
    sport: str
    elapsed_s: int | None


# Přechodné chyby serveru – krátký retry s backoffem. 429 mezi nimi NENÍ:
# na rate limit se čeká minuty, ne vteřiny, a krok stejně není fatální.
_RETRY_HTTP_CODES = frozenset({500, 502, 503, 504})
_RETRY_BACKOFF_S = (2, 5)


def _get_page(headers: dict, page: int) -> requests.Response:
    """Jedna stránka ``/athlete/activities`` s krátkým retry na 5xx."""
    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_BACKOFF_S) + 1):
        try:
            resp = requests.get(
                STRAVA_API_ACTIVITIES,
                headers=headers,
                params={"per_page": _PER_PAGE, "page": page},
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if resp.status_code == 429:
                raise StravaAuthError("Strava API rate limit (429) – zkus později.")
            if resp.status_code not in _RETRY_HTTP_CODES:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        if attempt < len(_RETRY_BACKOFF_S):
            wait = _RETRY_BACKOFF_S[attempt]
            log.warning("Strava API strana %d selhala (%s) – opakuji za %d s.", page, last_exc, wait)
            time.sleep(wait)
    raise StravaAuthError(f"Strava API nedostupné (strana {page}): {last_exc}")


def fetch_strava_activities(token: str) -> list[StravaActivity]:
    """Kompletní historie aktivit přes stránkované `/athlete/activities`."""
    headers = {"Authorization": f"Bearer {token}"}
    out: list[StravaActivity] = []
    page = 1
    while True:
        resp = _get_page(headers, page)
        if resp.status_code != 200:
            raise StravaAuthError(f"Strava API vrátilo {resp.status_code}: {resp.text[:200]}")

        batch = resp.json()
        if not batch:
            break
        for a in batch:
            start = _parse_start(a.get("start_date"))
            if start is None or a.get("id") is None:
                continue
            out.append(
                StravaActivity(
                    id=int(a["id"]),
                    start=start,
                    name=str(a.get("name") or ""),
                    sport=str(a.get("sport_type") or a.get("type") or ""),
                    elapsed_s=a.get("elapsed_time"),
                )
            )
        if len(batch) < _PER_PAGE:
            break
        page += 1
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Párování podle času startu
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _DbRow:
    activity_id: str
    start: datetime
    has_link: bool


def _match(
    db_rows: list[_DbRow],
    strava: list[StravaActivity],
    window_min: float,
) -> tuple[dict[str, int], int]:
    """
    Přiřadí každé jízdě nejvýš jednu Strava aktivitu (a naopak) tak, že se
    projdou všechny dvojice v okně a berou se od nejmenšího časového rozdílu.
    Vrací (activity_id -> strava_id, počet nejednoznačných).
    """
    window_s = window_min * 60.0
    db_sorted = sorted(db_rows, key=lambda r: r.start)
    sv_sorted = sorted(strava, key=lambda s: s.start)

    # Kandidátní dvojice v okně (dvojitý ukazatel přes seřazené seznamy).
    pairs: list[tuple[float, str, int]] = []  # (|Δt|, activity_id, strava_id)
    lo = 0
    for row in db_sorted:
        while lo < len(sv_sorted) and (row.start - sv_sorted[lo].start).total_seconds() > window_s:
            lo += 1
        j = lo
        while j < len(sv_sorted) and (sv_sorted[j].start - row.start).total_seconds() <= window_s:
            delta = abs((sv_sorted[j].start - row.start).total_seconds())
            pairs.append((delta, row.activity_id, sv_sorted[j].id))
            j += 1

    per_activity: dict[str, int] = {}
    used_strava: set[int] = set()
    taken_activity: set[str] = set()
    candidates_per_activity: dict[str, int] = {}
    for _, aid, sid in pairs:
        candidates_per_activity[aid] = candidates_per_activity.get(aid, 0) + 1

    for delta, aid, sid in sorted(pairs, key=lambda p: p[0]):
        if aid in taken_activity or sid in used_strava:
            continue
        per_activity[aid] = sid
        taken_activity.add(aid)
        used_strava.add(sid)

    ambiguous = sum(1 for aid in per_activity if candidates_per_activity.get(aid, 0) > 1)
    return per_activity, ambiguous


# ═══════════════════════════════════════════════════════════════════════════
# Hlavní vstupní bod
# ═══════════════════════════════════════════════════════════════════════════

def map_strava_ids(
    session: Session,
    *,
    force: bool = False,
    window_min: float = STRAVA_MATCH_WINDOW_MIN,
    dry_run: bool = False,
) -> MapResult:
    """
    Doplní `activities.strava_id` z REST API Stravy.

    force=False (výchozí): párují se jen jízdy bez odkazu.
    force=True: přepočítají se všechny; existující odkaz se přepíše jen tehdy,
    když se najde shoda – bez shody zůstane, co bylo.
    """
    result = MapResult(dry_run=dry_run)

    rows = session.execute(
        select(Activity.activity_id, Activity.start_time, Activity.strava_id)
    ).all()
    db_rows: list[_DbRow] = []
    pre_link: dict[str, str | None] = {}
    for activity_id, start_time, strava_id in rows:
        if start_time is None:
            continue
        has_link = strava_id is not None
        if has_link and not force:
            result.already_linked += 1
            continue
        pre_link[activity_id] = strava_id
        db_rows.append(_DbRow(activity_id=activity_id, start=start_time, has_link=has_link))
    result.db_activities = len(db_rows)

    if not db_rows:
        log.info("Strava párování: všechny jízdy už odkaz mají.")
        return result

    # dry-run nesmí mít vedlejší efekt: když Strava vrátí rotovaný refresh
    # token, do .env se nezapíše (jen se zaloguje varování).
    token = refresh_access_token(persist_rotation=not dry_run)
    strava = fetch_strava_activities(token)
    result.strava_activities = len(strava)
    if not strava:
        log.warning("Strava nevrátila žádné aktivity.")
        return result

    matches, ambiguous = _match(db_rows, strava, window_min)
    result.ambiguous = ambiguous
    result.matched = len(matches)
    result.unmatched = len(db_rows) - len(matches)

    sample_lines: list[str] = []
    for activity_id, strava_id in matches.items():
        sid = str(strava_id)
        if pre_link.get(activity_id) == sid:
            continue  # force: shoda beze změny
        result.updated += 1
        if len(sample_lines) < 8:
            sample_lines.append(f"{activity_id} → {sid}")
        if not dry_run:
            session.execute(
                update(Activity).where(Activity.activity_id == activity_id).values(strava_id=sid)
            )
    result.samples = sample_lines

    if dry_run:
        session.rollback()
    else:
        session.commit()

    log.info("Strava párování: %s", result.summary())
    for line in sample_lines:
        log.info("  %s", line)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _main(argv: Iterable[str] | None = None) -> int:
    import argparse

    from config.settings import LOGS_DIR
    from src.db.session import session_scope

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Doplní activities.strava_id z REST API Stravy")
    p.add_argument("--force", action="store_true",
                   help="Přepočítej i jízdy, které už odkaz mají")
    p.add_argument("--window", type=float, default=STRAVA_MATCH_WINDOW_MIN,
                   help=f"Okno shody času startu v minutách (výchozí {STRAVA_MATCH_WINDOW_MIN})")
    p.add_argument("--dry-run", action="store_true",
                   help="Jen ukaž, co by se spárovalo; nic nezapisuj")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        with session_scope() as session:
            res = map_strava_ids(
                session, force=args.force, window_min=args.window, dry_run=args.dry_run
            )
    except StravaAuthError as exc:
        log.error("%s", exc)
        return 1
    print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
