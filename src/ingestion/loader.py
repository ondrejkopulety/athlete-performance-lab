"""
loader.py  –  FIT soubory → databáze
=====================================

Nahrazuje dřívější dvojici kroků `parse` + `merge`:

  • starý `parse` (fit_parser.main) psal high_res_*.csv, které nikdo nečetl,
  • starý `merge` (master_rebuild.main) tytéž FIT soubory parsoval znovu.

Tady se každý FIT soubor otevře právě jednou a jeho obsah jde rovnou do DB.

Inkrementalita stojí na SHA-256 obsahu souboru: aktivita, jejíž hash sedí
s uloženým, se přeskočí bez otevření. Změněný soubor (Strava re-export,
oprava dat) se přeparsuje a jeho vteřinová data se nahradí.

R-R intervaly se ukládají do activity_metrics.rr_intervals_ms, takže
pozdější výpočty DFA-alpha1 a dechové frekvence už nemusí sahat na disk.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sqlalchemy.orm import Session

from src.db import repository as repo
from src.ingestion.dedup import canonical_fit_files, file_sha256
from src.ingestion.fit_parser import parse_fit_to_memory
from src.physio.rr_extract import extract_rr

log = logging.getLogger("loader")

# Po kolika aktivitách commitovat. Kompromis mezi velikostí transakce
# (paměť + WAL) a režií commitu.
COMMIT_EVERY = 25


@dataclass
class LoadResult:
    scanned: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    records_written: int = 0
    changed_activity_ids: list[str] = field(default_factory=list)
    earliest_changed_date: Any = None  # datetime.date | None

    def summary(self) -> str:
        return (
            f"{self.scanned} FIT souborů: {self.inserted} nových, "
            f"{self.updated} aktualizovaných, {self.skipped} beze změny, "
            f"{self.failed} chyb, {self.records_written} vteřinových záznamů"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Konverze hodnot z CSV-orientovaného parseru
# ═══════════════════════════════════════════════════════════════════════════

def _blank(v: Any) -> Any:
    """fit_parser vrací chybějící hodnoty jako "" (dědictví CSV) → None."""
    if v is None or v == "":
        return None
    return v


def _num(v: Any) -> float | None:
    v = _blank(v)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_rr_intervals_ms(fit_path: str) -> list[float]:
    """
    R-R intervaly (ms) z HRV zpráv FIT souboru.

    Ukládají se při načtení, aby DFA-alpha1 a RSA nemusely otevírat FIT
    soubor při každém běhu analytiky.

    Vlastní čtení dělá ``src.physio.rr_extract`` – jediné místo v repu, kde
    se ``hrv`` zprávy parsují. Dřív existovaly tři kopie téhle logiky a
    lišily se v tom, které hodnoty považují za platné.
    """
    extraction = extract_rr(fit_path)
    if extraction.error:
        log.debug("R-R intervaly nelze načíst z %s: %s", fit_path, extraction.error)
        return []

    # FIT ukládá R-R v sekundách; do DB jdou milisekundy (jednotka, se kterou
    # pracuje neurokit2 i všechny HRV vzorce).
    return [round(v * 1000.0, 1) for v in extraction.rr_seconds]


def _parse_worker(args: tuple) -> Optional[dict]:
    """Picklovatelný worker pro mp.Pool – jeden FIT soubor = jeden úkol."""
    path, source, sha, start_time, strava_id = args
    parsed = parse_fit_to_memory(path)
    if parsed is None:
        return None
    summary, rows = parsed
    return {
        "summary": summary,
        "rows": rows,
        "rr": extract_rr_intervals_ms(path),
        "path": path,
        "source": source,
        "sha": sha,
        "start_time": start_time,
        "strava_id": strava_id,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Mapování na DB řádky
# ═══════════════════════════════════════════════════════════════════════════

def _activity_row(res: dict) -> dict:
    s = res["summary"]
    act_date = _blank(s.get("date"))
    return {
        "activity_id": str(s["activity_id"]),
        "start_time": res.get("start_time"),
        "date": datetime.fromisoformat(act_date).date() if act_date else None,
        "activity_name": _blank(s.get("activity_name")),
        "sport": _blank(s.get("sport")),
        "duration_minutes": _num(s.get("duration_minutes")),
        "total_trimp": _num(s.get("total_trimp")),
        "avg_hr": _num(s.get("avg_hr")),
        "max_hr": _num(s.get("max_hr")),
        "time_in_z1": _num(s.get("time_in_z1")),
        "time_in_z2": _num(s.get("time_in_z2")),
        "time_in_z3": _num(s.get("time_in_z3")),
        "time_in_z4": _num(s.get("time_in_z4")),
        "time_in_z5": _num(s.get("time_in_z5")),
        "zone2_cap_used": _num(s.get("zone2_cap_used")),
        "records_count": int(_num(s.get("records_count")) or 0),
        "distance_km": _num(s.get("distance_km")),
        "ascent_m": _num(s.get("ascent_m")),
        "descent_m": _num(s.get("descent_m")),
        "avg_speed_kmh": _num(s.get("avg_speed_kmh")),
        "max_speed_kmh": _num(s.get("max_speed_kmh")),
        "calories": _num(s.get("calories")),
        "uphill_minutes": _num(s.get("uphill_minutes")),
        "downhill_minutes": _num(s.get("downhill_minutes")),
        "flat_minutes": _num(s.get("flat_minutes")),
        "avg_cadence": _num(s.get("avg_cadence")),
        "max_cadence": _num(s.get("max_cadence")),
        "avg_temp": _num(s.get("avg_temp")),
        "max_temp": _num(s.get("max_temp")),
        "training_effect_aerobic": _num(s.get("training_effect_aerobic")),
        "training_effect_anaerobic": _num(s.get("training_effect_anaerobic")),
        "vo2_max": _num(s.get("vo2_max")),
        "source": res["source"],
        "strava_id": res.get("strava_id"),
        "fit_path": res["path"],
        "fit_sha256": res["sha"],
    }


def _record_rows(activity_id: str, rows: list[dict]):
    """Vteřinové řádky převedené na tvar, který spolkne COPY."""
    for r in rows:
        ts = _blank(r.get("timestamp"))
        if ts is None:
            continue
        is_active = r.get("is_active")
        yield {
            "activity_id": activity_id,
            "timestamp": ts,
            "heart_rate": _num(r.get("heart_rate")),
            "speed": _num(r.get("speed")),
            "power": _num(r.get("power")),
            "cadence": _num(r.get("cadence")),
            "altitude": _num(r.get("altitude")),
            "distance": _num(r.get("distance")),
            "temperature": _num(r.get("temperature")),
            "vertical_oscillation": _num(r.get("vertical_oscillation")),
            "stance_time": _num(r.get("stance_time")),
            "respiratory_rate": _num(r.get("respiratory_rate")),
            "hrv": _num(r.get("hrv")),
            "position_lat": _num(r.get("position_lat")),
            "position_long": _num(r.get("position_long")),
            "hr_zone": _blank(r.get("hr_zone")),
            "is_active": bool(is_active) if is_active not in (None, "") else None,
            "trimp_increment": _num(r.get("trimp_increment")),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Hlavní vstupní bod
# ═══════════════════════════════════════════════════════════════════════════

def load_fit_files(
    session: Session,
    force: bool = False,
    limit: int | None = None,
    workers: int | None = None,
) -> LoadResult:
    """
    Načte kanonickou (deduplikovanou) sadu FIT souborů do databáze.

    force=True ignoruje shodu hashů a přeparsuje vše – použij po změně
    parsovací logiky ve fit_parser.py.
    """
    result = LoadResult()
    entries = canonical_fit_files()
    known = repo.existing_activity_hashes(session)

    todo: list[tuple] = []
    for e in entries:
        result.scanned += 1
        aid = e["activity_id"]
        try:
            sha = file_sha256(e["path"])
        except OSError as exc:
            log.warning("Nelze přečíst %s: %s", e["path"], exc)
            result.failed += 1
            continue

        if not force and aid in known and known[aid] == sha:
            result.skipped += 1
            continue
        todo.append((e["path"], e["source"], sha, e.get("start_time"), e.get("strava_id")))

    if limit is not None:
        todo = todo[:limit]

    if not todo:
        log.info("Žádné nové ani změněné FIT soubory (%d beze změny).", result.skipped)
        return result

    n_workers = workers or max(1, mp.cpu_count() - 1)
    log.info("Zpracovávám %d FIT souborů na %d procesech…", len(todo), n_workers)

    def _handle(res: Optional[dict]) -> None:
        if res is None:
            result.failed += 1
            return
        row = _activity_row(res)
        aid = row["activity_id"]

        repo.upsert_activities(session, [row])
        # Vteřinová data se nahrazují celá – částečný re-import by nechal
        # v tabulce zamíchané staré a nové vzorky.
        repo.delete_records(session, aid)
        session.flush()
        n = repo.copy_records(session, _record_rows(aid, res["rows"]))
        result.records_written += n

        repo.upsert_activity_metrics(
            session,
            [{
                "activity_id": aid,
                "metrics_version": 0,  # 0 = čeká na výpočet odvozených metrik
                "rr_intervals_ms": res["rr"] or None,
            }],
        )

        if aid in known:
            result.updated += 1
        else:
            result.inserted += 1
        result.changed_activity_ids.append(aid)
        if row["date"] is not None:
            if result.earliest_changed_date is None or row["date"] < result.earliest_changed_date:
                result.earliest_changed_date = row["date"]

    def _progress(i: int, res: Optional[dict]) -> None:
        """Průběžný commit – jedna transakce přes 864 aktivit by držela
        miliony řádků a při pádu by se zahodil celý několikaminutový běh."""
        _handle(res)
        if i % COMMIT_EVERY == 0:
            session.commit()
            log.info("  [%d/%d] průběžně uloženo (%d záznamů)",
                     i, len(todo), result.records_written)

    if n_workers == 1:
        for i, args in enumerate(todo, 1):
            log.info("  [%d/%d] %s", i, len(todo), os.path.basename(args[0]))
            _progress(i, _parse_worker(args))
    else:
        with mp.Pool(processes=n_workers) as pool:
            for i, res in enumerate(pool.imap(_parse_worker, todo, chunksize=2), 1):
                if res is None:
                    log.warning("  [%d/%d] chyba parsování", i, len(todo))
                elif i % 25 == 0:
                    log.info("  [%d/%d] %s", i, len(todo), os.path.basename(res["path"]))
                _progress(i, res)

    session.commit()
    log.info("Načtení dokončeno: %s", result.summary())
    return result
