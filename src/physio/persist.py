"""
persist.py – zápis výstupů src/physio do databáze
==================================================

Posudek R-R jde do ``activity_metrics``, tepová křivka a souvislé bloky do
``activity_hr_curve`` a ``activity_hr_blocks``.

Oddělené od CLI, aby ``src/physio`` šlo použít i bez běžící databáze:
import ``src.physio`` nesahá na SQLAlchemy, tenhle modul se importuje až
při ``--write-db``.

Do ``master_high_res_summary.csv`` se hodnoty dostanou standardním
exportem (``src/analytics/exports.py``) – CSV je výstup z databáze, ne
zdroj, takže se do něj nezapisuje přímo.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if TYPE_CHECKING:
    from src.physio.cli import ActivityRrReport

log = logging.getLogger("physio.persist")

# Sloupce, které tenhle modul vlastní. Nic jiného v activity_metrics
# nepřepisuje – prahy ani α1 nepočítá, takže je nechává být.
RR_DIAGNOSTIC_COLUMNS = [
    "dfa_quality",
    "rr_beat_count",
    "rr_artifact_pct",
    "rr_zero_diff_pct",
    "rr_unique_values",
    "rr_lattice_coverage",
    "rr_authenticity",
]


def report_to_row(report: ActivityRrReport) -> dict:
    """Převede posudek na řádek pro ``activity_metrics``."""
    return {
        "activity_id": report.activity_id,
        "dfa_quality": report.dfa_quality,
        # Nula se schválně nepřevádí na NULL: "hrv zprávy tam byly, ale ani
        # jedna hodnota nebyla platná" je jiný stav než "neměřeno".
        "rr_beat_count": report.beat_count if report.has_hrv_messages else None,
        "rr_artifact_pct": (
            round(report.artifact_pct, 5) if report.artifact_pct is not None else None
        ),
        "rr_zero_diff_pct": (
            round(report.zero_diff_pct, 5) if report.zero_diff_pct is not None else None
        ),
        "rr_unique_values": report.unique_values,
        "rr_lattice_coverage": (
            round(report.lattice_coverage, 5) if report.lattice_coverage is not None else None
        ),
        "rr_authenticity": report.verdict,
    }


def dedupe_reports(reports: list[ActivityRrReport]) -> list[ActivityRrReport]:
    """
    Jeden posudek na aktivitu.

    Táž aktivita bývá na disku pod dvěma názvy (``activity_123.fit`` i
    ``123_ACTIVITY.fit``) – bez sloučení by upsert dostal stejný klíč
    dvakrát a Postgres ho odmítne ("ON CONFLICT DO UPDATE command cannot
    affect row a second time").

    Vyhrává posudek s víc tepy: prázdný export téže aktivity nesmí přebít
    ten, ve kterém R-R jsou.

    Args:
        reports: Posudky, klidně s opakujícím se ``activity_id``.

    Returns:
        Posudky s unikátním ``activity_id``, v pořadí prvního výskytu.
    """
    best: dict[str, ActivityRrReport] = {}
    for report in reports:
        current = best.get(report.activity_id)
        if current is None or report.beat_count > current.beat_count:
            best[report.activity_id] = report

    if len(best) < len(reports):
        log.info("%d posudků sloučeno na %d aktivit (tentýž FIT pod víc názvy).",
                 len(reports), len(best))
    return list(best.values())


def write_reports_to_db(reports: list[ActivityRrReport]) -> int:
    """
    Zapíše posudky do ``activity_metrics``.

    Aktivity, které v tabulce ``activities`` nejsou (FIT soubor, který se
    nikdy nenačetl), se přeskočí – upsert by na cizí klíč spadl.

    Args:
        reports: Posudky z ``run_batch``.

    Returns:
        Počet zapsaných řádků.
    """
    from sqlalchemy import select

    from src.db import repository as repo
    from src.db.models import Activity
    from src.db.session import session_scope

    rows = [report_to_row(r) for r in dedupe_reports(reports)]
    if not rows:
        return 0

    with session_scope() as session:
        known = {
            aid
            for (aid,) in session.execute(
                select(Activity.activity_id).where(
                    Activity.activity_id.in_([r["activity_id"] for r in rows])
                )
            ).all()
        }
        skipped = [r["activity_id"] for r in rows if r["activity_id"] not in known]
        if skipped:
            log.warning("%d aktivit není v databázi, přeskakuji: %s",
                        len(skipped), ", ".join(skipped[:5]))

        payload = [r for r in rows if r["activity_id"] in known]
        if not payload:
            return 0
        repo.upsert_activity_metrics(session, payload)
        return len(payload)


# ═══════════════════════════════════════════════════════════════════════════
# Tepová křivka a souvislé bloky
# ═══════════════════════════════════════════════════════════════════════════

def write_hr_rows(session, curve_rows: list[dict], block_rows: list[dict]) -> dict[str, int]:
    """
    Zapíše tepovou křivku a bloky do jejich tabulek.

    Stejná cesta jako u posudku R-R: aktivity, které nejsou v ``activities``,
    se přeskočí, jinak by upsert spadl na cizí klíč.

    Session se sem předává zvenčí (na rozdíl od ``write_reports_to_db``),
    protože volající si ji stejně otevírá kvůli čtení vteřinových dat – dva
    nezávislé transakční kontexty nad týmž během nemají důvod existovat.

    Args:
        session: Otevřená session.
        curve_rows: Řádky pro ``activity_hr_curve``.
        block_rows: Řádky pro ``activity_hr_blocks``.

    Returns:
        ``{"curve": n, "blocks": n}`` – počty zapsaných řádků.
    """
    from sqlalchemy import select

    from src.db import repository as repo
    from src.db.models import Activity

    ids = {r["activity_id"] for r in curve_rows} | {r["activity_id"] for r in block_rows}
    if not ids:
        return {"curve": 0, "blocks": 0}

    known = {
        aid
        for (aid,) in session.execute(
            select(Activity.activity_id).where(Activity.activity_id.in_(list(ids)))
        ).all()
    }
    if len(known) < len(ids):
        missing = sorted(ids - known)
        log.warning("%d aktivit není v databázi, přeskakuji: %s",
                    len(missing), ", ".join(missing[:5]))

    curve = [r for r in curve_rows if r["activity_id"] in known]
    blocks = [r for r in block_rows if r["activity_id"] in known]

    return {
        "curve": repo.upsert_hr_curve(session, curve),
        "blocks": repo.upsert_hr_blocks(session, blocks),
    }
