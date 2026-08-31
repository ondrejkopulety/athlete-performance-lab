"""
Panely tepové křivky a souvislých bloků + nastavení prahu.

Odděleno od ``/api/dashboard``, který posílá celou historii najednou a
přepínač období řeší prohlížeč. Tady to nejde: obojí je agregace přes
zvolené období (nejlepší okno křivky, nejdelší blok), a hlavně přes
proměnlivý filtr – "jen úplná data" a práh z LTHR mění, které řádky do
agregace vůbec vstoupí. Posílat kvůli tomu do prohlížeče 17 tisíc řádků
bloků by bylo dražší než dotaz, který vrátí deset čísel.

Nic se tu nepočítá ze sekundových dat. Když na něco data nejsou, jde ven
``null`` a UI ukazuje pomlčku.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from config.settings import (
    HR_BLOCK_BRIDGE_TOLERANCES_S,
    HR_BLOCK_PRIMARY_ZONE,
    HR_BLOCK_THRESHOLDS_BPM,
    HR_COVERAGE_WARN_PCT,
    HR_CURVE_DURATIONS_S,
    HR_ZONE_LTHR_RATIO,
    LTHR_DEFAULT_BPM,
    LTHR_TEST_MINUTES,
    MAX_HR,
    THRESHOLD_STALE_DAYS,
    Z3_BRIDGE_TOLERANCE_S,
)
from src.analytics.hr_panels import (
    block_period_summary,
    curve_points,
    last_max_effort,
    zone_threshold_bpm,
)
from src.api.schemas import HrBlocksOut, HrCurveOut, ThresholdIn, ThresholdOut
from src.db import repository as repo
from src.db.session import get_session
from src.ingestion.sport import CYCLING_SPORT_REGEX as CYCLING_SPORT_PATTERN

router = APIRouter(tags=["hr"])

# Okno, ze kterého se čte "poslední maximální výkon". Dvacet minut je
# nejkratší úsilí, které se ještě nedá odjet na setrvačnost jedné pasáže.
LAST_EFFORT_DURATION_S = LTHR_TEST_MINUTES * 60
LAST_EFFORT_STALE_DAYS = 60


def _period_bounds(
    since: date | None, until: date | None
) -> tuple[date, date]:
    """Doplní chybějící kraje období na rozumné výchozí hodnoty."""
    end = until or date.today()
    start = since or (end - timedelta(days=90))
    return start, end


def _reference_window(start: date, end: date, mode: str) -> tuple[date, date, str]:
    """
    Referenční období pro srovnání.

    ``year`` = totéž období vloni (sezónnost: srpen se srovnává se srpnem),
    ``prev`` = předchozí okno stejné délky.
    """
    if mode == "year":
        return (
            start - timedelta(days=365),
            end - timedelta(days=365),
            "vloni touhle dobou",
        )
    length = (end - start).days
    return start - timedelta(days=length + 1), start - timedelta(days=1), (
        f"předchozích {length + 1} dní"
    )


def _curve_period(
    session: Session,
    start: date,
    end: date,
    label: str,
    complete_only: bool,
    today: date,
) -> dict:
    """Body křivky za jedno období i s počtem vyloučených jízd."""
    min_coverage = HR_COVERAGE_WARN_PCT if complete_only else None

    rows = repo.read_curve_rows(session, start, end, min_coverage_pct=min_coverage)
    points = curve_points(rows, HR_CURVE_DURATIONS_S)

    total = repo.count_rides_in_period(session, CYCLING_SPORT_PATTERN, start, end)
    kept = (
        repo.count_rides_in_period(
            session, CYCLING_SPORT_PATTERN, start, end, min_coverage_pct=min_coverage
        )
        if complete_only
        else total
    )

    return {
        "label": label,
        "from_date": start,
        "to_date": end,
        "rides_total": total,
        "rides_excluded": max(0, total - kept),
        "points": [p.as_dict() for p in points],
        "last_max_effort": last_max_effort(
            points, LAST_EFFORT_DURATION_S, today, LAST_EFFORT_STALE_DAYS
        ),
    }


@router.get("/hr/curve", response_model=HrCurveOut)
def hr_curve(
    session: Session = Depends(get_session),
    since: date | None = None,
    until: date | None = None,
    complete_only: bool = True,
    compare: Annotated[str, Query(pattern="^(prev|year|none)$")] = "prev",
) -> HrCurveOut:
    """
    Tepová křivka za období a referenční období k porovnání.

    ``complete_only`` je ve výchozím stavu zapnuté: metriky z jízd s dírami
    v tepu jsou systematicky podhodnocené a míchat je do maxima znamená
    srovnávat výkon s kvalitou záznamu.
    """
    start, end = _period_bounds(since, until)
    today = date.today()

    period = _curve_period(session, start, end, "zvolené období", complete_only, today)

    reference = None
    if compare != "none":
        ref_start, ref_end, ref_label = _reference_window(start, end, compare)
        reference = _curve_period(
            session, ref_start, ref_end, ref_label, complete_only, today
        )

    return HrCurveOut(
        durations_s=list(HR_CURVE_DURATIONS_S),
        complete_only=complete_only,
        coverage_warn_pct=HR_COVERAGE_WARN_PCT,
        period=period,
        reference=reference,
    )


@router.get("/hr/blocks", response_model=HrBlocksOut)
def hr_blocks(
    session: Session = Depends(get_session),
    since: date | None = None,
    until: date | None = None,
    zone: Annotated[str, Query(pattern="^Z[2-5]$")] = HR_BLOCK_PRIMARY_ZONE,
    tolerance: int = Z3_BRIDGE_TOLERANCE_S,
    complete_only: bool = True,
) -> HrBlocksOut:
    """
    Souvislé bloky nad prahem zóny za období.

    Práh se odvodí z nastaveného LTHR a zaokrouhlí na nejbližší uložený
    práh mřížky – **lookup, ne výpočet**. Změna LTHR proto mění jen to,
    na který řádek se sáhne; v ``activity_hr_blocks`` se nepřepisuje nic.
    """
    start, end = _period_bounds(since, until)
    if tolerance not in HR_BLOCK_BRIDGE_TOLERANCES_S:
        tolerance = HR_BLOCK_BRIDGE_TOLERANCES_S[-1]

    threshold_state = repo.read_current_threshold(session)
    lthr = int(threshold_state["lthr_bpm"]) if threshold_state else LTHR_DEFAULT_BPM

    requested = round(lthr * HR_ZONE_LTHR_RATIO[zone])
    threshold = zone_threshold_bpm(lthr, zone, HR_BLOCK_THRESHOLDS_BPM)
    min_coverage = HR_COVERAGE_WARN_PCT if complete_only else None

    rows = repo.read_block_rows(
        session, threshold, tolerance, start, end, min_coverage_pct=min_coverage
    )
    summary = block_period_summary(rows)

    # Trend proti předchozímu oknu stejné délky.
    length = (end - start).days
    prev_rows = repo.read_block_rows(
        session,
        threshold,
        tolerance,
        start - timedelta(days=length + 1),
        start - timedelta(days=1),
        min_coverage_pct=min_coverage,
    )
    previous = block_period_summary(prev_rows)["longest_block_s"]

    current = summary["longest_block_s"]
    trend = None
    if current is not None and previous:
        trend = round(100.0 * (current - previous) / previous, 1)

    total = repo.count_rides_in_period(session, CYCLING_SPORT_PATTERN, start, end)
    kept = (
        repo.count_rides_in_period(
            session, CYCLING_SPORT_PATTERN, start, end, min_coverage_pct=min_coverage
        )
        if complete_only
        else total
    )

    return HrBlocksOut(
        requested_bpm=requested,
        threshold_bpm=threshold,
        zone=zone,
        lthr_bpm=lthr,
        tolerance_s=tolerance,
        complete_only=complete_only,
        coverage_warn_pct=HR_COVERAGE_WARN_PCT,
        rides_total=total,
        rides_excluded=max(0, total - kept),
        longest_block_s=current,
        longest_block=summary["longest_block"],
        previous_longest_block_s=previous,
        trend_pct=trend,
        hist=summary["hist"],
        totals=summary["totals"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Práh a jeho stáří
# ═══════════════════════════════════════════════════════════════════════════

def _threshold_payload(state: dict | None, today: date) -> ThresholdOut:
    """
    Nastavený práh i s tím, jak je starý.

    Bez uloženého řádku platí měřené hodnoty ze settings (``source`` je pak
    ``"settings"``). Stáří se v tom případě počítá od dneška, tedy nula –
    tvrdit, že hodnota ze souboru zastarala před N dny, by bylo vymýšlení.
    """
    if state is None:
        lthr, hr_max, valid_from, note, source = (
            LTHR_DEFAULT_BPM,
            MAX_HR,
            today,
            "měřené zóny z config/settings.py",
            "settings",
        )
    else:
        lthr = int(state["lthr_bpm"])
        hr_max = int(state["hr_max_bpm"])
        valid_from = state["valid_from"]
        note = state["note"]
        source = "user"

    days = (today - valid_from).days
    return ThresholdOut(
        lthr_bpm=lthr,
        hr_max_bpm=hr_max,
        valid_from=valid_from,
        note=note,
        days_ago=days,
        stale=days > THRESHOLD_STALE_DAYS,
        stale_after_days=THRESHOLD_STALE_DAYS,
        source=source,
        # Zóna → práh z mřížky. Ukazuje se v UI, aby bylo vidět, na který
        # uložený řádek se panel bloků ptá.
        zone_thresholds={
            z: zone_threshold_bpm(lthr, z, HR_BLOCK_THRESHOLDS_BPM)
            for z in HR_ZONE_LTHR_RATIO
        },
    )


@router.get("/profile/threshold", response_model=ThresholdOut)
def get_threshold(session: Session = Depends(get_session)) -> ThresholdOut:
    """Aktuální práh a jeho stáří."""
    return _threshold_payload(repo.read_current_threshold(session), date.today())


@router.put("/profile/threshold", response_model=ThresholdOut)
def put_threshold(
    payload: ThresholdIn, session: Session = Depends(get_session)
) -> ThresholdOut:
    """
    Uloží nový práh jako další řádek historie.

    Žádná uložená data se tím nepřepočítávají: bloky jsou na mřížce
    absolutních prahů a zóny v settings jsou měřené. Mění se dotaz, ne data.
    """
    state = repo.insert_threshold(
        session,
        lthr_bpm=payload.lthr_bpm,
        hr_max_bpm=payload.hr_max_bpm,
        valid_from=payload.valid_from or date.today(),
        note=payload.note,
    )
    return _threshold_payload(state, date.today())
