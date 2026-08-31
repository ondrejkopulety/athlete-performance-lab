"""
Jeden agregační endpoint pro webový dashboard.

Frontend potřebuje celou historii najednou (přepínač rozsahu 7D–Vše se
přepíná v prohlížeči, ne dotazem na server), ale jen zlomek sloupců.
`/api/daily` by poslal 50 sloupců krát 1650 dní, `/api/pmc` zase nezná
readiness ani polarizaci – proto tenhle vlastní tvar.

Nic se tu nepočítá; jde o překlad databázových sloupců do tvaru, který
dashboard kreslí.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import (
    HR_BLOCK_THRESHOLDS_BPM,
    LTHR_DEFAULT_BPM,
    Z3_BRIDGE_TOLERANCE_S,
)
from src.analytics.hr_panels import Coverage, unintentional_z3_s, zone_threshold_bpm
from src.api.schemas import (
    DashboardActivity,
    DashboardBiometric,
    DashboardOut,
    DashboardRide,
    DashboardToday,
    RideCoverage,
)
from src.db import repository as repo
from src.db.models import Activity, ActivityMetrics, DailyMetrics
from src.db.session import get_session
from src.ingestion.sport import CYCLING_SPORT_REGEX, EBIKE_SPORT_PATTERN

router = APIRouter(tags=["dashboard"])

# Kolik posledních jízd se vypisuje v seznamu pod grafy.
RIDES_LIMIT = 5


def _coverage(raw: dict | None) -> RideCoverage | None:
    """Pokrytí jízdy jako verdikt. ``None`` = ještě nespočítané, což není
    totéž co špatné – karta pak nedostane odznak ani vysvětlení."""
    if raw is None:
        return None
    return RideCoverage(**Coverage(**raw).as_dict())


def _num(value: float | None, default: float | None = None) -> float | None:
    """Zaokrouhlení na rozumný počet míst; None zůstává None."""
    if value is None:
        return default
    return round(float(value), 3)


def _zero(value: float | None) -> float:
    """Řady, které se kreslí jako spojitá čára – díra v nich je opravdu nula
    (netrénovaný den má TRIMP 0, ne 'neznámo')."""
    return 0.0 if value is None else round(float(value), 3)


def _biometric(
    session: Session,
    value_col,
    reference_col,
) -> DashboardBiometric | None:
    """
    Poslední den, kdy hodnota vůbec byla naměřená, i s referencí z téhož dne.

    Ranní biometrie se z Garminu často stáhne až s odstupem, takže poslední
    den v `daily_metrics` bývá bez HRV, tepu i spánku. Bez tohohle dohledání
    by dashboard místo čísel ukazoval pomlčky.
    """
    row = session.execute(
        select(DailyMetrics.date, value_col, reference_col)
        .where(value_col.is_not(None))
        .order_by(DailyMetrics.date.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return DashboardBiometric(date=row[0], value=_num(row[1]), reference=_num(row[2]))


@router.get("/dashboard", response_model=DashboardOut)
def dashboard(session: Session = Depends(get_session)) -> DashboardOut:
    """Kompletní podklad pro dashboard: denní řady, poslední jízdy, biometrie."""

    # ── Denní řady ────────────────────────────────────────────────────────
    day_rows = session.execute(
        select(
            DailyMetrics.date,
            DailyMetrics.ctl,
            DailyMetrics.atl,
            DailyMetrics.tsb,
            DailyMetrics.trimp,
            DailyMetrics.strain,
            DailyMetrics.readiness_score,
            DailyMetrics.acwr,
        ).order_by(DailyMetrics.date)
    ).all()

    days = [
        [
            r.date.isoformat(),
            _zero(r.ctl),
            _zero(r.atl),
            _zero(r.tsb),
            _zero(r.trimp),
            _num(r.strain),
            _num(r.readiness_score),
            _num(r.acwr),
        ]
        for r in day_rows
    ]

    # ── Cyklistické aktivity ──────────────────────────────────────────────
    # Stejný filtr jako `export_cycling` – dashboard tak ukazuje přesně tu
    # množinu jízd, která je i v cycling_summary.csv. Elektrokolo je mimo:
    # do objemu (km, převýšení) se nepočítá, do formy přispívá přes TRIMP
    # v denních metrikách.
    act_rows = session.execute(
        select(
            Activity.activity_id,
            Activity.date,
            Activity.duration_minutes,
            Activity.distance_km,
            Activity.avg_hr,
            Activity.max_hr,
            Activity.ascent_m,
            Activity.total_trimp,
            Activity.calories,
            Activity.uphill_minutes,
            Activity.downhill_minutes,
            Activity.flat_minutes,
            Activity.time_in_z1,
            Activity.time_in_z2,
            Activity.time_in_z3,
            Activity.time_in_z4,
            Activity.time_in_z5,
            ActivityMetrics.avg_gradient_pct,
            ActivityMetrics.max_hrr_60s,
            ActivityMetrics.fat_g,
            ActivityMetrics.carb_g,
        )
        .outerjoin(ActivityMetrics, ActivityMetrics.activity_id == Activity.activity_id)
        .where(Activity.sport.op("~*")(CYCLING_SPORT_REGEX))
        .where(~Activity.sport.op("~*")(EBIKE_SPORT_PATTERN))
        .order_by(Activity.date, Activity.activity_id)
    ).all()

    # ── Pokrytí a nezáměrná Z3 ────────────────────────────────────────────
    # Obojí stojí na předpočítaných tabulkách; tady se jen překládá.
    coverage = repo.read_hr_coverage(
        session, activity_ids=[r.activity_id for r in act_rows]
    )

    # Nezáměrná Z3 = čas v Z3 mimo souvislé bloky. Hranice Z3 se odvodí
    # z LTHR lookupem na mřížku prahů – dvě čísla do dotazu, nic k přepočtu.
    threshold_state = repo.read_current_threshold(session)
    lthr = int(threshold_state["lthr_bpm"]) if threshold_state else LTHR_DEFAULT_BPM
    z3_lo = zone_threshold_bpm(lthr, "Z3", HR_BLOCK_THRESHOLDS_BPM)
    z3_hi = zone_threshold_bpm(lthr, "Z4", HR_BLOCK_THRESHOLDS_BPM)
    blocks = repo.read_block_totals(session, [z3_lo, z3_hi], Z3_BRIDGE_TOLERANCE_S)

    def _z3_unintentional(activity_id: str) -> int | None:
        per_threshold = blocks.get(activity_id)
        if not per_threshold:
            return None       # bloky ještě spočítané nejsou – to není nula
        return unintentional_z3_s(
            {t: v["total_s"] for t, v in per_threshold.items()},
            {t: v["long_s"] for t, v in per_threshold.items()},
            z3_lo,
            z3_hi,
        )

    activities = [
        DashboardActivity(
            id=r.activity_id,
            d=r.date,
            z=[
                _zero(r.time_in_z1),
                _zero(r.time_in_z2),
                _zero(r.time_in_z3),
                _zero(r.time_in_z4),
                _zero(r.time_in_z5),
            ],
            up=_num(r.uphill_minutes),
            down=_num(r.downhill_minutes),
            flat=_num(r.flat_minutes),
            asc=_num(r.ascent_m),
            grad=_num(r.avg_gradient_pct),
            hrr=_num(r.max_hrr_60s),
            km=_num(r.distance_km),
            dur=_num(r.duration_minutes),
            kcal=_num(r.calories),
            fat=_num(r.fat_g),
            carb=_num(r.carb_g),
            z3u=_z3_unintentional(r.activity_id),
            cov=_coverage(coverage.get(r.activity_id)),
        )
        for r in act_rows
    ]

    rides = [
        DashboardRide(
            id=r.activity_id,
            d=r.date,
            dur=_num(r.duration_minutes, 0.0),
            km=_num(r.distance_km, 0.0),
            avg=_num(r.avg_hr),
            max=_num(r.max_hr),
            asc=_num(r.ascent_m, 0.0),
            trimp=_num(r.total_trimp, 0.0),
            kcal=_num(r.calories),
            z=[
                _zero(r.time_in_z1),
                _zero(r.time_in_z2),
                _zero(r.time_in_z3),
                _zero(r.time_in_z4),
                _zero(r.time_in_z5),
            ],
            cov=_coverage(coverage.get(r.activity_id)),
        )
        for r in reversed(act_rows[-RIDES_LIMIT:])
    ]

    # ── Dnešek + dohledaná biometrie ──────────────────────────────────────
    today_row = session.scalars(
        select(DailyMetrics).order_by(DailyMetrics.date.desc()).limit(1)
    ).first()

    today = (
        DashboardToday.model_validate(today_row)
        if today_row is not None
        else None
    )

    last_known = {
        "hrv": _biometric(session, DailyMetrics.hrv_last_night, DailyMetrics.hrv_weekly_avg),
        "rhr": _biometric(session, DailyMetrics.rhr_day, DailyMetrics.rhr_baseline_14d),
        "sleep": _biometric(
            session, DailyMetrics.sleep_duration_min, DailyMetrics.sleep_need_min
        ),
    }

    return DashboardOut(
        generated_at=datetime.now(),
        today=today,
        last_known=last_known,
        days=days,
        rides=rides,
        activities=activities,
    )
