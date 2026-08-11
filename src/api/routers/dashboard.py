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

from src.analytics.exports import CYCLING_SPORT_PATTERN
from src.api.schemas import (
    DashboardActivity,
    DashboardBiometric,
    DashboardOut,
    DashboardRide,
    DashboardToday,
)
from src.db.models import Activity, ActivityMetrics, DailyMetrics
from src.db.session import get_session

router = APIRouter(tags=["dashboard"])

# Kolik posledních jízd se vypisuje v seznamu pod grafy.
RIDES_LIMIT = 5


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
    # množinu jízd, která je i v cycling_summary.csv.
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
            Activity.time_in_z1,
            Activity.time_in_z2,
            Activity.time_in_z3,
            Activity.time_in_z4,
            Activity.time_in_z5,
            ActivityMetrics.avg_gradient_pct,
            ActivityMetrics.max_hrr_60s,
        )
        .outerjoin(ActivityMetrics, ActivityMetrics.activity_id == Activity.activity_id)
        .where(Activity.sport.op("~*")(CYCLING_SPORT_PATTERN))
        .order_by(Activity.date, Activity.activity_id)
    ).all()

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
            asc=_num(r.ascent_m),
            grad=_num(r.avg_gradient_pct),
            hrr=_num(r.max_hrr_60s),
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
