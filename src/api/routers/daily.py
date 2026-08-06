"""Denní metriky a PMC časová řada – hlavní zdroj dat pro dashboard."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.analytics.calendar import today_local
from src.api.schemas import DailyMetricsOut, PmcPoint
from src.db.models import DailyMetrics
from src.db.session import get_session

router = APIRouter(tags=["daily"])


@router.get("/daily", response_model=list[DailyMetricsOut])
def list_daily(
    date_from: date | None = Query(None, alias="from"),
    date_to: date | None = Query(None, alias="to"),
    session: Session = Depends(get_session),
) -> list[DailyMetrics]:
    """Denní metriky v zadaném rozsahu (výchozí: posledních 90 dní)."""
    date_to = date_to or today_local()
    date_from = date_from or (date_to - timedelta(days=90))
    if date_from > date_to:
        raise HTTPException(status_code=400, detail="'from' musí být <= 'to'")

    stmt = (
        select(DailyMetrics)
        .where(DailyMetrics.date >= date_from, DailyMetrics.date <= date_to)
        .order_by(DailyMetrics.date)
    )
    return list(session.scalars(stmt))


@router.get("/daily/latest", response_model=DailyMetricsOut)
def latest_daily(session: Session = Depends(get_session)) -> DailyMetrics:
    """
    Nejnovější den s metrikami.

    Kalendář vždy končí dneškem, takže tohle je dnešní stav i ve dnech,
    kdy se netrénovalo – právě proto se ranní biometrie neztrácí.
    """
    row = session.scalars(
        select(DailyMetrics).order_by(DailyMetrics.date.desc()).limit(1)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Žádné denní metriky v databázi")
    return row


@router.get("/pmc", response_model=list[PmcPoint])
def pmc(
    days: int = Query(180, ge=7, le=3650),
    session: Session = Depends(get_session),
) -> list[DailyMetrics]:
    """Performance Management Chart – CTL / ATL / TSB / ACWR."""
    start = today_local() - timedelta(days=days)
    stmt = (
        select(DailyMetrics)
        .where(DailyMetrics.date >= start)
        .order_by(DailyMetrics.date)
    )
    return list(session.scalars(stmt))
