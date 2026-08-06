"""
Endpointy pro AI trenéra.

Zatím jen produkce kontextu – napojení na LLM přijde v dalším kole.
Doporučený postup pro chatbota: statickou část (athlete + metric_glossary)
poslat jako cachovaný prefix system promptu, měnící se `today` a `trends`
až za ním.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config.settings import METRIC_META
from src.coach.context import build_daily_context, build_history_series
from src.db.session import get_session

router = APIRouter(prefix="/coach", tags=["coach"])


@router.get("/context")
def coach_context(
    day: date | None = Query(None, alias="date"),
    session: Session = Depends(get_session),
) -> dict:
    """Strukturovaný denní kontext pro LLM (viz src/coach/context.py)."""
    return build_daily_context(session, day)


@router.get("/history/{metric}")
def coach_history(
    metric: str,
    days: int = Query(90, ge=7, le=1825),
    session: Session = Depends(get_session),
) -> dict:
    """
    Časová řada jedné metriky.

    Určeno jako tool pro chatbota: kontext nese jen 7 a 28denní souhrn,
    tímhle si model doplní historii, na kterou se uživatel zeptá.
    """
    if metric not in METRIC_META:
        raise HTTPException(
            status_code=404,
            detail=f"Neznámá metrika '{metric}'. Dostupné: {', '.join(sorted(METRIC_META))}",
        )
    return build_history_series(session, metric, days=days)


@router.get("/glossary")
def glossary() -> dict:
    """Význam, jednotky a směr všech metrik."""
    return METRIC_META
