"""Seznam aktivit, detail a downsamplovaná vteřinová data pro grafy."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.schemas import ActivityDetailOut, ActivityOut, RecordPoint
from src.db import repository as repo
from src.db.models import Activity, ActivityMetrics, DailyMetrics
from src.db.session import get_session

router = APIRouter(prefix="/activities", tags=["activities"])

# Povolená rozlišení pro downsampling. Whitelist, ne volný text – hodnota
# jde do SQL intervalu a i s parametrizací nemá smysl pouštět dál cokoli.
RESOLUTIONS = {
    "1s": "1 second",
    "5s": "5 seconds",
    "10s": "10 seconds",
    "30s": "30 seconds",
    "1m": "1 minute",
    "5m": "5 minutes",
}


@router.get("", response_model=list[ActivityOut])
def list_activities(
    date_from: date | None = Query(None, alias="from"),
    date_to: date | None = Query(None, alias="to"),
    sport: str | None = None,
    limit: int = Query(50, ge=1, le=2000),
    session: Session = Depends(get_session),
) -> list[Activity]:
    stmt = select(Activity)
    if date_from:
        stmt = stmt.where(Activity.date >= date_from)
    if date_to:
        stmt = stmt.where(Activity.date <= date_to)
    if sport:
        stmt = stmt.where(Activity.sport.ilike(f"%{sport}%"))
    stmt = stmt.order_by(Activity.date.desc(), Activity.activity_id).limit(limit)
    return list(session.scalars(stmt))


@router.get("/{activity_id}", response_model=ActivityDetailOut)
def get_activity(activity_id: str, session: Session = Depends(get_session)) -> dict:
    activity = session.get(Activity, activity_id)
    if activity is None:
        raise HTTPException(status_code=404, detail=f"Aktivita {activity_id} neexistuje")

    payload = {c.name: getattr(activity, c.name) for c in Activity.__table__.columns}
    metrics = session.get(ActivityMetrics, activity_id)
    if metrics is not None:
        payload.update(
            {
                c.name: getattr(metrics, c.name)
                for c in ActivityMetrics.__table__.columns
                # rr_intervals_ms je surový vstup pro DFA – desítky tisíc
                # čísel, která frontend nikdy nepotřebuje
                if c.name not in ("activity_id", "computed_at", "rr_intervals_ms")
            }
        )

    # Whoop strain je denní, ne per-aktivita – ukazuje se v detailu jízdy
    # jako strain CELÉHO dne, protože jinou hodnotu pro jednu jízdu nemáme.
    # Proto se posílá i počet aktivit toho dne: při druhé jízdě už číslo
    # nepatří jen týhle a UI to musí umět přiznat, ne tvářit se přesně.
    daily = session.get(DailyMetrics, activity.date)
    payload["whoop_strain"] = daily.whoop_strain if daily is not None else None
    payload["activities_same_day"] = session.scalar(
        select(func.count()).select_from(Activity).where(Activity.date == activity.date)
    )
    return payload


@router.get("/{activity_id}/records", response_model=list[RecordPoint])
def get_records(
    activity_id: str,
    resolution: str = Query("10s", description="1s | 5s | 10s | 30s | 1m | 5m"),
    session: Session = Depends(get_session),
) -> list[dict]:
    """
    Vteřinová data agregovaná přes TimescaleDB time_bucket.

    Syrová data mají desítky tisíc bodů na aktivitu – posílat je do
    prohlížeče nedává smysl, graf stejně nemá tolik pixelů.
    """
    if resolution not in RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Neplatné rozlišení. Povolené: {', '.join(RESOLUTIONS)}",
        )
    if session.get(Activity, activity_id) is None:
        raise HTTPException(status_code=404, detail=f"Aktivita {activity_id} neexistuje")

    df = repo.downsample_records(session, activity_id, RESOLUTIONS[resolution])
    if df.empty:
        return []
    return df.astype(object).where(df.notna(), None).to_dict("records")
