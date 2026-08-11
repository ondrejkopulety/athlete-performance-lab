"""
FastAPI aplikace
================

    uvicorn src.api.app:app --reload

Endpointy jsou read-only nad databází (kromě POST /api/sync/run, který
spouští pipeline na pozadí). Výpočty se tady nedějí – API jen servíruje,
co spočítala pipeline, takže odpovědi jsou v jednotkách milisekund.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import API_CORS_ORIGINS, SYNC_CRON_HOUR  # noqa: E402
from src.api.routers import activities, coach, daily, dashboard, sync  # noqa: E402
from src.api.schemas import HealthResponse  # noqa: E402
from src.db.models import Activity, DailyMetrics  # noqa: E402
from src.db.session import check_connection, get_session  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Noční sync přes APScheduler; vypne se prázdným SYNC_CRON_HOUR."""
    global _scheduler
    if SYNC_CRON_HOUR is not None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            from src.api.routers.sync import run_pipeline_job

            _scheduler = BackgroundScheduler(timezone="Europe/Prague")
            _scheduler.add_job(
                run_pipeline_job,
                CronTrigger(hour=SYNC_CRON_HOUR, minute=0),
                id="nightly_sync",
                max_instances=1,       # souběžné běhy by se přetahovaly o data
                coalesce=True,         # po výpadku doběhne jednou, ne pětkrát
            )
            _scheduler.start()
            log.info("Noční sync naplánován na %02d:00.", SYNC_CRON_HOUR)
        except Exception:  # noqa: BLE001
            log.exception("Scheduler se nepodařilo spustit – API běží bez něj.")

    if not check_connection():
        log.warning("Databáze neodpovídá. Běží `docker compose up -d db`?")

    yield

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)


app = FastAPI(
    title="Garmin Training Analytics API",
    description=(
        "Tréninková zátěž (PMC), regenerace, biometrie a kontext pro AI trenéra."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(daily.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(activities.router, prefix="/api")
app.include_router(coach.router, prefix="/api")
app.include_router(sync.router, prefix="/api")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(session: Session = Depends(get_session)) -> HealthResponse:
    """Stav databáze a rozsah dat – vhodné pro monitoring i pro rychlou kontrolu."""
    db_ok = check_connection()
    if not db_ok:
        return HealthResponse(status="degraded", database=False, activities=0, days=0)

    return HealthResponse(
        status="ok",
        database=True,
        activities=session.scalar(select(func.count()).select_from(Activity)) or 0,
        days=session.scalar(select(func.count()).select_from(DailyMetrics)) or 0,
        last_activity_date=session.scalar(select(func.max(Activity.date))),
        last_metrics_date=session.scalar(select(func.max(DailyMetrics.date))),
    )
