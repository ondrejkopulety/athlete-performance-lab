"""
Spuštění pipeline z API.

Sync běží na pozadí a je chráněný zámkem – dva souběžné běhy by si
navzájem přepisovaly aktivity a stahovaly stejná data z Garminu dvakrát,
což je nejrychlejší cesta k HTTP 429 a hodinovému banu.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from src.api.schemas import SyncStatusOut, SyncTriggerOut
from src.db import repository as repo
from src.db.session import get_session, session_scope

router = APIRouter(prefix="/sync", tags=["sync"])
log = logging.getLogger("api.sync")

_lock = threading.Lock()
SYNC_STATE_KEY = "last_pipeline_run"


def is_running() -> bool:
    return _lock.locked()


def run_pipeline_job(skip_download: bool = False) -> None:
    """Tělo běhu na pozadí. Výsledek se zapisuje do sync_state."""
    if not _lock.acquire(blocking=False):
        log.warning("Pipeline už běží – nový požadavek ignorován.")
        return

    started = datetime.now()
    outcome: dict = {"started_at": started.isoformat(timespec="seconds")}
    try:
        # Import uvnitř funkce: garmin_sync tahá těžké závislosti a při
        # startu API je nepotřebujeme.
        from src.pipeline import run_full_pipeline

        result = run_full_pipeline(skip_download=skip_download)
        outcome.update(result)
        outcome["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - stav musí přežít i pád
        log.exception("Pipeline selhala")
        outcome["status"] = "error"
        outcome["error"] = str(exc)
    finally:
        outcome["finished_at"] = datetime.now().isoformat(timespec="seconds")
        outcome["duration_s"] = round((datetime.now() - started).total_seconds(), 1)
        try:
            with session_scope() as session:
                repo.set_state(session, SYNC_STATE_KEY, outcome)
        except Exception:  # noqa: BLE001
            log.exception("Nepodařilo se uložit stav běhu")
        _lock.release()


@router.post("/run", response_model=SyncTriggerOut)
def trigger_sync(
    background: BackgroundTasks,
    skip_download: bool = False,
) -> SyncTriggerOut:
    """Spustí pipeline na pozadí (stažení → načtení FIT → analytika)."""
    if is_running():
        return SyncTriggerOut(accepted=False, message="Pipeline už běží.")
    background.add_task(run_pipeline_job, skip_download)
    return SyncTriggerOut(accepted=True, message="Pipeline spuštěna na pozadí.")


@router.get("/status", response_model=SyncStatusOut)
def sync_status(session: Session = Depends(get_session)) -> SyncStatusOut:
    return SyncStatusOut(running=is_running(), last_run=repo.get_state(session, SYNC_STATE_KEY))
