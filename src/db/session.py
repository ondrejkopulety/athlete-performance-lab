"""
Engine, session factory a FastAPI dependency.

Jediné místo, kde se vytváří spojení do DB – CLI, migrace i API sdílí
stejný engine, takže se konfigurace nikde nerozjede.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import DATABASE_URL  # noqa: E402

engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # přežije restart DB kontejneru bez pádu API
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Session pro CLI a pipeline: závěrečný commit při úspěchu, rollback při chybě,
    vždy ``close``.

    POZOR: není to jedna atomická transakce přes celý blok. Dlouhé kroky
    (LOAD commituje po 25 aktivitách, ANALYZE a STRAVA commitují samy)
    zapisují průběžně – jedna transakce přes 884 aktivit by držela miliony
    řádků a při pádu by se zahodil několikaminutový běh. Závěrečný commit
    tady tedy jen dorovná to, co ještě viselo od posledního kroku; kroky,
    které doběhly, zůstávají zapsané i když pozdější krok spadne.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency (read-only endpointy si commit neřeší)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_connection() -> bool:
    """True, pokud DB odpovídá. Používá /health a CLI preflight."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
