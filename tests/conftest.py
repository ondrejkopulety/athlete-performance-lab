"""Sdílené fixtures. Testy, které potřebují databázi, se přeskočí,
pokud neběží (docker compose up -d db)."""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.db.session import SessionLocal, check_connection  # noqa: E402


@pytest.fixture(scope="session")
def db_available() -> bool:
    return check_connection()


@pytest.fixture
def session(db_available):
    if not db_available:
        pytest.skip("Databáze neběží – spusť `docker compose up -d db`")
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
