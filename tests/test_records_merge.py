"""
Slučování fragmentů vteřinových dat.

Strava exporty rozdělují jednu vteřinu do několika record zpráv, každou
s jinou podmnožinou polí:

    15:44:45  {distance: 38.24}
    15:44:45  {speed: 0.018, position_lat: …}
    15:44:45  {heart_rate: 88}

Nejsou to duplicity, ale fragmenty téhož vzorku. Naivní „ponech poslední
řádek" by z té vteřiny nechalo jen tep a zbytek přepsalo NULLy — proto
tenhle test existuje.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import delete

from src.db import repository as repo
from src.db.models import Record

TEST_ACTIVITY = "__test_fragments__"
T0 = datetime(2024, 9, 27, 15, 44, 45)


@pytest.fixture
def clean_activity(session):
    def _clean():
        session.execute(delete(Record).where(Record.activity_id == TEST_ACTIVITY))
        session.commit()

    _clean()
    yield
    _clean()


def _row(ts: datetime, **fields):
    base = {c: None for c in repo.RECORD_COLUMNS}
    base["activity_id"] = TEST_ACTIVITY
    base["timestamp"] = ts.isoformat()
    base.update(fields)
    return base


def test_fragments_of_one_second_are_merged(session, clean_activity):
    """Tři fragmenty jedné vteřiny musí dát jeden úplný řádek."""
    rows = [
        _row(T0, distance=38.24),
        _row(T0, speed=0.018, position_lat=58.63),
        _row(T0, heart_rate=88.0),
    ]
    repo.copy_records(session, rows)
    session.commit()

    df = repo.read_records(session, TEST_ACTIVITY)
    assert len(df) == 1, "Jedna vteřina = jeden řádek"

    row = df.iloc[0]
    assert row["distance"] == pytest.approx(38.24)
    assert row["speed"] == pytest.approx(0.018)
    assert row["position_lat"] == pytest.approx(58.63)
    assert row["heart_rate"] == pytest.approx(88.0)


def test_newer_value_wins_on_reimport(session, clean_activity):
    """Při skutečném re-importu má vyhrát novější hodnota, ne ta první."""
    repo.copy_records(session, [_row(T0, heart_rate=120.0)])
    session.commit()
    repo.copy_records(session, [_row(T0, heart_rate=130.0)])
    session.commit()

    df = repo.read_records(session, TEST_ACTIVITY)
    assert len(df) == 1
    assert df.iloc[0]["heart_rate"] == pytest.approx(130.0)


def test_null_does_not_overwrite_existing_value(session, clean_activity):
    """
    Fragment bez dané hodnoty nesmí přepsat dřívější měření NULLem –
    přesně tohle byla ta chyba.
    """
    repo.copy_records(
        session,
        [
            _row(T0, heart_rate=88.0, speed=3.5),
            _row(T0, distance=100.0),  # nese jen vzdálenost
        ],
    )
    session.commit()

    row = repo.read_records(session, TEST_ACTIVITY).iloc[0]
    assert row["heart_rate"] == pytest.approx(88.0)
    assert row["speed"] == pytest.approx(3.5)
    assert row["distance"] == pytest.approx(100.0)


def test_distinct_seconds_stay_separate(session, clean_activity):
    """Slučování nesmí spojit dvě různé vteřiny."""
    rows = [
        _row(T0, heart_rate=88.0),
        _row(T0 + timedelta(seconds=1), heart_rate=90.0),
        _row(T0 + timedelta(seconds=2), heart_rate=92.0),
    ]
    repo.copy_records(session, rows)
    session.commit()

    df = repo.read_records(session, TEST_ACTIVITY).sort_values("timestamp")
    assert len(df) == 3
    assert list(df["heart_rate"]) == [88.0, 90.0, 92.0]


def test_text_and_boolean_columns_merge(session, clean_activity):
    """Sloučení musí fungovat i pro nečíselné sloupce."""
    repo.copy_records(
        session,
        [
            _row(T0, hr_zone="Z2"),
            _row(T0, is_active=True),
            _row(T0, trimp_increment=0.0123),
        ],
    )
    session.commit()

    row = repo.read_records(session, TEST_ACTIVITY).iloc[0]
    assert row["hr_zone"] == "Z2"
    assert bool(row["is_active"]) is True  # pandas vrací numpy.bool_
    assert row["trimp_increment"] == pytest.approx(0.0123)
