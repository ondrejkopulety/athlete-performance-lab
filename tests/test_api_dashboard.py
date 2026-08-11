"""
Endpoint /api/dashboard – kontrola tvaru a věrnosti dat.

Endpoint nic nepočítá, takže se testuje jediné, co se může rozbít:
že sloupce sedí na databázi a že se do jízd nevloudí běh nebo plavání.
"""

from __future__ import annotations

import re
from datetime import date

import pytest
from sqlalchemy import select

from src.analytics.exports import CYCLING_SPORT_PATTERN
from src.api.routers.dashboard import RIDES_LIMIT, dashboard
from src.db.models import Activity, DailyMetrics


@pytest.fixture
def payload(session):
    return dashboard(session)


def test_days_kopiruji_daily_metrics(session, payload):
    """Poslední řádek řady musí sedět na poslední den v databázi."""
    if not payload.days:
        pytest.skip("Prázdná databáze")

    row = session.scalars(
        select(DailyMetrics).order_by(DailyMetrics.date.desc()).limit(1)
    ).first()

    date_s, ctl, atl, tsb, trimp, _strain, ready, acwr = payload.days[-1]
    assert date_s == row.date.isoformat()
    assert ctl == pytest.approx(row.ctl or 0.0, abs=0.01)
    assert atl == pytest.approx(row.atl or 0.0, abs=0.01)
    assert tsb == pytest.approx(row.tsb or 0.0, abs=0.01)
    assert trimp == pytest.approx(row.trimp or 0.0, abs=0.01)
    assert ready == (None if row.readiness_score is None else pytest.approx(row.readiness_score, abs=0.01))
    assert acwr == (None if row.acwr is None else pytest.approx(row.acwr, abs=0.01))


def test_days_jsou_serazene_a_bez_der_v_grafovych_radach(payload):
    """Řady, které se kreslí jako spojitá čára, nesmí obsahovat None."""
    dates = [d[0] for d in payload.days]
    assert dates == sorted(dates)
    for row in payload.days:
        assert None not in row[1:5], f"díra v grafové řadě {row[0]}"


def test_jizdy_jsou_jen_cyklistika(session, payload):
    """Stejný filtr jako cycling_summary.csv – žádný běh v seznamu jízd."""
    assert len(payload.rides) <= RIDES_LIMIT
    pattern = re.compile(CYCLING_SPORT_PATTERN, re.IGNORECASE)

    for ride in payload.rides:
        sport = session.get(Activity, ride.id).sport
        assert pattern.search(sport or ""), f"{sport} není cyklistika"

    # Nejnovější jízda první.
    assert [r.d for r in payload.rides] == sorted((r.d for r in payload.rides), reverse=True)


def test_aktivity_nesou_podklady_pro_grafy(payload):
    """Zóny, stoupání i zotavovací tep jdou z jednoho seznamu."""
    if not payload.activities:
        pytest.skip("Prázdná databáze")

    assert all(len(a.z) == 5 for a in payload.activities)
    dates = [a.d for a in payload.activities]
    assert dates == sorted(dates)
    # Aspoň někde musí být stoupání a HRR, jinak by grafy byly prázdné.
    assert any(a.up for a in payload.activities)
    assert any(a.hrr for a in payload.activities)


def test_biometrie_ma_dohledanou_hodnotu(payload):
    """Poslední den bývá bez ranní biometrie – endpoint musí nabídnout
    poslední naměřenou hodnotu i s datem."""
    last_day = date.fromisoformat(payload.days[-1][0])
    for key in ("hrv", "rhr", "sleep"):
        entry = payload.last_known[key]
        if entry is None:
            continue
        assert entry.value is not None
        assert entry.date <= last_day
