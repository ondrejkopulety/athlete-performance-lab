"""
Endpointy /api/hr/* nad reálnou databází.

Testuje se to, co se v čistě jednotkovém testu ověřit nedá:

  • agregace přes období, ve kterém žádná jízda nemá dané okno, jde ven jako
    ``null`` – ne jako nula
  • filtr "jen úplná data" opravdu ubírá jízdy a počet vyloučených sedí
  • změna LTHR **nesahá na uložená data** (bod, kvůli kterému jsou bloky na
    mřížce absolutních prahů)
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import func, select

from config.settings import HR_COVERAGE_WARN_PCT
from src.api.routers.dashboard import dashboard
from src.api.routers.hr import get_threshold, hr_blocks, hr_curve, put_threshold
from src.api.schemas import ThresholdIn
from src.db.models import ActivityHrBlocks, ActivityHrCurve


@pytest.fixture
def period(session):
    """Posledních 90 dní."""
    if session.scalar(select(func.count()).select_from(ActivityHrCurve)) == 0:
        pytest.skip("Tepová křivka není spočítaná – spusť `python -m src.physio.cli hr`")
    today = date.today()
    return today - timedelta(days=90), today


# ═══════════════════════════════════════════════════════════════════════════
# Chybějící okno není nula
# ═══════════════════════════════════════════════════════════════════════════

def test_kratke_obdobi_bez_hodinoveho_okna_vraci_null(session):
    """
    Jednodenní okno kolem jízdy 12. 7. 2026: ta má nejdelší souvislý úsek dat
    2 710 s, takže hodinový bod křivky nemá. Musí přijít ``null`` – kdyby
    přišla nula, graf by čáru stáhl na osu a tabulka by tvrdila "0 tepů".
    """
    out = hr_curve(
        session,
        since=date(2026, 7, 12),
        until=date(2026, 7, 12),
        complete_only=False,
        compare="none",
    )
    points = {p.d: p.hr for p in out.period.points}
    if not any(v is not None for v in points.values()):
        pytest.skip("Jízda 12. 7. 2026 v databázi není")

    assert points[3600] is None
    assert points[1200] is not None       # kratší okno vyjde


def test_obdobi_bez_jizd_nevraci_nuly(session):
    """Období, ve kterém se nejelo: samé null, žádná nula."""
    out = hr_curve(
        session,
        since=date(1990, 1, 1),
        until=date(1990, 1, 31),
        complete_only=False,
        compare="none",
    )
    assert all(p.hr is None for p in out.period.points)
    assert out.period.last_max_effort is None

    blocks = hr_blocks(
        session, since=date(1990, 1, 1), until=date(1990, 1, 31), complete_only=False
    )
    # None, ne 0: v období není jízda, ze které by nula mohla vzniknout.
    assert blocks.longest_block_s is None
    assert blocks.rides_total == 0


# ═══════════════════════════════════════════════════════════════════════════
# Filtr "jen úplná data"
# ═══════════════════════════════════════════════════════════════════════════

def test_filtr_ubira_jizdy_a_hlasi_kolik(session, period):
    since, until = period
    filtered = hr_curve(session, since=since, until=until, complete_only=True, compare="none")
    everything = hr_curve(session, since=since, until=until, complete_only=False, compare="none")

    if filtered.period.rides_total == 0:
        pytest.skip("V posledních 90 dnech nejsou jízdy")

    assert everything.period.rides_excluded == 0
    assert filtered.period.rides_excluded >= 0
    assert filtered.period.rides_excluded <= filtered.period.rides_total
    assert filtered.coverage_warn_pct == HR_COVERAGE_WARN_PCT

    # Vyloučit se smí jen dolů: filtrovaná maxima nesmí být vyšší.
    for a, b in zip(filtered.period.points, everything.period.points):
        if a.hr is not None and b.hr is not None:
            assert a.hr <= b.hr


def test_pokryti_u_jizdy_je_verdikt_ne_tri_procenta(session):
    payload = dashboard(session)
    with_coverage = [r for r in payload.rides if r.cov is not None]
    if not with_coverage:
        pytest.skip("Pokrytí není spočítané")

    for ride in with_coverage:
        assert isinstance(ride.cov.ok, bool)
        # Vysvětlení jen tam, kde je co vysvětlovat – žádné "100 % OK".
        assert (ride.cov.note is None) == ride.cov.ok
        if not ride.cov.ok:
            assert ride.cov.pct is not None and ride.cov.pct < HR_COVERAGE_WARN_PCT


def test_ridky_zapis_sam_o_sobe_nevaruje(session):
    """
    Smart Recording má nízkou hustotu vzorků, ale plné pokrytí. Takové jízdy
    v databázi jsou (341 z 799) a odznak dostat nesmí.
    """
    payload = dashboard(session)
    sparse_but_full = [
        a
        for a in payload.activities
        if a.cov and a.cov.density is not None and a.cov.density < 60 and a.cov.ok
    ]
    if not sparse_but_full:
        pytest.skip("Žádná jízda s řídkým zápisem a plným pokrytím")

    for activity in sparse_but_full:
        assert activity.cov.note is None


# ═══════════════════════════════════════════════════════════════════════════
# Změna prahu nepřepočítává data
# ═══════════════════════════════════════════════════════════════════════════

def test_zmena_lthr_nesahne_na_ulozene_bloky(session):
    """
    Kvůli tomuhle jsou bloky uložené na mřížce absolutních prahů. Změna LTHR
    smí změnit jen to, na který řádek se panel ptá – ani jeden uložený řádek
    se nesmí dotknout.
    """
    fingerprint = select(
        func.count(),
        func.sum(ActivityHrBlocks.longest_block_s),
        func.sum(ActivityHrBlocks.total_time_s),
        func.max(ActivityHrBlocks.computed_at),
    )
    before = session.execute(fingerprint).one()
    original = get_threshold(session)

    try:
        changed = put_threshold(
            ThresholdIn(lthr_bpm=original.lthr_bpm + 6, hr_max_bpm=original.hr_max_bpm),
            session,
        )
        session.flush()

        after = session.execute(fingerprint).one()
        assert after == before, "změna LTHR přepsala uložené bloky"

        # A přitom se opravdu změnil práh, na který se panel ptá.
        panel = hr_blocks(
            session, since=date(2026, 1, 1), until=date(2026, 12, 31), complete_only=False
        )
        assert changed.lthr_bpm == original.lthr_bpm + 6
        assert panel.lthr_bpm == changed.lthr_bpm
    finally:
        session.rollback()


def test_prah_hlasi_svoje_stari(session):
    out = get_threshold(session)
    assert out.days_ago >= 0
    assert out.stale == (out.days_ago > out.stale_after_days)
    assert set(out.zone_thresholds) >= {"Z2", "Z3", "Z4", "Z5"}
