"""
Odhad prahového tepu z terénních dat
=====================================

DFA-alpha1 měl dodávat hranice zón z vlastních měření, ale na těchto datech
nefunguje: R-R intervaly má jen 73 z 866 aktivit a i tam zůstává alpha1
systematicky nad 1,2 i při tepu nad prahem.

Terénní odhad `0,95 × nejlepší 20minutový průměr tepu` naproti tomu funguje
nad každou aktivitou s tepem a na reálných datech vychází 172 bpm – shodně
s laktátovým testem.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import (
    LTHR_FACTOR,
    LTHR_TEST_MINUTES,
    LTHR_WINDOW_DAYS,
    ZONES,
)
from src.analytics.activity import compute_best_hr_windows
from src.db import repository as repo


def _records(hr_profile: list[tuple[float, int]]) -> pd.DataFrame:
    """Postaví vteřinovou řadu z dvojic (tep, minuty)."""
    hrs: list[float] = []
    for hr, minutes in hr_profile:
        hrs.extend([hr] * minutes * 60)
    ts = pd.date_range("2026-06-01 10:00", periods=len(hrs), freq="1s")
    return pd.DataFrame({"timestamp": ts, "heart_rate": hrs})


def test_finds_the_hard_block():
    """Nejlepší 20min okno musí najít tvrdý blok, ne průměr celé jízdy."""
    rec = _records([(120, 40), (175, 25), (120, 40)])
    best = compute_best_hr_windows(rec, [20])
    assert best["best_20min_hr"] == pytest.approx(175, abs=1)


def test_longer_window_cannot_exceed_shorter():
    """60minutový průměr nemůže být vyšší než 20minutový."""
    rec = _records([(130, 30), (180, 25), (130, 30)])
    best = compute_best_hr_windows(rec, [20, 60])
    assert best["best_20min_hr"] >= best["best_60min_hr"]


def test_short_activity_yields_nothing():
    """Z desetiminutovky nelze odhadnout dvacetiminutový výkon."""
    best = compute_best_hr_windows(_records([(170, 10)]), [20])
    assert best["best_20min_hr"] is None


def test_empty_input_is_safe():
    assert compute_best_hr_windows(pd.DataFrame(), [20])["best_20min_hr"] is None


def test_window_is_long_enough_to_catch_a_hard_effort():
    """
    Odhad je dolní mez – potřebuje, aby v okně opravdu proběhlo maximální
    úsilí. Při 90 dnech vycházelo 164 bpm místo 172, protože v okně žádné
    nebylo.
    """
    assert LTHR_WINDOW_DAYS >= 180


# ── Nad reálnými daty ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily():
    from src.db.session import SessionLocal, check_connection

    if not check_connection():
        pytest.skip("Databáze neběží")
    s = SessionLocal()
    try:
        df = repo.read_daily_metrics(s)
    finally:
        s.close()
    if df.empty:
        pytest.skip("Žádné denní metriky")
    return df


def test_estimate_matches_lactate_test(daily):
    """
    Nezávislé ověření nastavených zón: odhad z terénních dat musí sedět
    s hranicí Z3/Z4 z laktátového testu. Pokud se rozejde o víc než pár
    bpm, buď se posunula forma, nebo je něco špatně ve výpočtu.
    """
    values = pd.to_numeric(daily["lthr_estimate"], errors="coerce").dropna()
    if values.empty:
        pytest.skip("lthr_estimate zatím nespočítané")

    configured = ZONES["Z4"][0]
    recent = values.tail(200).median()
    assert abs(recent - configured) <= 5, (
        f"Odhad {recent:.0f} bpm vs nastavená hranice {configured} bpm – "
        "zkontroluj, jestli se neposunul práh nebo výpočet"
    )


def test_estimate_is_physiologically_plausible(daily):
    values = pd.to_numeric(daily["lthr_estimate"], errors="coerce").dropna()
    if values.empty:
        pytest.skip("lthr_estimate zatím nespočítané")
    assert values.min() >= 120
    assert values.max() <= 199  # nemůže překročit maximální tep


def test_estimate_derives_from_best_effort(daily):
    """lthr_estimate musí být LTHR_FACTOR × nějaký reálně dosažený výkon."""
    from src.db.session import SessionLocal

    s = SessionLocal()
    try:
        best = pd.read_sql(
            f"SELECT max(best_{LTHR_TEST_MINUTES}min_hr) AS m FROM activity_metrics",
            s.connection(),
        )
    finally:
        s.close()
    if best["m"].isna().all():
        pytest.skip("Žádné nejlepší úseky")

    ceiling = float(best["m"].iloc[0]) * LTHR_FACTOR
    values = pd.to_numeric(daily["lthr_estimate"], errors="coerce").dropna()
    assert values.max() <= ceiling + 0.51  # tolerance zaokrouhlení
