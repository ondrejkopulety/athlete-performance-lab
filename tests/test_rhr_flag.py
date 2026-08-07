"""
Klidový tep relativně k vlastnímu baseline
===========================================

Původně se `high_rhr` vlajka spouštěla při překročení pevných 46 bpm.
Na reálných datech to znamenalo, že hořela 44 % dní – práh totiž ležel
přesně na mediánu. Varování, které svítí každý druhý den, nenese žádnou
informaci a navíc ředilo `illness_warning`, které se skládá ze čtyř vlajek
a spouští se při třech.

Nově se posuzuje elevace nad vlastním 14denním baseline. Testy níže hlídají
jak samotné pravidlo, tak jeho promítnutí do textového doporučení.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import ILLNESS_FLAG_COUNT, RHR_BASELINE_DAYS, RHR_ELEVATION_BPM
from src.analytics.biometrics import compute_illness_warning, compute_recovery
from src.db import repository as repo


# ── Jednotkové testy nad umělými daty ──────────────────────────────────────

def _frame(rhr_values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(rhr_values), freq="D", name="date")
    return pd.DataFrame(
        {
            "rhr_day": rhr_values,
            "trimp": [0.0] * len(rhr_values),
            "strain": [np.nan] * len(rhr_values),
            "pure_recovery_score": [np.nan] * len(rhr_values),
            "hrv_last_night": [np.nan] * len(rhr_values),
            "hrv_weekly_avg": [np.nan] * len(rhr_values),
            "sleep_score_day": [np.nan] * len(rhr_values),
            "sleep_duration_min": [np.nan] * len(rhr_values),
        },
        index=idx,
    )


def test_stable_rhr_never_raises_flag():
    """Stabilní klidový tep nesmí spustit vlajku, ať je jakkoli vysoký."""
    daily = compute_illness_warning(_frame([55.0] * 30))
    assert not daily["stress_flags"].str.contains("high_rhr").any()


def test_elevation_above_threshold_raises_flag():
    """Skok o RHR_ELEVATION_BPM nad ustálený baseline vlajku spustí."""
    values = [45.0] * 20 + [45.0 + RHR_ELEVATION_BPM + 1]
    daily = compute_illness_warning(_frame(values))
    assert daily["stress_flags"].str.contains("high_rhr").iloc[-1]
    assert not daily["stress_flags"].str.contains("high_rhr").iloc[:-1].any()


def test_elevation_just_below_threshold_does_not_raise():
    values = [45.0] * 20 + [45.0 + RHR_ELEVATION_BPM - 1]
    daily = compute_illness_warning(_frame(values))
    assert not daily["stress_flags"].str.contains("high_rhr").iloc[-1]


def test_absolute_value_alone_is_irrelevant():
    """
    Jádro změny: rozhoduje odchylka, ne absolutní číslo. Sportovec
    s klidovým tepem 60 nesmí mít trvale zapnutou vlajku jen proto,
    že má vyšší tep než někdo jiný.
    """
    high_but_stable = compute_illness_warning(_frame([60.0] * 30))
    low_but_spiking = compute_illness_warning(_frame([40.0] * 20 + [48.0]))

    assert not high_but_stable["stress_flags"].str.contains("high_rhr").any()
    assert low_but_spiking["stress_flags"].str.contains("high_rhr").iloc[-1]


def test_baseline_excludes_today():
    """
    Baseline se počítá z předchozích dní. Kdyby zahrnoval dnešek,
    zvýšená hodnota by si nadzvedla vlastní referenci a signál by zeslabila.
    """
    values = [40.0] * 20 + [60.0]
    daily = compute_illness_warning(_frame(values))
    elevation = daily["rhr_elevation_bpm"].iloc[-1]
    # Baseline musí zůstat na 40 → elevace přesně 20, ne méně
    assert elevation == pytest.approx(20.0, abs=0.05)


def test_flag_needs_enough_history():
    """Bez dostatečné historie se vlajka nespouští (žádné falešné poplachy)."""
    daily = compute_illness_warning(_frame([45.0, 70.0]))
    assert not daily["stress_flags"].str.contains("high_rhr").any()


# ── Testy nad reálnými daty v databázi ─────────────────────────────────────

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
    return df[df["rhr_day"].notna()]


def test_flag_matches_stored_elevation(daily):
    """Vlajka v databázi musí odpovídat uložené elevaci."""
    has_flag = daily["stress_flags"].fillna("").str.contains("high_rhr")
    elevation = pd.to_numeric(daily["rhr_elevation_bpm"], errors="coerce")

    known = elevation.notna()
    expected = elevation[known] >= RHR_ELEVATION_BPM
    assert (has_flag[known] == expected).all()


def test_baseline_is_plausible(daily):
    """Baseline musí ležet v rozumném rozpětí kolem skutečných hodnot."""
    base = pd.to_numeric(daily["rhr_baseline_14d"], errors="coerce").dropna()
    rhr = pd.to_numeric(daily["rhr_day"], errors="coerce").dropna()
    assert base.min() >= 30 and base.max() <= 100
    assert abs(base.median() - rhr.median()) < 3


def test_flag_is_now_informative(daily):
    """
    Vlajka musí být výjimkou, ne pravidlem. Nad ~20 % dní by opět
    přestala nést informaci a ředila by illness_warning.
    """
    rate = daily["stress_flags"].fillna("").str.contains("high_rhr").mean()
    assert rate < 0.20, f"high_rhr hoří {rate:.0%} dní – práh je zase moc citlivý"


def test_illness_warning_still_requires_enough_flags(daily):
    flagged = daily[daily["illness_warning"] == True]  # noqa: E712
    if flagged.empty:
        return
    counts = pd.to_numeric(flagged["stress_flag_count"], errors="coerce")
    assert counts.min() >= ILLNESS_FLAG_COUNT


def test_coach_advice_reports_elevation_when_flagged(daily):
    """Text doporučení musí zmínit zvýšený tep právě tehdy, když vlajka hoří."""
    advice = daily["coach_advice"].fillna("")
    elevation = pd.to_numeric(daily["rhr_elevation_bpm"], errors="coerce")

    known = elevation.notna()
    mentions = advice[known].str.contains("Zvýšený RHR")
    expected = elevation[known] >= RHR_ELEVATION_BPM
    assert (mentions == expected).all()


def test_coach_advice_shows_baseline_context(daily):
    """
    Hlášení musí nést i referenci – „48 bpm" samo o sobě nic neříká,
    „48 při běžných 43" ano.
    """
    mentioning = daily[daily["coach_advice"].fillna("").str.contains("Zvýšený RHR")]
    if mentioning.empty:
        pytest.skip("Žádný den se zvýšeným tepem")
    assert mentioning["coach_advice"].str.contains("nad 14denním průměrem").all()


def test_recovery_component_uses_same_baseline_window(daily):
    """
    pure_recovery_score počítá RHR složku proti stejnému oknu, takže
    obě cesty musí vidět tentýž baseline.
    """
    assert RHR_BASELINE_DAYS == 14
    assert daily["rhr_baseline_14d"].notna().any()
