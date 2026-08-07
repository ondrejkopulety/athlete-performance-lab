"""
Klidový tep jako časová řada
=============================

TRIMP používá Karvonenův poměr (HR − RHR) / (MaxHR − RHR). Dokud byl RHR
pevná konstanta 41 bpm, byla zátěž škálovaná číslem, které neodpovídalo
realitě: skutečný klidový tep se v datech pohybuje 44 → 52 → 47 bpm.

Dvě různá okna slouží dvěma různým úlohám:
  • 14 dní → „je dnešek mimo?" (vlajka, recovery score)
  • 90 dní → „jaká je moje aktuální úroveň?" (škálování TRIMP)

Zdroje se nesmí slévat: Apple Watch a Garmin měří jinak a v datech na
sebe navazují (Apple končí 2025-08-05, Garmin začíná 2025-08-07).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import RHR_BASELINE_LONG_DAYS, RHR_BASELINE_LONG_MIN, RESTING_HR
from src.analytics.activity import compute_trimp_from_records
from src.analytics.biometrics import rhr_baseline_series
from src.db import repository as repo


# ── Baseline ───────────────────────────────────────────────────────────────

def _biometrics(values: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "resting_heart_rate": values, "source": "garmin"})


def test_baseline_tracks_level_not_noise():
    """Jedna špatná noc nesmí baseline utáhnout – proto medián, ne průměr."""
    values = [45.0] * 100
    values[50] = 70.0  # nemoc / alkohol
    idx = pd.date_range("2025-01-01", periods=100, freq="D")

    baseline = rhr_baseline_series(_biometrics(values), idx)
    assert baseline.iloc[-1] == pytest.approx(45.0, abs=0.5)


def test_baseline_follows_genuine_drift():
    """Trvalý posun úrovně se v baseline projevit musí."""
    values = [44.0] * 120 + [52.0] * 120
    idx = pd.date_range("2025-01-01", periods=240, freq="D")

    baseline = rhr_baseline_series(_biometrics(values), idx)
    assert baseline.iloc[100] == pytest.approx(44.0, abs=1.0)
    assert baseline.iloc[-1] == pytest.approx(52.0, abs=1.0)


def test_baseline_needs_minimum_measurements():
    """Bez dost měření zůstává NaN – volající pak sáhne po konstantě."""
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    baseline = rhr_baseline_series(_biometrics([45.0] * 10), idx)
    assert baseline.isna().all()
    assert RHR_BASELINE_LONG_MIN > 10


def test_baseline_survives_days_without_measurement():
    """Klidový tep se neztratí jen proto, že se ho ten den nepodařilo změřit."""
    values = [45.0] * 60
    df = _biometrics(values)
    df.loc[30:40, "resting_heart_rate"] = np.nan
    idx = pd.date_range("2025-01-01", periods=60, freq="D")

    baseline = rhr_baseline_series(df, idx)
    assert baseline.iloc[45:].notna().all()


def test_window_length_matches_settings():
    assert RHR_BASELINE_LONG_DAYS == 90


# ── Vliv na TRIMP ──────────────────────────────────────────────────────────

def _records(hr: float, minutes: int) -> pd.DataFrame:
    ts = pd.date_range("2025-06-01 10:00", periods=minutes * 60, freq="1s")
    return pd.DataFrame({"timestamp": ts, "heart_rate": hr, "is_active": True})


def test_higher_rhr_lowers_trimp():
    """
    Jádro změny: stejný tep při vyšším klidovém tepu znamená NIŽŠÍ relativní
    zátěž, protože se zmenšila tepová rezerva.
    """
    rec = _records(hr=150, minutes=60)
    low = compute_trimp_from_records(rec, rhr=41)
    high = compute_trimp_from_records(rec, rhr=52)
    assert high < low


def test_trimp_scales_with_intensity():
    easy = compute_trimp_from_records(_records(hr=120, minutes=60), rhr=45)
    hard = compute_trimp_from_records(_records(hr=175, minutes=60), rhr=45)
    assert hard > easy * 2


def test_hr_below_rhr_contributes_nothing():
    assert compute_trimp_from_records(_records(hr=40, minutes=30), rhr=45) == 0.0


def test_inactive_records_are_ignored():
    """Zastávka na semaforu nesmí přičítat zátěž."""
    rec = _records(hr=150, minutes=30)
    rec["is_active"] = False
    assert compute_trimp_from_records(rec, rhr=45) == 0.0


def test_long_recording_gap_is_capped():
    """
    Díra v nahrávání se stropuje na 120 s – jinak by pauza přes noc
    přičetla stovky TRIMP.
    """
    ts = [pd.Timestamp("2025-06-01 10:00"), pd.Timestamp("2025-06-01 18:00")]
    rec = pd.DataFrame({"timestamp": ts, "heart_rate": [150.0, 150.0], "is_active": True})
    trimp = compute_trimp_from_records(rec, rhr=45)
    assert trimp is not None and trimp < 10


# ── Nad reálnými daty ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def session_data():
    from src.db.session import SessionLocal, check_connection

    if not check_connection():
        pytest.skip("Databáze neběží")
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def test_sources_are_kept_separate(session_data):
    """Obě měření musí v databázi zůstat, ať je odkud dohledatelné."""
    raw = repo.read_biometrics(session_data)
    if raw.empty or "source" not in raw.columns:
        pytest.skip("Žádná biometrie")
    assert set(raw["source"].unique()) <= {"garmin", "apple"}


def test_resolution_prefers_garmin(session_data):
    """Kde měří obojí, vyhrává Garmin – a jeden den má právě jednu řádku."""
    resolved = repo.read_biometrics_resolved(session_data)
    if resolved.empty:
        pytest.skip("Žádná biometrie")

    assert resolved["date"].is_unique

    raw = repo.read_biometrics(session_data)
    garmin_days = set(raw.loc[raw["source"] == "garmin", "date"])
    overlap = resolved[resolved["date"].isin(garmin_days)]
    assert (overlap["source"] == "garmin").all()


def test_trimp_used_plausible_rhr(session_data):
    """Použitý klidový tep musí být fyziologicky možný."""
    df = pd.read_sql(
        "SELECT rhr_used FROM activity_metrics WHERE rhr_used IS NOT NULL",
        session_data.connection(),
    )
    if df.empty:
        pytest.skip("Zatím žádný přepočtený TRIMP")
    assert df["rhr_used"].min() >= 30
    assert df["rhr_used"].max() <= 80
