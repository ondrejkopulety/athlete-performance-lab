"""Testy vzorců tréninkové zátěže – čisté funkce, bez databáze."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import ATL_DAYS, CTL_DAYS
from src.analytics.load import (
    HIKING_TRIMP_COEFFICIENT,
    build_daily_load,
    compute_acwr,
    compute_ctl_atl_tsb,
    ema_decay,
)


def _calendar(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    idx = pd.date_range(start=start, periods=n, freq="D")
    idx.name = "date"
    return idx


def test_ema_uses_true_decay_constant():
    """PMC vyžaduje alpha = 1/N, ne pandas default 2/(span+1)."""
    s = pd.Series([100.0] + [0.0] * 5)
    out = ema_decay(s, span=42)

    alpha = 1 / 42
    expected = 100.0
    for _ in range(5):
        expected = (1 - alpha) * expected
    assert out.iloc[-1] == pytest.approx(expected)

    # Kdyby se použil pandas default, hodnota by byla znatelně jiná
    assert out.iloc[-1] != pytest.approx(s.ewm(span=42, adjust=False).mean().iloc[-1])


def test_tsb_uses_previous_day():
    """TSB je ranní forma PŘED dnešním tréninkem → posun o jeden den."""
    cal = _calendar(10)
    daily = pd.DataFrame({"trimp": [50.0] * 10, "trimp_epoc": [50.0] * 10}, index=cal)
    out = compute_ctl_atl_tsb(daily)

    assert pd.isna(out["tsb"].iloc[0])  # první den nemá „včera"
    for i in range(1, len(out)):
        assert out["tsb"].iloc[i] == pytest.approx(
            out["ctl"].iloc[i - 1] - out["atl"].iloc[i - 1]
        )


def test_ctl_reacts_slower_than_atl():
    """
    Fitness (42 d) se musí budovat pomaleji než únava (7 d).

    Řada musí začínat odpočinkem: u konstantní zátěže se EMA seeduje první
    hodnotou, takže by CTL i ATL od začátku ležely na stejném čísle.
    """
    cal = _calendar(30)
    trimp = [0.0] * 5 + [100.0] * 25
    daily = pd.DataFrame({"trimp": trimp, "trimp_epoc": trimp}, index=cal)
    out = compute_ctl_atl_tsb(daily)

    assert CTL_DAYS > ATL_DAYS
    assert out["ctl"].iloc[-1] < out["atl"].iloc[-1]
    assert out["tsb"].iloc[-1] < 0  # náhlý nárůst zátěže = záporná forma


def test_ctl_recovers_slower_than_atl_after_rest():
    """Po vysazení musí únava opadat rychleji, než mizí fitness."""
    cal = _calendar(60)
    trimp = [100.0] * 40 + [0.0] * 20
    daily = pd.DataFrame({"trimp": trimp, "trimp_epoc": trimp}, index=cal)
    out = compute_ctl_atl_tsb(daily)

    assert out["atl"].iloc[-1] < out["ctl"].iloc[-1]
    assert out["tsb"].iloc[-1] > 0  # odpočinek → čerstvost


def test_hiking_trimp_is_reduced():
    """Dlouhá Z1 túra nesmí nafouknout PMC jako stejně dlouhý trénink."""
    cal = _calendar(1, "2024-05-01")
    activities = pd.DataFrame({
        "date": ["2024-05-01", "2024-05-01"],
        "sport": ["hiking", "cycling"],
        "total_trimp": [100.0, 100.0],
        "epoc_score": [0.0, 0.0],
    })
    out = build_daily_load(activities, cal)
    assert out["trimp"].iloc[0] == pytest.approx(100.0 * HIKING_TRIMP_COEFFICIENT + 100.0)


def test_rest_days_are_zero_not_missing():
    """Den bez tréninku má nulovou zátěž – to je fakt, ne chybějící údaj."""
    cal = _calendar(5, "2024-03-01")
    activities = pd.DataFrame({
        "date": ["2024-03-03"],
        "sport": ["cycling"],
        "total_trimp": [80.0],
        "epoc_score": [10.0],
    })
    out = build_daily_load(activities, cal)
    assert len(out) == 5
    assert out["trimp"].notna().all()
    assert out["trimp"].iloc[0] == 0.0
    assert out["trimp"].iloc[2] == 80.0


def test_acwr_sweet_spot_for_steady_load():
    """Konstantní zátěž musí dát ACWR = 1.0 (akutní == chronická)."""
    cal = _calendar(60)
    daily = pd.DataFrame({"trimp": [60.0] * 60, "trimp_epoc": [60.0] * 60}, index=cal)
    daily = compute_ctl_atl_tsb(daily)
    out = compute_acwr(daily)
    assert out["acwr"].iloc[-1] == pytest.approx(1.0)
    # Prvních 27 dní nemá plné 28denní okno
    assert out["acwr"].iloc[:27].isna().all()


def test_acwr_is_not_clipped():
    """Hodnoty > 1.5 jsou legitimní signál, ne chyba k oříznutí."""
    cal = _calendar(60)
    trimp = [10.0] * 50 + [200.0] * 10
    daily = pd.DataFrame({"trimp": trimp, "trimp_epoc": trimp}, index=cal)
    daily = compute_ctl_atl_tsb(daily)
    out = compute_acwr(daily)
    assert out["acwr"].iloc[-1] > 1.5
