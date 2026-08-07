"""
advice.py  –  textové doporučení trenéra
=========================================

Deterministické shrnutí dne do jedné věty. Vzniklo před AI trenérem a
zůstává i vedle něj: je levné, předvídatelné a funguje i bez sítě.
Chatbot ho dostává v kontextu jako signál „na co se dívat".

Vzorec i formulace jsou převzaté beze změny z athlete_analytics.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    CTL_RAMP_WARN,
    HRV_DROP_THRESHOLD,
    LOW_SLEEP_SCORE,
    RHR_ELEVATION_BPM,
    SHORT_SLEEP_MINUTES,
)

STRAIN_HIGH = 800
STRAIN_MODERATE = 400
ACWR_DANGER = 1.5
ACWR_ELEVATED = 1.3


def _row_advice(row: pd.Series) -> str:
    parts: list[str] = []

    strain_val = row.get("strain", np.nan)
    if pd.notna(strain_val):
        if strain_val > STRAIN_HIGH:
            parts.append("VYSOKÝ STRAIN – priorita je regenerace")
        elif strain_val > STRAIN_MODERATE:
            parts.append("Střední zátěž – sleduj únavu")

    sleep_score = row.get("sleep_score_day", np.nan)
    if pd.notna(sleep_score) and sleep_score < LOW_SLEEP_SCORE:
        parts.append(f"Špatný spánek ({sleep_score:.0f}/100)")

    sleep_dur = row.get("sleep_duration_min", np.nan)
    if pd.notna(sleep_dur) and sleep_dur < SHORT_SLEEP_MINUTES:
        parts.append(f"Krátký spánek ({sleep_dur:.0f} min)")

    hrv_ln = row.get("hrv_last_night", np.nan)
    hrv_wa = row.get("hrv_weekly_avg", np.nan)
    if pd.notna(hrv_ln) and pd.notna(hrv_wa) and hrv_wa > 0:
        drop_pct = 1 - hrv_ln / hrv_wa
        if drop_pct > HRV_DROP_THRESHOLD:
            parts.append(
                f"HRV pokles {drop_pct * 100:.0f}% pod průměr "
                f"({hrv_ln:.0f} vs {hrv_wa:.0f} ms)"
            )

    # Zvýšený tep se hlásí vůči vlastnímu baseline, ne proti pevnému číslu –
    # „48 bpm" samo o sobě nic neříká, „48 při běžných 43" ano.
    rhr = row.get("rhr_day", np.nan)
    rhr_elev = row.get("rhr_elevation_bpm", np.nan)
    if pd.notna(rhr) and pd.notna(rhr_elev) and rhr_elev >= RHR_ELEVATION_BPM:
        parts.append(
            f"Zvýšený RHR ({rhr:.0f} bpm, +{rhr_elev:.0f} nad 14denním průměrem)"
        )

    if row.get("illness_warning", False):
        parts.insert(0, "⚠️ ILLNESS WARNING – zvažuj odpočinek")

    recovery = row.get("pure_recovery_score", np.nan)
    if pd.notna(recovery):
        if recovery >= 80:
            if not parts:
                parts.append("Výborná regenerace – připraven na trénink")
        elif recovery >= 60:
            if not parts:
                parts.append("Dobrá regenerace – standardní trénink")
        elif recovery < 40:
            parts.append(f"Nízká regenerace ({recovery:.0f}/100) → sniž intenzitu")

    tsb_val = row.get("tsb", np.nan)
    if pd.notna(tsb_val):
        if tsb_val < -30:
            parts.append("TSB kriticky nízké – odpočinek!")
        elif tsb_val > 10:
            parts.append("Čerstvost – můžeš zvýšit zátěž")

    acwr_val = row.get("acwr", np.nan)
    if pd.notna(acwr_val):
        if acwr_val > ACWR_DANGER:
            parts.insert(
                0,
                f"⚠ ACWR v červené zóně ({acwr_val:.1f}) – sniž objem, nebo hrozí zranění!",
            )
        elif acwr_val > ACWR_ELEVATED:
            parts.append(f"ACWR zvýšené ({acwr_val:.1f}) – opatrně s nárůstem")

    ramp_val = row.get("ctl_ramp_rate", np.nan)
    if pd.notna(ramp_val) and ramp_val > CTL_RAMP_WARN:
        parts.append(f"CTL ramp příliš strmý ({ramp_val:+.1f}/týden) – hrozí burn-out!")

    return " | ".join(parts)


def compute_coach_advice(daily: pd.DataFrame) -> pd.DataFrame:
    """Přidá textový sloupec coach_advice pro každý den."""
    daily = daily.copy()
    daily["coach_advice"] = [_row_advice(row) for _, row in daily.iterrows()]
    return daily
