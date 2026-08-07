"""
biometrics.py  –  Regenerace, připravenost a varovné signály
=============================================================

Pure Recovery Score, Bio-Readiness (dual-era), sleep performance, HRV CV
a multi-indikátorový illness warning.

Vzorce jsou převzaté beze změny z původního athlete_analytics.py; změnil se
jen zdroj dat – místo tří CSV (hrv/daily_health/sleep) přichází jedna
tabulka daily_biometrics.

Rolling okna (HRV baseline, RHR baseline) se počítají nad **přítomnými**
hodnotami, ne nad kalendářem. Den bez měření tak okno neposune ani
neznehodnotí – chová se stejně jako původní implementace, která si každý
zdroj načítala zvlášť a chybějící dny do něj vůbec nedostala.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    HRV_DROP_THRESHOLD,
    ILLNESS_FLAG_COUNT,
    LOW_SLEEP_SCORE,
    RHR_BASELINE_DAYS,
    RHR_ELEVATION_BPM,
    SHORT_SLEEP_MINUTES,
)

# Váhy Pure Recovery Score
W_HRV, W_RHR, W_SLEEP = 0.40, 0.30, 0.30
# Váhy moderního Bio-Readiness (HRV / spánek / forma)
W_R_HRV, W_R_SLEEP, W_R_TSB = 0.30, 0.30, 0.40

# Základní potřeba spánku (7.5 h) + 0.5 min za každý TRIMP včerejška.
# Regenerační daň (recovery_tax_hours) do vzorce záměrně nevstupuje:
# 10 h snížené kapacity neznamená 10 h spánku navíc.
SLEEP_BASE_MIN = 450
SLEEP_TRIMP_FACTOR = 0.5
SLEEP_NEED_CAP_MIN = 780


def _series_from(biometrics: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> pd.Series:
    """Sloupec z daily_biometrics zarovnaný na kalendář (chybějící dny = NaN)."""
    if biometrics is None or biometrics.empty or column not in biometrics.columns:
        return pd.Series(np.nan, index=index)
    s = biometrics[["date", column]].dropna(subset=["date"]).copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.drop_duplicates(subset=["date"], keep="last").set_index("date")[column]
    return pd.to_numeric(s, errors="coerce").reindex(index)


def _rolling_on_present(values: pd.Series, window: int, min_periods: int, stat: str) -> pd.Series:
    """
    Klouzavá statistika počítaná jen nad dny, kdy měření existuje.

    Výsledek se vrací zpět na původní (kalendářní) index.
    """
    present = values.dropna()
    if present.empty:
        return pd.Series(np.nan, index=values.index)
    roll = present.rolling(window=window, min_periods=min_periods)
    out = roll.mean() if stat == "mean" else roll.std()
    return out.reindex(values.index)


def compute_recovery(daily: pd.DataFrame, biometrics: pd.DataFrame) -> pd.DataFrame:
    """
    Pure Recovery Score (0–100) = 40 % HRV + 30 % RHR + 30 % spánek.

    Doplní i surové biometrické sloupce, sleep performance a HRV CV.
    """
    daily = daily.copy()
    idx = daily.index

    hrv_last = _series_from(biometrics, "hrv_last_night", idx)
    hrv_weekly = _series_from(biometrics, "hrv_weekly_avg", idx)
    rhr = _series_from(biometrics, "resting_heart_rate", idx)
    sleep_score = _series_from(biometrics, "sleep_score", idx)
    sleep_dur = _series_from(biometrics, "sleep_duration_min", idx)
    stress = _series_from(biometrics, "stress_average", idx)

    daily["hrv_last_night"] = hrv_last
    daily["hrv_weekly_avg"] = hrv_weekly
    daily["rhr_day"] = rhr
    daily["sleep_score_day"] = sleep_score
    daily["sleep_duration_min"] = sleep_dur
    daily["avg_stress_day"] = stress

    # ── HRV složka (40 %) – poměr k 7dennímu baseline ───────────────────────
    hrv_baseline = _rolling_on_present(hrv_last, window=7, min_periods=3, stat="mean")
    ratio = hrv_last / hrv_baseline
    hrv_component = ((ratio.clip(0.7, 1.3) - 0.7) / 0.6) * 100  # 0.7→0, 1.3→100

    # ── RHR složka (30 %) – odchylka od 14denního baseline ──────────────────
    rhr_baseline = _rolling_on_present(rhr, window=RHR_BASELINE_DAYS, min_periods=5, stat="mean")
    diff = rhr - rhr_baseline  # záporné = nižší tep = lepší
    rhr_component = (1 - (diff.clip(-5, 10) + 5) / 15) * 100
    daily["rhr_baseline_14d"] = rhr_baseline.round(1)

    # ── Spánková složka (30 %) – skóre je už 0–100 ──────────────────────────
    sleep_component = sleep_score.clip(0, 100)

    combined = (
        W_HRV * hrv_component.fillna(50)
        + W_RHR * rhr_component.fillna(50)
        + W_SLEEP * sleep_component.fillna(50)
    )
    daily["pure_recovery_score"] = combined.clip(0, 100).round(1)

    # ── Sleep performance (Whoop-style) ─────────────────────────────────────
    trimp_prev = daily["trimp"].shift(1).fillna(0)  # první den neextrapolujeme
    sleep_need = (SLEEP_BASE_MIN + trimp_prev * SLEEP_TRIMP_FACTOR).clip(upper=SLEEP_NEED_CAP_MIN)
    daily["sleep_need_min"] = sleep_need.round(0)
    daily["sleep_performance_pct"] = (
        (sleep_dur / sleep_need.replace(0, np.nan)) * 100
    ).round(1)

    # ── HRV koeficient variace (7 dní, z hrubého RMSSD) ─────────────────────
    hrv_std = hrv_last.rolling(7, min_periods=3).std()
    hrv_mean = hrv_last.rolling(7, min_periods=3).mean()
    daily["hrv_cv_pct"] = ((hrv_std / hrv_mean.replace(0, np.nan)) * 100).round(1)

    return daily


def compute_readiness(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Bio-Readiness Score (0–100) ve dvou režimech podle dostupnosti dat.

    Legacy (období před Garminem – chybí HRV i spánek):
        readiness = f(TSB); mapování −30 → 0, +10 → 100.

    Modern (existuje sleep_score i hrv_last_night pro daný den):
        0.30 × HRV z-score + 0.30 × sleep_score + 0.40 × normalizované TSB

    Přepínač je per-řádek, ne globální – přechod mezi érami tak nevytvoří
    v grafu skok způsobený změnou vzorce.
    """
    daily = daily.copy()
    if "tsb" not in daily.columns:
        daily["readiness_score"] = np.nan
        return daily

    tsb = pd.to_numeric(daily["tsb"], errors="coerce")
    tsb_normed = ((tsb.clip(-30, 10) + 30) / 40) * 100

    hrv = pd.to_numeric(daily.get("hrv_last_night", pd.Series(np.nan, index=daily.index)), errors="coerce")
    hrv_30d_mean = _rolling_on_present(hrv, window=30, min_periods=7, stat="mean")
    hrv_30d_std = _rolling_on_present(hrv, window=30, min_periods=7, stat="std").replace(0, np.nan)
    hrv_z = ((hrv - hrv_30d_mean) / hrv_30d_std).clip(-3, 3)
    normed_hrv = ((hrv_z + 3) / 6) * 100  # −3 → 0, +3 → 100

    sleep_score = pd.to_numeric(
        daily.get("sleep_score_day", pd.Series(np.nan, index=daily.index)), errors="coerce"
    ).clip(0, 100)

    is_modern = sleep_score.notna() & hrv.notna()
    readiness_modern = (
        W_R_HRV * normed_hrv.fillna(50) + W_R_SLEEP * sleep_score + W_R_TSB * tsb_normed
    )

    readiness = tsb_normed.copy()
    readiness[is_modern] = readiness_modern[is_modern]
    readiness = readiness.clip(0, 100).round(1)
    readiness[tsb.isna()] = np.nan  # bez tréninkových dat nemá smysl

    daily["readiness_score"] = readiness
    return daily


def compute_illness_warning(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-indikátorový varovný systém.

    Čtyři nezávislé vlajky:
      1. HRV pod týdenním průměrem o víc než HRV_DROP_THRESHOLD
      2. RHR o RHR_ELEVATION_BPM nad vlastním 14denním baseline
      3. bad_sleep = nízké skóre NEBO krátký spánek (sloučeno záměrně –
         jedna špatná noc nesmí zvednout dvě vlajky a předčasně spustit alarm)
      4. Strain v horním kvartilu při současně nízké regeneraci

    illness_warning se zapne při ILLNESS_FLAG_COUNT a více vlajkách současně.

    Všechny vlajky jsou relativní k vlastnímu baseline, ne k pevným číslům.
    U RHR to platí od chvíle, kdy se ukázalo, že absolutní práh 46 bpm ležel
    přesně na mediánu a vlajka proto hořela 44 % dní – tedy nenesla žádnou
    informaci a jen ředila celý alarm.
    """
    daily = daily.copy()
    flags = pd.DataFrame(index=daily.index)

    ln = pd.to_numeric(daily.get("hrv_last_night"), errors="coerce")
    wa = pd.to_numeric(daily.get("hrv_weekly_avg"), errors="coerce")
    if ln is not None and wa is not None:
        flags["hrv_drop"] = (
            ln.notna() & wa.notna() & (wa > 0) & (ln < wa * (1 - HRV_DROP_THRESHOLD))
        )
    else:
        flags["hrv_drop"] = False

    # Baseline se počítá z PŘEDCHOZÍCH dní (shift(1)) – kdyby zahrnoval
    # dnešek, elevovaná hodnota by si zvedla vlastní referenci a tlumila
    # tak signál, který má právě detekovat.
    rhr = pd.to_numeric(daily.get("rhr_day"), errors="coerce")
    rhr_baseline = _rolling_on_present(
        rhr.shift(1), window=RHR_BASELINE_DAYS, min_periods=5, stat="mean"
    )
    # Zaokrouhlit dřív, než se porovnává. Jinak by se rozhodovalo nad jiným
    # číslem, než jaké uvidí uživatel: elevace 4,96 se zobrazí jako +5,0,
    # ale varování by nepřišlo.
    elevation = (rhr - rhr_baseline).round(1)
    flags["high_rhr"] = (elevation >= RHR_ELEVATION_BPM).fillna(False)
    daily["rhr_elevation_bpm"] = elevation

    low_score = (
        pd.to_numeric(daily.get("sleep_score_day"), errors="coerce") < LOW_SLEEP_SCORE
    ).fillna(False)
    short_dur = (
        pd.to_numeric(daily.get("sleep_duration_min"), errors="coerce") < SHORT_SLEEP_MINUTES
    ).fillna(False)
    flags["bad_sleep"] = low_score | short_dur

    if "strain" in daily.columns and "pure_recovery_score" in daily.columns:
        strain = pd.to_numeric(daily["strain"], errors="coerce")
        # Kvartil na klouzavém 30denním okně – prahem je aktuální tréninková
        # kondice, ne historické maximum z jiné části sezóny.
        strain_q75 = strain.rolling(30, min_periods=10).quantile(0.75)
        flags["high_strain"] = (
            (strain > strain_q75) & (daily["pure_recovery_score"] < 50)
        ).fillna(False)
    else:
        flags["high_strain"] = False

    flag_count = flags.sum(axis=1).fillna(0).astype(int)
    daily["stress_flag_count"] = flag_count
    daily["illness_warning"] = flag_count >= ILLNESS_FLAG_COUNT

    daily["stress_flags"] = (
        pd.concat([flags[c].map({True: c, False: ""}) for c in flags.columns], axis=1)
        .apply(lambda r: " | ".join([x for x in r if x]), axis=1)
    )
    return daily
