#!/usr/bin/env python3
"""
athlete_analytics.py – Virtual Lab & Load Management Suite v7.0  (PRODUCTION)
==============================================================================
Hloubková analýza tréninkové zátěže a biometrických reakcí.

Toto je **analytická vrstva** – jediné místo, kde se kombinují
tréninková data (z FIT parseru) s biometrií (HRV, spánek, RHR).

Moduly
------
A. Cardiac Drift (Aerobní decoupling Pa:HR)
B. Maximal 60 s HR Drop (max_hrr_60s) – resample→interpolate→max drop
C. Training Monotony, Strain & Whoop Logarithmic Strain (7denní okno)
D. Efficiency Index (TRIMP / km)
E. Bio-Readiness Score (0–100 %)
F. Pure Recovery Score (HRV + RHR + Spánek) + Sleep Performance % + HRV CV
G. Illness Warning (více-indikátorový varovný systém)
H. Coach Advice (textové doporučení na základě Strain, Spánku, HRV, ACWR)
I. Polarization Score (80/20 Tactic – 14denní klouzavý rozbor zón)
J. VAM (Velocità Ascensionale Media – rychlost stoupání)
K. ACWR (Acute:Chronic Workload Ratio) + CTL Ramp Rate  ← EPOC-weighted TRIMP
L. External vs Internal Load – Fatigue Index
M. Fueling Model (Fat/Carb kcal by HR zone)
N. Fluid Loss Estimate
O. Temp-Effect Model (heat vs cardiac drift)
P. Durability (EF decay in long activities)
Q. TTE (Time to Exhaustion estimate)
R. EPOC & Oxygen Debt
S. DFA-alpha1 Proxy (AeT breakpoint) – fallback for Module U
T. Climb Score (avg gradient + category)
U. DFA-alpha1 REAL (neurokit2)  – AeT (α1=0.75) & AnT (α1=0.50) from R-R
V. Respiration Rate from HRV (RSA via Welch PSD on R-R intervals)
W. TATI (Time Above Threshold Impulse) & Critical Heart Rate (Monod-Scherrer HR model)

Vstupní soubory
---------------
* data/summaries/master_high_res_summary.csv
* data/summaries/master_high_res_training_data.csv
* data/summaries/hrv.csv
* data/summaries/daily_health.csv
* data/summaries/sleep.csv
* data/summaries/activities.csv  (distance_meters)

Výstup
------
* data/summaries/athlete_readiness.csv  (denní metriky vč. recovery, coach_advice)
* Zpětný zápis cardiac_drift, max_hrr_60s do master_high_res_summary.csv

PRODUKČNÍ REŽIM – zpracovává kompletní databázi bez omezení.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Optional

import glob

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

from config.settings import (
    MAX_HR as ATHLETE_MAX_HR, RESTING_HR as ATHLETE_RHR, ZONE_PCTS as ATHLETE_ZONE_PCTS,
    CTL_DAYS, ATL_DAYS, EF_WINDOW, MONOTONY_WINDOW,
    HRV_DROP_THRESHOLD, HIGH_RHR_THRESHOLD, LOW_SLEEP_SCORE,
    SHORT_SLEEP_MINUTES, ILLNESS_FLAG_COUNT,
    WARMUP_SECONDS, ACWR_ACUTE_DAYS, ACWR_CHRONIC_DAYS, CTL_RAMP_WARN,
    FAT_PCT_BY_ZONE, KCAL_PER_MIN_BY_ZONE,
    DURABILITY_MIN_DURATION_MIN as DURABILITY_MIN_DURATION,
    DFA_WINDOW_BEATS, DFA_AET_THRESHOLD, DFA_ANT_THRESHOLD,
    MAX_REALISTIC_HRR, MIN_STARTING_HR, MAX_3S_JUMP,
    SUMMARIES_DIR, FIT_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional advanced-physiology dependencies ─────────────────────────────────
try:
    import neurokit2 as nk
    from scipy.signal import welch
    from scipy.ndimage import uniform_filter1d
    from scipy.interpolate import interp1d
    from fitparse import FitFile as FitFileParser
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS (from config.settings + local)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = _PROJECT_ROOT
_SUMMARIES = str(SUMMARIES_DIR)

MASTER_CSV   = os.path.join(_SUMMARIES, "master_high_res_summary.csv")
TRAINING_CSV = os.path.join(_SUMMARIES, "master_high_res_training_data.csv")
ACTIVITIES_CSV = os.path.join(_SUMMARIES, "activities.csv")
HRV_CSV      = os.path.join(_SUMMARIES, "hrv.csv")
HEALTH_CSV   = os.path.join(_SUMMARIES, "daily_health.csv")
SLEEP_CSV    = os.path.join(_SUMMARIES, "sleep.csv")
OUTPUT_CSV   = os.path.join(_SUMMARIES, "athlete_readiness.csv")

EFFICIENCY_TREND_WINDOW = 14

# ── Stress / Recovery thresholds (imported from config.settings) ────────────

# Sports eligible for cardiac-drift / EF / max_hrr_60s analysis
CARDIO_SPORTS = ("running", "cycling")

MIN_DURATION_DRIFT = 20  # minutes – minimum duration for cardiac drift

# ── Max HRR 60 s (resample + interpolate) ─────────────────────────────────────
HRR_RESAMPLE_FREQ = "1s"     # převzorkování na 1 vteřinu
HRR_INTERPOLATE_LIMIT = 30   # max 30 s interpolace (nezaplácne velké díry)
HRR_DROP_WINDOW = 60         # hledáme max pokles HR za 60 s

# ── Fueling model (local-only constants) ──────────────────────────────────────
KCAL_PER_G_FAT = 9.0
KCAL_PER_G_CARB = 4.0

# ── EPOC / Oxygen debt ───────────────────────────────────────────────────────
EPOC_SLEEP_MIN_PER_10_Z5 = 15   # každých 10 min v Z5 = +15 min potřeby spánku

# ── DFA-alpha1 (local-only constants) ─────────────────────────────────────────
DFA_SLIDE_BEATS = 30         # sliding step (beats)
DFA_BOX_SIZES = list(range(4, 17))  # box sizes n for short-range DFA-alpha1

# ── TATI / Critical Heart Rate ────────────────────────────────────────────────
WPRIME_MIN_ACTIVITIES = 5    # minimum cardio activities for CHR estimation
WPRIME_LONG_DURATION = 60    # minutes – activities used for CHR baseline

# ── FIT data folder ──────────────────────────────────────────────────────────
FIT_FOLDER = str(FIT_DIR)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def ema_decay(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with true decay constant alpha = 1/N.

    Using alpha=1/span (instead of the pandas default 2/(span+1)) gives the
    canonical 42-day/7-day time constants required by the Bannister PMC model.
    """
    return series.ewm(alpha=1 / span, adjust=False).mean()


def safe_div(a, b, default=np.nan):
    """Division safe against zero / NaN."""
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return default
    result = a / b
    if isinstance(result, float) and np.isnan(result):
        return default
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FIT FILE ACCESS & R-R INTERVAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def find_fit_file(activity_id: str) -> Optional[str]:
    """Locate FIT file on disk for a given activity ID."""
    strava_folder = os.path.join(FIT_FOLDER, "strava_originals")
    patterns = [
        os.path.join(FIT_FOLDER, f"activity_{activity_id}.fit"),
        os.path.join(FIT_FOLDER, f"{activity_id}_ACTIVITY.fit"),
        os.path.join(FIT_FOLDER, f"{activity_id}.fit"),
        os.path.join(strava_folder, f"activity_{activity_id}.fit"),
        os.path.join(strava_folder, f"{activity_id}_ACTIVITY.fit"),
        os.path.join(strava_folder, f"{activity_id}.fit"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    # glob fallback – search both folders
    for folder in (FIT_FOLDER, strava_folder):
        for p in glob.glob(os.path.join(folder, f"*{activity_id}*.fit")):
            return p
    return None


def extract_rr_for_activity(activity_id: str) -> list[float]:
    """Extract R-R intervals (seconds) from FIT file for given activity."""
    if not HAS_NEUROKIT:
        return []
    fit_path = find_fit_file(activity_id)
    if fit_path is None:
        return []
    try:
        fitfile = FitFileParser(fit_path)
        rr: list[float] = []
        for msg in fitfile.get_messages("hrv"):
            intervals = msg.get_values().get("time")
            if intervals is None:
                continue
            if isinstance(intervals, (list, tuple)):
                rr.extend(v for v in intervals if v is not None and v > 0)
            elif isinstance(intervals, (int, float)) and intervals > 0:
                rr.append(float(intervals))
        return rr
    except Exception as exc:
        log.debug("Cannot extract RR from %s: %s", activity_id, exc)
        return []


def clean_rr_intervals(rr_s: list[float], max_pct_change: float = 0.20) -> np.ndarray:
    """
    Clean R-R interval array by removing physiologically implausible values.

    Parameters
    ----------
    rr_s : R-R intervals in seconds
    max_pct_change : max allowed % change between consecutive beats (ectopic filter)

    Returns
    -------
    Cleaned R-R intervals in milliseconds.
    """
    rr_ms = np.array(rr_s) * 1000.0  # convert to ms

    # 1. Remove implausible values (HR 30–220 bpm → RR 273–2000 ms)
    mask = (rr_ms >= 273) & (rr_ms <= 2000)
    rr_ms = rr_ms[mask]

    if len(rr_ms) < 10:
        return rr_ms

    # 2. Remove ectopic beats (>20 % change from previous beat)
    clean_indices = [0]
    for i in range(1, len(rr_ms)):
        pct_change = abs(rr_ms[i] - rr_ms[i - 1]) / rr_ms[i - 1]
        if pct_change <= max_pct_change:
            clean_indices.append(i)

    return rr_ms[clean_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_csv(path: str, required: bool = True) -> pd.DataFrame:
    """Generic CSV loader with existence check."""
    if not os.path.exists(path):
        if required:
            sys.exit(f"[ERROR] Nenalezen soubor: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_master() -> pd.DataFrame:
    df = load_csv(MASTER_CSV)
    df = df.assign(
        date=pd.to_datetime(df["date"]).dt.date,
        activity_id=df["activity_id"].astype(str),
    )
    return df


def load_activities() -> pd.DataFrame:
    df = load_csv(ACTIVITIES_CSV, required=False)
    if df.empty:
        return df
    df = df.assign(activity_id=df["activity_id"].astype(str))
    return df


def load_hrv() -> pd.DataFrame:
    df = load_csv(HRV_CSV, required=False)
    if df.empty:
        return df
    df = df.assign(date=pd.to_datetime(df["date"]).dt.date)
    return df


def load_health() -> pd.DataFrame:
    df = load_csv(HEALTH_CSV, required=False)
    if df.empty:
        return df
    df = df.assign(date=pd.to_datetime(df["date"]).dt.date)
    return df


def load_sleep() -> pd.DataFrame:
    """Načte sleep.csv – denní skóre spánku, délku, fáze."""
    df = load_csv(SLEEP_CSV, required=False)
    if df.empty:
        return df
    df = df.assign(date=pd.to_datetime(df["date"]).dt.date)
    return df


def load_training_data_bulk(activity_ids: set) -> dict:
    """
    Single-pass read of TRAINING_CSV for a set of activity IDs.
    Returns dict {activity_id: DataFrame} – much faster than one-by-one reads.
    """
    frames_by_id: dict = {aid: [] for aid in activity_ids}
    for chunk in pd.read_csv(TRAINING_CSV, chunksize=500_000, low_memory=False):
        chunk = chunk.assign(activity_id=chunk["activity_id"].astype(str))
        filtered = chunk.loc[chunk["activity_id"].isin(activity_ids)]
        if filtered.empty:
            continue
        for aid, grp in filtered.groupby("activity_id"):
            frames_by_id[str(aid)].append(grp)

    result = {}
    for aid, frames in frames_by_id.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df.assign(timestamp=pd.to_datetime(df["timestamp"]))
        df = df.sort_values("timestamp").reset_index(drop=True)
        result[aid] = df
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE A – CARDIAC DRIFT (Aerobní decoupling)
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum allowed altitude variation (m) over the active segment before the
# EF = speed/HR metric becomes unreliable (terrain-induced HR changes).
CARDIAC_DRIFT_MAX_ALT_RANGE_M: float = 30.0

def cardiac_drift_for_activity(tdata: pd.DataFrame, sport: str) -> Optional[float]:
    """
    Přeskoč 10 min warm-up, zbytek rozděl na dvě poloviny.
    Pro cyklistiku: EF = power / HR (speed nedává smysl kvůli terénu).
    Pro běh: EF = speed / HR.
    Drift = (EF_1 - EF_2) / EF_1 * 100   [%]

    Terrain & steady-state filters (running only):
      • Flat-terrain guard: if altitude range across the active segment exceeds
        ``CARDIAC_DRIFT_MAX_ALT_RANGE_M`` metres the EF = speed/HR relationship
        is confounded by gradient-induced HR changes → return None.
      • Z2 steady-state filter: only Z1/Z2 records are used for EF calculation
        so that Z4/Z5 intervals (tempo, threshold efforts, sprints) do not
        artificially inflate the second-half HR average.

    For cycling the power meter already controls for terrain, so neither filter
    is applied (power/HR remains meaningful on any gradient).

    Returns None if insufficient data or filters are not met.
    """
    if tdata.empty:
        return None

    is_cycling = "cycling" in sport or "bike" in sport
    effort_col = "power" if is_cycling else "speed"

    # check effort column exists
    if effort_col not in tdata.columns:
        return None

    # only active data with valid HR & effort
    active = tdata.loc[
        (tdata["is_active"] == True)
        & tdata["heart_rate"].notna()
        & (tdata["heart_rate"] > 0)
        & tdata[effort_col].notna()
        & (tdata[effort_col] > 0)
    ].copy()

    if len(active) < 20:
        return None

    # elapsed seconds from first row
    t0 = active["timestamp"].iloc[0]
    elapsed = (active["timestamp"] - t0).dt.total_seconds().values

    # skip warm-up (first 10 min)
    mask_post_warmup = elapsed >= WARMUP_SECONDS
    post = active.loc[mask_post_warmup]

    if len(post) < 20:
        return None

    # ── Terrain flatness guard (running only) ────────────────────────────
    # On hilly terrain speed/HR is confounded: gradient raises HR independently
    # of aerobic drift. Skip if total altitude range exceeds threshold.
    if not is_cycling and "altitude" in post.columns:
        alt_vals = pd.to_numeric(post["altitude"], errors="coerce").dropna()
        if len(alt_vals) >= 10:
            alt_range = float(alt_vals.max() - alt_vals.min())
            if alt_range > CARDIAC_DRIFT_MAX_ALT_RANGE_M:
                return None

    # ── Z2 steady-state filter (running only) ────────────────────────────
    # Drop Z4/Z5 (threshold/VO2max) intervals that would skew the second-half
    # HR average upward and falsely suggest cardiac drift.
    if not is_cycling and "hr_zone" in post.columns:
        post = post.loc[post["hr_zone"].isin(["Z1", "Z2", ""])]
        if len(post) < 20:
            return None

    # Time-based midpoint: handles non-uniform Smart Recording intervals
    # correctly (row-count split would skew halves when recording rate varies).
    t_start = post["timestamp"].iloc[0]
    t_end   = post["timestamp"].iloc[-1]
    t_mid   = t_start + (t_end - t_start) / 2
    first_half  = post[post["timestamp"] <= t_mid]
    second_half = post[post["timestamp"] >  t_mid]

    if first_half.empty or second_half.empty:
        return None

    hr1, hr2 = first_half["heart_rate"].mean(), second_half["heart_rate"].mean()
    eff1, eff2 = first_half[effort_col].mean(), second_half[effort_col].mean()

    if hr1 == 0 or hr2 == 0:
        return None

    # Guard: reject near-zero effort segments
    if is_cycling:
        # cycling: reject if average power < 30 W in either half
        if eff1 < 30 or eff2 < 30:
            return None
    else:
        # running: reject near-stationary segments (< 2.0 km/h = 0.5556 m/s)
        if eff1 < 2.0 / 3.6 or eff2 < 2.0 / 3.6:
            return None

    ef1 = safe_div(eff1, hr1)
    ef2 = safe_div(eff2, hr2)

    if ef1 is None or ef1 == 0 or np.isnan(ef1):
        return None

    drift = (ef1 - ef2) / ef1 * 100.0
    return round(drift, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE B – MAXIMAL 60 s HR DROP (max_hrr_60s)
# ═══════════════════════════════════════════════════════════════════════════════
def _window_is_smooth(hr_window: np.ndarray, max_3s_jump: int = MAX_3S_JUMP) -> bool:
    """Return True if no consecutive 3-second span has a drop > *max_3s_jump*."""
    if len(hr_window) < 4:
        return True
    # rolling 3-second differences: hr[t] - hr[t+3]
    diffs = hr_window[:-3] - hr_window[3:]
    return bool(np.all(diffs <= max_3s_jump))


def max_hrr_60s_for_activity(tdata: pd.DataFrame) -> Optional[float]:
    """
    Robustní výpočet maximálního poklesu tepové frekvence za 60 s
    s ochranou proti sensor-glitch artefaktům.

    Algoritmus:
      1. Převzorkuj na 1 s, interpoluj (max 30 s gap).
      2. Klouzavé okno (60 s): pro každý start *t* spočti drop = HR[t] − HR[t+60].
      3. Sanity checks pro každé okno:
         a) HR[t] >= MIN_STARTING_HR  (recovery jen z dostatečné zátěže)
         b) drop   <= MAX_REALISTIC_HRR  (> 90 bpm/min = glitch)
         c) žádný 3s skok > MAX_3S_JUMP  (plynulý průběh)
      4. Ze všech validních oken vrátí nejvyšší pokles.

    Returns max drop (bpm) nebo None.
    """
    if tdata.empty or "heart_rate" not in tdata.columns:
        return None

    df = tdata[["timestamp", "heart_rate"]].dropna(subset=["heart_rate"]).copy()
    if len(df) < 60:
        return None

    df = df.set_index("timestamp").sort_index()

    # Převzorkování na 1 vteřinu
    df_1s = df.resample(HRR_RESAMPLE_FREQ).mean()

    # Interpolace chybějících dat (max 30 s gap)
    df_1s = df_1s.interpolate(method="time", limit=HRR_INTERPOLATE_LIMIT)

    hr = df_1s["heart_rate"].dropna()
    n = len(hr)
    if n < HRR_DROP_WINDOW + 1:
        return None

    hr_vals = hr.values  # numpy array for speed

    # ── Naivní absolutní maximum (pro logging) ───────────────────────────
    naive_drops = hr_vals[:-HRR_DROP_WINDOW] - hr_vals[HRR_DROP_WINDOW:]
    naive_max = float(np.nanmax(naive_drops)) if len(naive_drops) > 0 else 0.0

    # ── Sliding-window search with sanity checks ─────────────────────────
    best_drop: float = 0.0
    glitch_logged = False

    for t in range(n - HRR_DROP_WINDOW):
        start_hr = hr_vals[t]
        end_hr = hr_vals[t + HRR_DROP_WINDOW]
        drop = start_hr - end_hr

        if drop <= 0:
            continue

        # (a) Recovery jen z dostatečné zátěže
        if start_hr < MIN_STARTING_HR:
            continue

        # (b) Pokles nesmí překročit realistickou hranici
        if drop > MAX_REALISTIC_HRR:
            if not glitch_logged:
                log.debug(
                    "Ignorován glitch (%.0f bpm), hledám nejlepší "
                    "reálnou alternativu...",
                    drop,
                )
                glitch_logged = True
            continue

        # (c) Plynulý průběh – žádný 3s skok
        window = hr_vals[t : t + HRR_DROP_WINDOW + 1]
        if not _window_is_smooth(window):
            if not glitch_logged and drop > best_drop:
                log.debug(
                    "Ignorován glitch (%.0f bpm), hledám nejlepší "
                    "reálnou alternativu...",
                    drop,
                )
                glitch_logged = True
            continue

        if drop > best_drop:
            best_drop = drop

    # ── Log pokud naivní max byl nesmyslný a musel být přeskočen ─────────
    if naive_max > MAX_REALISTIC_HRR and not glitch_logged:
        log.debug(
            "Ignorován glitch (%.0f bpm), hledám nejlepší "
            "reálnou alternativu...",
            naive_max,
        )

    if best_drop <= 0:
        return None

    return round(best_drop, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE CARDIAC DRIFT & MAX HRR 60 s  (FULL DATABASE)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_per_activity_metrics(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produkční režim – kompletní databáze.
    Načte vteřinová data v jednom průchodu pro všechny eligible aktivity
    a vypočte cardiac_drift + max_hrr_60s.

    Returns updated master DataFrame.
    """
    master = master.copy()

    # initialise columns
    if "cardiac_drift" not in master.columns:
        master = master.assign(cardiac_drift=np.nan)
    if "max_hrr_60s" not in master.columns:
        master = master.assign(max_hrr_60s=np.nan)

    if not os.path.exists(TRAINING_CSV):
        print("  [WARN] High-res training data not found – skipping Drift & Max HRR.")
        return master

    # ── eligible activities (cardio, ≥ 20 min) ───────────────────────────
    is_cardio = master["sport"].str.contains(
        "|".join(CARDIO_SPORTS), case=False, na=False
    )
    long_enough = master["duration_minutes"] >= MIN_DURATION_DRIFT
    eligible = master.loc[is_cardio & long_enough].copy()

    if eligible.empty:
        print("  [WARN] Žádné eligible aktivity (cardio ≥ 20 min).")
        return master

    aid_set = set(eligible["activity_id"].astype(str).tolist())
    n = len(aid_set)
    print(f"  PRODUKČNÍ REŽIM: {n} aktivit ke skenování (celá databáze)...")
    print("  Načítám vteřinová data (1 průchod)...")

    # ── single-pass bulk load ────────────────────────────────────────────
    tdata_map = load_training_data_bulk(aid_set)
    print(f"  Načteno {len(tdata_map)} aktivit ze souboru.")

    # Collect results into dicts first, then assign as whole columns at the end.
    # This avoids per-row .loc[] writes which cause memory fragmentation on large
    # DataFrames (each write may trigger a copy of the underlying block store).
    drift_results: dict = {}
    hrr_results: dict = {}

    for i, (idx, row) in enumerate(eligible.iterrows(), 1):
        aid = str(row["activity_id"])
        tdata = tdata_map.get(aid, pd.DataFrame())

        sport = str(row["sport"]).lower()
        drift = cardiac_drift_for_activity(tdata, sport)
        hrr_60 = max_hrr_60s_for_activity(tdata)

        drift_results[idx] = drift
        hrr_results[idx] = hrr_60

        if i % 20 == 0 or i == n:
            print(
                f"    [{i}/{n}]  {row['date']}  "
                f"drift={drift}  max_hrr_60s={hrr_60}"
            )

    # Bulk column update – single write to the backing store
    if drift_results:
        master.loc[list(drift_results.keys()), "cardiac_drift"] = list(drift_results.values())
        master.loc[list(hrr_results.keys()), "max_hrr_60s"] = list(hrr_results.values())

    computed_drift = master["cardiac_drift"].notna().sum()
    computed_hrr = master["max_hrr_60s"].notna().sum()
    print(
        f"  ✓ Cardiac Drift: {computed_drift} aktivit  "
        f"|  Max HRR 60s: {computed_hrr} aktivit"
    )
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE C – TRAINING MONOTONY, STRAIN & WHOOP LOGARITHMIC STRAIN
# ═══════════════════════════════════════════════════════════════════════════════
def compute_monotony_strain(daily: pd.DataFrame) -> pd.DataFrame:
    """
    7-day rolling window on daily TRIMP.
    Monotony = mean(TRIMP) / std(TRIMP)
    Strain   = Monotony * sum(TRIMP)
    Whoop Logarithmic Strain = 21 * (1 − exp(−0.005 * TRIMP))
    """
    daily = daily.copy()
    roll = daily["trimp"].rolling(window=MONOTONY_WINDOW, min_periods=MONOTONY_WINDOW)
    roll_mean = roll.mean()
    roll_std = roll.std()
    roll_sum = roll.sum()

    # Bezpečné dělení (epsilon 1e-5) a fyziologický strop pro monotonii (max 4.0).
    # Epsilon zabraňuje dělení nulou; strop 4.0 filtruje artefakty dat
    # (při nízké odchylce a vysoké průměrné zátěži by monotony nereálně explodovala).
    monotony = (roll_mean / (roll_std + 1e-5)).clip(upper=4.0)
    strain = monotony * roll_sum

    daily.loc[:, "monotony"] = monotony.round(2)
    daily.loc[:, "strain"] = strain.round(1)

    # Whoop Logarithmic Strain (0–21 scale, based on daily TRIMP)
    daily.loc[:, "whoop_strain"] = (21 * (1 - np.exp(-0.005 * daily["trimp"]))).round(2)

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE D – EFFICIENCY INDEX (TRIMP / km)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_efficiency_index(
    master: pd.DataFrame, activities: pd.DataFrame
) -> pd.DataFrame:
    """
    TRIMP / distance_km per day → 14-day trend.
    Restricted to **running only** to avoid sport-mixing bias: cycling produces
    roughly half the TRIMP/km of running, so merging the two makes fatigue_index
    meaningless when sport changes across the training block.
    Aggregation: sum(TRIMP) / sum(km) per day — avoids bias from short high-TRIMP efforts.
    Returns DataFrame indexed by date with columns: daily_efficiency, ef_trend.
    """
    cardio = master.loc[
        master["sport"].str.contains("running", case=False, na=False)
    ].copy()

    if cardio.empty:
        return pd.DataFrame(columns=["daily_efficiency", "ef_trend"])

    if "distance_km" not in cardio.columns:
        return pd.DataFrame(columns=["daily_efficiency", "ef_trend"])

    cardio = cardio.assign(
        date=pd.to_datetime(cardio["date"]),
        distance_km=pd.to_numeric(cardio["distance_km"], errors="coerce"),
        total_trimp=pd.to_numeric(cardio["total_trimp"], errors="coerce"),
    )
    # Drop rows missing either metric
    cardio = cardio.dropna(subset=["distance_km", "total_trimp"])
    cardio = cardio.loc[cardio["distance_km"] > 0]

    if cardio.empty:
        return pd.DataFrame(columns=["daily_efficiency", "ef_trend"])

    # sum(TRIMP) / sum(km) per day — avoids small-distance bias of per-activity mean
    agg = cardio.groupby("date").agg(
        _sum_trimp=("total_trimp", "sum"),
        _sum_km=("distance_km", "sum"),
    )
    daily_eff = (agg["_sum_trimp"] / agg["_sum_km"]).to_frame("daily_efficiency")

    full_range = pd.date_range(daily_eff.index.min(), daily_eff.index.max(), freq="D")
    daily_eff = daily_eff.reindex(full_range)
    daily_eff.index.name = "date"

    # Rolling trend computed on raw (sparse) data so rest days don't artificially
    # smooth the metric. ffill() is applied afterwards only for display continuity.
    daily_eff = daily_eff.assign(
        ef_trend=daily_eff["daily_efficiency"]
        .rolling(window=EFFICIENCY_TREND_WINDOW, min_periods=3)
        .mean()
        .ffill()
    )
    daily_eff = daily_eff.round(4)
    return daily_eff


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE E – BIO-READINESS SCORE (0–100 %)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_readiness(
    daily: pd.DataFrame, hrv_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Readiness = 0.6 * normalised_HRV_zscore + 0.4 * normalised_TSB.
    HRV Z-Score = (last_night_avg - 30d_rolling_mean) / 30d_rolling_std.
    Result clamped to 0–100.
    """
    daily = daily.copy()

    # ── HRV Z-score ──────────────────────────────────────────────────────
    if hrv_df.empty or "last_night_avg" not in hrv_df.columns:
        daily.loc[:, "readiness_score"] = np.nan
        return daily

    hrv = hrv_df[["date", "last_night_avg"]].copy()
    hrv = hrv.assign(date=pd.to_datetime(hrv["date"]))
    hrv = hrv.set_index("date").sort_index()

    hrv_val = hrv["last_night_avg"].astype(float)
    hrv_30d_mean = hrv_val.rolling(window=30, min_periods=7).mean()
    hrv_30d_std = hrv_val.rolling(window=30, min_periods=7).std().replace(0, np.nan)
    hrv_zscore = (hrv_val - hrv_30d_mean) / hrv_30d_std

    # normalise z-score → 0-100 (clip to ±3)
    hrv_zscore_clipped = hrv_zscore.clip(-3, 3)
    normed_hrv = ((hrv_zscore_clipped + 3) / 6) * 100  # maps -3→0, +3→100

    # ── Normalised TSB ────────────────────────────────────────────────────
    tsb = daily["TSB"].copy()
    tsb_clipped = tsb.clip(-40, 30)
    normed_tsb = ((tsb_clipped + 40) / 70) * 100  # maps -40→0, +30→100

    # ── merge on date index ───────────────────────────────────────────────
    normed_hrv_frame = normed_hrv.to_frame("norm_hrv")
    daily = daily.join(normed_hrv_frame, how="left")

    readiness = 0.6 * daily["norm_hrv"].fillna(50) + 0.4 * normed_tsb
    daily.loc[:, "readiness_score"] = readiness.clip(0, 100).round(1)
    daily = daily.drop(columns=["norm_hrv"], errors="ignore")
    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE F – PURE RECOVERY SCORE (HRV + RHR + Spánek)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_pure_recovery(
    daily: pd.DataFrame,
    hrv_df: pd.DataFrame,
    health_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pure Recovery Score (0–100) kombinuje tři složky:
      • HRV baseline comparison (40 %)  – last_night vs 7d rolling mean
      • RHR deviation (30 %)            – odchylka od 14d baseline
      • Sleep quality (30 %)            – sleep_score normalizovaný na 0–100

    Výstupní sloupce: pure_recovery_score, rhr_day, hrv_status, sleep_score_day
    """
    daily = daily.copy()
    idx = daily.index  # DatetimeIndex

    # ── HRV component (40 %) ─────────────────────────────────────────────
    hrv_norm = pd.Series(np.nan, index=idx, name="hrv_component")
    if not hrv_df.empty and "last_night_avg" in hrv_df.columns:
        hrv = hrv_df[["date", "last_night_avg"]].dropna(subset=["last_night_avg"]).copy()
        hrv = hrv.assign(date=pd.to_datetime(hrv["date"]))
        hrv = hrv.set_index("date").sort_index()
        val = hrv["last_night_avg"].astype(float)
        rolling_mean = val.rolling(window=7, min_periods=3).mean()
        # ratio: 1.0 = on baseline, >1 = better, <1 = worse
        ratio = val / rolling_mean
        # map 0.7–1.3 → 0–100
        hrv_norm_raw = ((ratio.clip(0.7, 1.3) - 0.7) / 0.6) * 100
        hrv_norm = hrv_norm.combine_first(hrv_norm_raw.reindex(idx, method=None))
        # Store last_night and weekly_avg for downstream columns
        daily = daily.join(
            hrv_df.set_index(pd.to_datetime(hrv_df["date"]))[["last_night_avg", "weekly_avg"]]
            .rename(columns={"last_night_avg": "hrv_last_night", "weekly_avg": "hrv_weekly_avg"}),
            how="left",
        )
    else:
        daily.loc[:, "hrv_last_night"] = np.nan
        daily.loc[:, "hrv_weekly_avg"] = np.nan

    # ── RHR component (30 %) ─────────────────────────────────────────────
    rhr_norm = pd.Series(np.nan, index=idx, name="rhr_component")
    if not health_df.empty and "resting_heart_rate" in health_df.columns:
        rhr = health_df[["date", "resting_heart_rate"]].dropna(subset=["resting_heart_rate"]).copy()
        rhr = rhr.assign(date=pd.to_datetime(rhr["date"]))
        rhr = rhr.set_index("date").sort_index()
        rhr_val = rhr["resting_heart_rate"].astype(float)
        rhr_14d = rhr_val.rolling(window=14, min_periods=5).mean()
        # difference from baseline (lower RHR = better recovery)
        diff = rhr_val - rhr_14d  # negative = good
        # map -5..+10 → 100..0
        rhr_norm_raw = (1 - (diff.clip(-5, 10) + 5) / 15) * 100
        rhr_norm = rhr_norm.combine_first(rhr_norm_raw.reindex(idx, method=None))
        # Store rhr_day
        daily = daily.join(
            rhr_val.to_frame("rhr_day"), how="left",
        )
    else:
        daily.loc[:, "rhr_day"] = np.nan

    # ── Sleep component (30 %) ────────────────────────────────────────────
    sleep_norm = pd.Series(np.nan, index=idx, name="sleep_component")
    if not sleep_df.empty and "sleep_score" in sleep_df.columns:
        slp = sleep_df[["date", "sleep_score", "duration_minutes"]].copy()
        slp = slp.assign(date=pd.to_datetime(slp["date"]))
        slp = slp.set_index("date").sort_index()
        # sleep_score je už 0–100
        sleep_norm_raw = slp["sleep_score"].astype(float).clip(0, 100)
        sleep_norm = sleep_norm.combine_first(sleep_norm_raw.reindex(idx, method=None))
        daily = daily.join(
            slp[["sleep_score", "duration_minutes"]]
            .rename(columns={"sleep_score": "sleep_score_day", "duration_minutes": "sleep_duration_min"}),
            how="left",
        )
    else:
        daily.loc[:, "sleep_score_day"] = np.nan
        daily.loc[:, "sleep_duration_min"] = np.nan

    # ── Combined Pure Recovery Score ──────────────────────────────────────
    combined = (
        0.40 * hrv_norm.fillna(50)
        + 0.30 * rhr_norm.fillna(50)
        + 0.30 * sleep_norm.fillna(50)
    )
    daily.loc[:, "pure_recovery_score"] = combined.clip(0, 100).round(1)

    # ── Sleep Performance % (Whoop-style) ────────────────────────────────
    # sleep_need = 450 min (7.5 h base) + 0.5 * TRIMP předchozího dne + recovery_tax
    # sleep_performance = (actual_duration / sleep_need) * 100
    trimp_prev = daily["trimp"].shift(1).fillna(0)  # den předtím (první den = 0, neextrapolujeme)
    tax_prev = daily["recovery_tax_min"].shift(1).fillna(0) if "recovery_tax_min" in daily.columns else 0
    sleep_need = 450 + trimp_prev * 0.5 + tax_prev
    sleep_dur = daily["sleep_duration_min"] if "sleep_duration_min" in daily.columns else pd.Series(np.nan, index=daily.index)
    sleep_perf = (sleep_dur / sleep_need.replace(0, np.nan)) * 100
    daily.loc[:, "sleep_need_min"] = sleep_need.round(0)
    daily.loc[:, "sleep_performance_pct"] = sleep_perf.round(1)

    # ── HRV Coefficient of Variation (7-day) ─────────────────────────────
    if "hrv_last_night" in daily.columns:
        hrv_7d_std = daily["hrv_last_night"].rolling(7, min_periods=3).std()
        hrv_7d_avg = daily["hrv_last_night"].rolling(7, min_periods=3).mean()
        daily.loc[:, "hrv_cv_pct"] = ((hrv_7d_std / hrv_7d_avg.replace(0, np.nan)) * 100).round(1)
    else:
        daily.loc[:, "hrv_cv_pct"] = np.nan

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE I – POLARIZATION SCORE (80/20 Tactic)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_polarization(daily: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    14-day rolling polarization analysis based on HR zone minutes.
    Low intensity = Z1 + Z2,  High intensity = Z4 + Z5.
    Returns polarization_low_pct, polarization_high_pct.
    """
    daily = daily.copy()

    zone_cols = ["time_in_z1", "time_in_z2", "time_in_z3", "time_in_z4", "time_in_z5"]
    if not all(c in master.columns for c in zone_cols):
        daily.loc[:, "polarization_low_pct"] = np.nan
        daily.loc[:, "polarization_high_pct"] = np.nan
        return daily

    m = master[["date"] + zone_cols].copy()
    m = m.assign(date=pd.to_datetime(m["date"]))
    for c in zone_cols:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    m = m.assign(
        low_min=m["time_in_z1"] + m["time_in_z2"],
        mid_min=m["time_in_z3"],
        high_min=m["time_in_z4"] + m["time_in_z5"],
    )
    zone_daily = m.groupby("date")[["low_min", "mid_min", "high_min"]].sum()
    zone_daily = zone_daily.reindex(daily.index, fill_value=0)

    low_14d = zone_daily["low_min"].rolling(14, min_periods=3).sum()
    high_14d = zone_daily["high_min"].rolling(14, min_periods=3).sum()
    # Jmenovatel = celkový čas ve VŠECH zónách (Z1–Z5), aby Z3 „šedá zóna"
    # správně snižovala polarizační skóre
    mid_14d = zone_daily["mid_min"].rolling(14, min_periods=3).sum()
    total_14d = (low_14d + mid_14d + high_14d).replace(0, np.nan)

    daily.loc[:, "polarization_low_pct"] = ((low_14d / total_14d) * 100).round(1)
    daily.loc[:, "polarization_high_pct"] = ((high_14d / total_14d) * 100).round(1)

    # ── Z3 "Junk Miles" penalty ───────────────────────────────────────────────
    # Z3 (155–171 bpm) je aerobně-anaerobní přechodová zóna.  Čas strávený zde
    # snižuje polarizační efektivitu, protože není ani recovery (Z1/Z2) ani
    # vědomě intenzivní (Z4/Z5).
    #
    # z3_pct          : % celkového tréninkového času tvořený Z3 (14denní průměr)
    # polarization_efficiency : jak moc je trénink „čistý" (100 % = nulový Z3)
    #                           pokles o 1 % za každé 1 % z3_pct nad 5 %
    z3_pct = ((mid_14d / total_14d) * 100).round(1)
    daily.loc[:, "z3_junk_pct"] = z3_pct
    # Penalty = lineární srážka za Z3 nad tolerovanou 5% hranici
    z3_penalty = (z3_pct - 5.0).clip(lower=0.0)
    daily.loc[:, "polarization_efficiency"] = (
        (daily["polarization_low_pct"] + daily["polarization_high_pct"] - z3_penalty)
        .clip(lower=0.0, upper=100.0)
        .round(1)
        .clip(lower=0.0)   # strict floor – ensures no negative % after rounding
    )

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE J – VAM (Velocità Ascensionale Media)
# ═══════════════════════════════════════════════════════════════════════════════
# Sports where VAM is a meaningful performance metric (human-powered uphill).
# Gravity sports (alpine_skiing, snowboard, etc.) are intentionally excluded
# because the athlete is being carried up by a lift, making VAM nonsensical.
VAM_SPORTS = (
    "running", "trail_running", "cycling", "gravel_cycling", "mountain_biking",
    "hiking", "walking", "mountaineering", "alpinism", "rock_climbing",
    "cross_country_skiing", "skate_skiing",
)


def compute_vam(master: pd.DataFrame) -> pd.DataFrame:
    """
    VAM = ascent_m / (uphill_minutes / 60) [m/h].

    Uses ``uphill_minutes`` (time actually spent climbing, extracted by the FIT
    parser) so that stopwatch time and descending segments are excluded.
    This gives a true climbing speed rather than an average over the full bout.

    Fallback: if ``uphill_minutes`` is missing or zero, uses
    ``duration_minutes / 2`` as a conservative estimate (assumes roughly half
    the activity time is spent ascending on a typical out-and-back route).

    Only for human-powered sports with ascent_m > 50 and duration > 15 min.
    Gravity-assisted sports (alpine_skiing, snowboard …) are excluded.
    Adds column vam_m_per_h to master.
    """
    master = master.copy()
    if "ascent_m" not in master.columns or "duration_minutes" not in master.columns:
        master.loc[:, "vam_m_per_h"] = np.nan
        return master

    ascent = pd.to_numeric(master["ascent_m"], errors="coerce").fillna(0)
    dur    = pd.to_numeric(master["duration_minutes"], errors="coerce").fillna(0)

    # Prefer uphill_minutes (true climb time from FIT parser); fall back to dur/2
    if "uphill_minutes" in master.columns:
        uphill = pd.to_numeric(master["uphill_minutes"], errors="coerce").fillna(0)
    else:
        uphill = pd.Series(0.0, index=master.index)
    climb_time = uphill.where(uphill > 0, dur / 2.0)

    # Restrict to human-powered sports
    sport_col = master["sport"].str.lower().fillna("")
    is_vam_sport = sport_col.str.contains("|".join(VAM_SPORTS), na=False)

    # Strictly kill VAM for non-VAM sports (gravity sports like alpine
    # skiing can leak through the fallback logic with messy elevation data).
    master.loc[:, "vam_m_per_h"] = np.nan
    if not is_vam_sport.any():
        return master

    # VAM jen u kopcovitých/horských aktivit – průměrný gradient musí být > 4 %
    dist_km_col = pd.to_numeric(master.get("distance_km", pd.Series(dtype=float)), errors="coerce").fillna(0)
    is_climb = (ascent / (dist_km_col * 1000 + 1e-5)) > 0.04
    mask = is_vam_sport & (ascent > 50) & (dur > 15) & (climb_time > 0) & is_climb
    master.loc[:, "vam_m_per_h"] = np.nan
    master.loc[mask, "vam_m_per_h"] = (ascent[mask] / (climb_time[mask] / 60)).round(1)

    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE G – ILLNESS WARNING (multi-indikátorový systém)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_illness_warning(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Vyhodnotí denní varovné signály a generuje illness_warning + stress_flag.

    Signály (4 vlajky – sleeping metrics sloučeny do jedné):
      1. HRV pokles > HRV_DROP_THRESHOLD pod weekly_avg
      2. RHR > HIGH_RHR_THRESHOLD
      3. bad_sleep = nízké sleep score NEBO krátký spánek (sloučeno, 1 vlajka)
      4. Strain v horním kvartilu + nízký recovery

    illness_warning = True pokud ≥ ILLNESS_FLAG_COUNT signálů současně.
    """
    daily = daily.copy()

    flags = pd.DataFrame(index=daily.index)

    # 1. HRV pokles
    if "hrv_last_night" in daily.columns and "hrv_weekly_avg" in daily.columns:
        ln = daily["hrv_last_night"].astype(float)
        wa = daily["hrv_weekly_avg"].astype(float)
        flags["hrv_drop"] = (
            ln.notna() & wa.notna() & (wa > 0)
            & (ln < wa * (1 - HRV_DROP_THRESHOLD))
        )
    else:
        flags["hrv_drop"] = False

    # 2. Vysoký RHR
    if "rhr_day" in daily.columns:
        flags["high_rhr"] = daily["rhr_day"].astype(float) > HIGH_RHR_THRESHOLD
    else:
        flags["high_rhr"] = False

    # 3 + 4. Kombinovaná spánková vlajka – skóre NEBO délka spánku.
    # Sloučení zabraňuje double-countingu: jeden špatný spánek nesmí
    # aktivovat dvě nezávislé vlajky a tím předčasně spustit illness_warning.
    _low_score = (
        (daily["sleep_score_day"].astype(float) < LOW_SLEEP_SCORE)
        if "sleep_score_day" in daily.columns
        else pd.Series(False, index=daily.index)
    )
    _short_dur = (
        (daily["sleep_duration_min"].astype(float) < SHORT_SLEEP_MINUTES)
        if "sleep_duration_min" in daily.columns
        else pd.Series(False, index=daily.index)
    )
    flags["bad_sleep"] = _low_score | _short_dur

    # 5. High Strain + nízký recovery
    #    Aktivuje se, pokud je dnešní strain v horním kvartilu (>75. percentil)
    #    a zároveň pure_recovery_score < 50 %.
    #    Kvartil je počítán na klouzavém 30denním okně, aby odráží aktuální
    #    tréninkovou kondici, nikoli celkové historické maximum.
    if "strain" in daily.columns and "pure_recovery_score" in daily.columns:
        strain_series = daily["strain"].astype(float)
        # rolling(30, min_periods=10): potřebuje aspoň 10 dní pro stabilní kvartil
        strain_q75 = strain_series.rolling(30, min_periods=10).quantile(0.75)
        flags["high_strain"] = (
            (strain_series > strain_q75)
            & (daily["pure_recovery_score"] < 50)
        ).fillna(False)
    else:
        flags["high_strain"] = False

    # Celkový počet varovných signálů
    flag_count = flags.sum(axis=1).fillna(0).astype(int)
    daily.loc[:, "stress_flag_count"] = flag_count
    daily.loc[:, "illness_warning"] = flag_count >= ILLNESS_FLAG_COUNT

    # Textový stress_flag pro ladění
    stress_parts = []
    for col in flags.columns:
        stress_parts.append(flags[col].map({True: col, False: ""}))
    if stress_parts:
        combined_text = pd.concat(stress_parts, axis=1).apply(
            lambda r: " | ".join([x for x in r if x]), axis=1
        )
        daily.loc[:, "stress_flags"] = combined_text
    else:
        daily.loc[:, "stress_flags"] = ""

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE H – COACH ADVICE (textové doporučení)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_coach_advice(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Generuje textový sloupec coach_advice na základě denního Strain,
    Spánku a HRV. Přímo v athlete_readiness.csv.
    """
    daily = daily.copy()
    advices = []

    for idx, row in daily.iterrows():
        parts = []

        # Strain assessment
        strain_val = row.get("strain", np.nan)
        if pd.notna(strain_val):
            if strain_val > 800:
                parts.append("VYSOKÝ STRAIN – priorita je regenerace")
            elif strain_val > 400:
                parts.append("Střední zátěž – sleduj únavu")

        # Sleep
        sleep_score = row.get("sleep_score_day", np.nan)
        sleep_dur = row.get("sleep_duration_min", np.nan)
        if pd.notna(sleep_score) and sleep_score < LOW_SLEEP_SCORE:
            parts.append(f"Špatný spánek ({sleep_score:.0f}/100)")
        if pd.notna(sleep_dur) and sleep_dur < SHORT_SLEEP_MINUTES:
            parts.append(f"Krátký spánek ({sleep_dur:.0f} min)")

        # HRV
        hrv_ln = row.get("hrv_last_night", np.nan)
        hrv_wa = row.get("hrv_weekly_avg", np.nan)
        if pd.notna(hrv_ln) and pd.notna(hrv_wa) and hrv_wa > 0:
            drop_pct = 1 - hrv_ln / hrv_wa
            if drop_pct > HRV_DROP_THRESHOLD:
                parts.append(
                    f"HRV pokles {drop_pct*100:.0f}% pod průměr "
                    f"({hrv_ln:.0f} vs {hrv_wa:.0f} ms)"
                )

        # RHR
        rhr = row.get("rhr_day", np.nan)
        if pd.notna(rhr) and rhr > HIGH_RHR_THRESHOLD:
            parts.append(f"Vysoký RHR ({rhr:.0f} bpm)")

        # Illness warning
        illness = row.get("illness_warning", False)
        if illness:
            parts.insert(0, "⚠️ ILLNESS WARNING – zvažuj odpočinek")

        # Recovery score
        recovery = row.get("pure_recovery_score", np.nan)
        if pd.notna(recovery):
            if recovery >= 80:
                if not parts:
                    parts.append("Výborná regenerace – připraven na trénink")
            elif recovery >= 60:
                if not parts:
                    parts.append("Dobrá regenerace – standardní trénink")
            elif recovery < 40:
                parts.append(f"Nízká regenerace ({recovery:.0f}/100) → snič intenzitu")

        # TSB recommendation
        tsb_val = row.get("TSB", np.nan)
        if pd.notna(tsb_val):
            if tsb_val < -30:
                parts.append("TSB kriticky nízké – odpočinek!")
            elif tsb_val > 10:
                parts.append("Čerstvost – můžeš zvýšit zátěž")

        # ACWR warning
        acwr_val = row.get("acwr", np.nan)
        if pd.notna(acwr_val):
            if acwr_val > 1.5:
                parts.insert(0,
                    f"⚠ ACWR v červené zóně ({acwr_val:.1f}) – sniž objem, nebo hrozí zranění!")
            elif acwr_val > 1.3:
                parts.append(f"ACWR zvýšené ({acwr_val:.1f}) – opatrně s nárůstem")

        # CTL Ramp warning
        ramp_val = row.get("ctl_ramp_rate", np.nan)
        if pd.notna(ramp_val) and ramp_val > CTL_RAMP_WARN:
            parts.append(f"CTL ramp příliš strmý ({ramp_val:+.1f}/týden) – hrozí burn-out!")

        advices.append(" | ".join(parts) if parts else "")

    daily.loc[:, "coach_advice"] = advices
    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY TRIMP AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════
def build_daily_trimp(master: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate TRIMP per calendar day, fill gaps with 0.
    Also builds epoc_weighted_trimp for intensity-aware ACWR.

    Hiking / Walking TRIMP Reduction:
      Activities where sport contains 'hiking' or 'walking' get a linear
      reduction coefficient of 0.6 (−40 %).  This prevents 6–8 hour Z1
      hikes from inflating TRIMP and distorting the PMC (ATL/CTL/TSB).
      Unlike the previous logarithmic soft-cap, a linear coefficient
      preserves the distribution's shape so that Banister's model and
      ACWR remain mathematically sound.

    Returns DataFrame indexed by DatetimeIndex with columns 'trimp', 'trimp_epoc'.
    """
    agg_dict: dict = {"total_trimp": "sum"}
    if "epoc_score" in master.columns:
        agg_dict["epoc_score"] = "sum"
    if "recovery_tax_min" in master.columns:
        agg_dict["recovery_tax_min"] = "sum"

    # Include sport for hiking/walking coefficient; then drop before agg
    _extra_cols = []
    if "sport" in master.columns:
        _extra_cols.append("sport")

    m = master[["date"] + list(agg_dict.keys()) + _extra_cols].copy()
    m["date"] = pd.to_datetime(m["date"])
    for c in agg_dict:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    # ── Lineární koeficient pro pěší sporty ──────────────────────────────────
    # Odstraněno logaritmické tlumení. Aplikujeme lineární koeficient pro
    # hiking/walking, aby PMC (ATL/CTL) neztratilo dynamiku a netrpělo
    # deformací distribuce dat. Redukce TRIMPu o 40 % zabrání inflaci
    # u 6–8hodinových Z1 aktivit, zatímco zachová linearitu pro Banisterův model.
    sport_col = m["sport"].str.lower().fillna("") if "sport" in m.columns else pd.Series("", index=m.index)

    is_hiking = sport_col.str.contains("hiking|walking", na=False)
    if is_hiking.any():
        m.loc[is_hiking, "total_trimp"] = m.loc[is_hiking, "total_trimp"] * 0.6

    # Drop extra cols before aggregation
    m = m.drop(columns=_extra_cols, errors="ignore")

    daily = m.groupby("date").agg(agg_dict).reset_index()
    daily = daily.rename(columns={"total_trimp": "trimp"})
    daily = daily.set_index("date").sort_index()

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "date"

    # EPOC-weighted TRIMP: trimp + 10 % of EPOC score
    if "epoc_score" in daily.columns:
        daily["trimp_epoc"] = daily["trimp"] + daily["epoc_score"] * 0.1
    else:
        daily["trimp_epoc"] = daily["trimp"]

    # Ensure recovery_tax_min column exists (filled with 0 if not in master)
    if "recovery_tax_min" not in daily.columns:
        daily["recovery_tax_min"] = 0.0

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# CTL / ATL / TSB
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ctl_atl_tsb(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily.loc[:, "CTL"] = ema_decay(daily["trimp"], CTL_DAYS)
    daily.loc[:, "ATL"] = ema_decay(daily["trimp"], ATL_DAYS)
    # TSB reflektuje stav PŘED dnešním tréninkem → použij včerejší CTL a ATL
    daily.loc[:, "TSB"] = daily["CTL"].shift(1) - daily["ATL"].shift(1)
    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
def recommend(tsb: float) -> str:
    if tsb < -30:
        return "Vysoké riziko zranění! Okamžitý odpočinek."
    elif tsb < -10:
        return "Optimální tréninková zátěž."
    elif tsb <= 5:
        return "Udržovací režim."
    elif tsb <= 10:
        return "Čerstvost – můžeš zvýšit zátěž."
    else:
        return "Čerstvost / Připravenost na výkon."


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE K – ACWR (Acute:Chronic Workload Ratio)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_acwr(daily: pd.DataFrame) -> pd.DataFrame:
    """
    ACWR = 7d rolling mean / 28d rolling mean.
    Uses EPOC-weighted TRIMP (trimp_epoc) when available for intensity-aware
    load tracking.  Fallback: raw TRIMP.
    Sweet spot 0.8–1.3.  >1.5 = danger zone.
    Also computes CTL Ramp Rate (CTL − CTL_7d_ago > 8 → burn-out warning).
    """
    daily = daily.copy()

    # Prefer EPOC-weighted TRIMP for intensity-aware ACWR
    trimp_col = daily["trimp_epoc"] if "trimp_epoc" in daily.columns else daily["trimp"]

    acute = trimp_col.rolling(ACWR_ACUTE_DAYS, min_periods=ACWR_ACUTE_DAYS).mean()
    chronic = trimp_col.rolling(ACWR_CHRONIC_DAYS, min_periods=ACWR_CHRONIC_DAYS).mean()
    # Raw mathematical ACWR – no artificial clipping so the database
    # reflects true overtraining danger (values > 2.0 or < 0.5 are
    # legitimate signals that downstream consumers should interpret).
    daily.loc[:, "acwr"] = (acute / chronic.replace(0, np.nan)).round(2)

    # CTL Ramp Rate – weekly CTL delta
    if "CTL" in daily.columns:
        daily.loc[:, "ctl_ramp_rate"] = (daily["CTL"] - daily["CTL"].shift(7)).round(2)
        daily.loc[:, "ctl_ramp_warning"] = daily["ctl_ramp_rate"] > CTL_RAMP_WARN
    else:
        daily.loc[:, "ctl_ramp_rate"] = np.nan
        daily.loc[:, "ctl_ramp_warning"] = False

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE L – EXTERNAL vs INTERNAL LOAD  (Skrytá únava)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_fatigue_index(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Porovnání mechanické práce (speed/distance) vs. interní reakce (HR).
    Efficiency = daily_efficiency (TRIMP / km).  Nižší = lepší ekonomie.
    fatigue_index = today's efficiency / 7d rolling avg efficiency.
    > 1.0 → organismus reaguje hůř na stejnou práci → skrytá únava.
    """
    daily = daily.copy()
    if "daily_efficiency" not in daily.columns:
        daily.loc[:, "fatigue_index"] = np.nan
        return daily

    eff = daily["daily_efficiency"].copy()
    eff_7d = eff.rolling(7, min_periods=3).mean()
    daily.loc[:, "fatigue_index"] = (eff / eff_7d.replace(0, np.nan)).round(3)
    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE M – FUELING MODEL  (Fat vs Carb kcal by zone)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_fueling(master: pd.DataFrame) -> pd.DataFrame:
    """
    Na základě času v HR zónách odhadne spalování tuků vs. cukrů (kcal).
    Přidá sloupce fat_kcal, carb_kcal, fat_g, carb_g do master.
    """
    master = master.copy()
    zone_cols = {"time_in_z1": "Z1", "time_in_z2": "Z2", "time_in_z3": "Z3",
                 "time_in_z4": "Z4", "time_in_z5": "Z5"}

    fat_kcal = pd.Series(0.0, index=master.index)
    carb_kcal = pd.Series(0.0, index=master.index)

    for col, zone in zone_cols.items():
        if col not in master.columns:
            continue
        mins = pd.to_numeric(master[col], errors="coerce").fillna(0)
        total_kcal_zone = mins * KCAL_PER_MIN_BY_ZONE.get(zone, 8.0)
        fat_pct = FAT_PCT_BY_ZONE.get(zone, 0.5)
        fat_kcal += total_kcal_zone * fat_pct
        carb_kcal += total_kcal_zone * (1 - fat_pct)

    master.loc[:, "fat_kcal"] = fat_kcal.round(1)
    master.loc[:, "carb_kcal"] = carb_kcal.round(1)
    master.loc[:, "fat_g"] = (fat_kcal / KCAL_PER_G_FAT).round(1)
    master.loc[:, "carb_g"] = (carb_kcal / KCAL_PER_G_CARB).round(1)
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE N – FLUID LOSS ESTIMATE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_fluid_loss(master: pd.DataFrame) -> pd.DataFrame:
    """
    Odhad ztráty tekutin:
      fluid_loss_L = duration_h * (0.6 + max(0, avg_temp - 20) * 0.1)
    """
    master = master.copy()
    dur_h = pd.to_numeric(master["duration_minutes"], errors="coerce").fillna(0) / 60.0
    avg_temp = pd.to_numeric(master.get("avg_temp", pd.Series(dtype=float)), errors="coerce").fillna(20.0)
    heat_factor = (avg_temp - 20).clip(lower=0) * 0.1
    master.loc[:, "fluid_loss_L"] = (dur_h * (0.6 + heat_factor)).round(2)
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE O – TEMP-EFFECT MODEL  (heat vs cardiac drift correlation)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_temp_effect(master: pd.DataFrame) -> pd.DataFrame:
    """
    Přidá sloupec heat_flag (avg_temp > 25 °C) a korelaci teplota × drift.
    hot_drift_avg: průměr cardiac drift v horku.
    cool_drift_avg: průměr cardiac drift při normální teplotě.
    """
    master = master.copy()
    if "avg_temp" not in master.columns or "cardiac_drift" not in master.columns:
        master.loc[:, "heat_flag"] = False
        return master

    avg_temp = pd.to_numeric(master["avg_temp"], errors="coerce")
    master.loc[:, "heat_flag"] = avg_temp > 25.0

    hot = master.loc[master["heat_flag"] & master["cardiac_drift"].notna(), "cardiac_drift"]
    cool = master.loc[~master["heat_flag"] & master["cardiac_drift"].notna(), "cardiac_drift"]

    hot_avg = hot.mean() if len(hot) >= 3 else np.nan
    cool_avg = cool.mean() if len(cool) >= 3 else np.nan

    if not np.isnan(hot_avg) and not np.isnan(cool_avg):
        log.info(
            "Temp-Effect: hot_drift_avg=%.1f%%  cool_drift_avg=%.1f%%  "
            "delta=%.1f pp",
            hot_avg, cool_avg, hot_avg - cool_avg,
        )
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE P – DURABILITY  (EF decay in long activities)
# ═══════════════════════════════════════════════════════════════════════════════
def durability_for_activity(tdata: pd.DataFrame, sport: str) -> Optional[float]:
    """
    Pro aktivity delší než 2 h porovná EF v 1. a 2. polovině.
    Pro cyklistiku: EF = power / HR (speed nedává smysl kvůli terénu).
    Pro běh: EF = speed / HR.
    Durability = (EF_half2 / EF_half1 - 1) * 100  [%].
    Záporné číslo = pokles = únava.
    """
    if tdata.empty:
        return None

    is_cycling = "cycling" in sport or "bike" in sport
    effort_col = "power" if is_cycling else "speed"

    # check effort column exists
    if effort_col not in tdata.columns:
        return None

    active = tdata.loc[
        (tdata["is_active"] == True)
        & tdata["heart_rate"].notna()
        & (tdata["heart_rate"] > 0)
        & tdata[effort_col].notna()
        & (tdata[effort_col] > 0)
    ].copy()
    if len(active) < 20:
        return None

    t0 = active["timestamp"].iloc[0]
    t_end = active["timestamp"].iloc[-1]
    dur_min = (t_end - t0).total_seconds() / 60.0
    if dur_min < DURABILITY_MIN_DURATION:
        return None

    t_mid = t0 + (t_end - t0) / 2
    h1 = active[active["timestamp"] <= t_mid]
    h2 = active[active["timestamp"] > t_mid]
    if h1.empty or h2.empty:
        return None

    hr1, hr2 = h1["heart_rate"].mean(), h2["heart_rate"].mean()
    eff1, eff2 = h1[effort_col].mean(), h2[effort_col].mean()
    if hr1 == 0 or hr2 == 0:
        return None

    # Guard: reject near-zero effort segments
    if is_cycling:
        if eff1 < 30 or eff2 < 30:
            return None
    else:
        if eff1 < 0.56 or eff2 < 0.56:
            return None

    ef1 = eff1 / hr1
    ef2 = eff2 / hr2
    if ef1 == 0:
        return None
    durability = (ef2 / ef1 - 1) * 100.0
    return round(durability, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE Q – TTE  (Time to Exhaustion estimate)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_tte(master: pd.DataFrame) -> pd.DataFrame:
    """
    Odhad TTE = jak dlouho dokáže atlet vydržet na úrovni Z4/Z5.
    Použijeme historickou HR-Duration křivku: pro každou aktivitu se podíváme
    kolik minut strávil v Z4+Z5 a jak dlouhá byla aktivita.
    TTE = max(time_in_z4 + time_in_z5) ze všech aktivit (historický max).
    Přidá sloupec tte_z4z5_min do master.
    """
    master = master.copy()
    z4 = pd.to_numeric(master.get("time_in_z4", pd.Series(dtype=float)), errors="coerce").fillna(0)
    z5 = pd.to_numeric(master.get("time_in_z5", pd.Series(dtype=float)), errors="coerce").fillna(0)
    master.loc[:, "time_at_threshold_min"] = (z4 + z5).round(2)
    # Historical maximum TTE across all activities (expanding window)
    master.loc[:, "tte_z4z5_min"] = master["time_at_threshold_min"].expanding().max().round(2)
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE R – EPOC & Oxygen Debt
# ═══════════════════════════════════════════════════════════════════════════════
def compute_epoc(master: pd.DataFrame) -> pd.DataFrame:
    """
    Akumulovaný kyslíkový dluh.

    Recovery tax (extra sleep need in minutes):
      Exponential model: 15 × exp(Z5_min / 15)
      • At Z5=0 min  → 15 min baseline (residual EPOC from sub-threshold effort)
      • At Z5=10 min → ≈41 min  (vs ≈15 min with the old linear model)
      • At Z5=30 min → ≈111 min (reflects true oxygen-debt accumulation curve)
    The exponential shape mirrors the non-linear relationship between high-
    intensity duration and post-exercise oxygen consumption documented in
    exercise physiology literature (excess oxygen debt grows super-linearly).

    EPOC proxy score = time_in_z4 * 2 + time_in_z5 * 5 (arbitrary intensity units).
    """
    master = master.copy()
    z4 = pd.to_numeric(master.get("time_in_z4", pd.Series(dtype=float)), errors="coerce").fillna(0)
    z5 = pd.to_numeric(master.get("time_in_z5", pd.Series(dtype=float)), errors="coerce").fillna(0)
    master.loc[:, "epoc_score"] = (z4 * 2 + z5 * 5).round(1)
    # Exponential recovery tax: 15 × e^(z5/15)
    # Strict physiological cap at 240 min (4 h) – the raw exponential
    # explodes for long Z5 durations (e.g. 60 min → 13+ h), which is
    # physiologically impossible for recovery-sleep demand.
    master.loc[:, "recovery_tax_min"] = (15.0 * np.exp(z5 / 15.0)).clip(upper=240.0).round(0)
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE S – DFA-alpha1 PROXY  (Aerobní práh z linearity HR vs Speed)
# ═══════════════════════════════════════════════════════════════════════════════
def dfa_alpha1_proxy_for_activity(tdata: pd.DataFrame) -> Optional[int]:
    """
    Proxy odhad aerobního prahu (AeT) jako bod, kde se láme linearita
    nárůstu tepu vůči rychlosti.

    Metoda: seřadíme aktivní záznamy dle rychlosti, rozdělíme na segmenty,
    spočítáme slope HR/speed. AeT ≈ HR, kde slope prudce roste.

    Pokud jsou dostupná HRV data (sloupec 'hrv'), detekujeme pokles variability.

    Returns estimated AeT HR (bpm) nebo None.
    """
    if tdata.empty:
        return None

    # Pokus o přímou detekci z HRV dat (pokles variability)
    if "hrv" in tdata.columns and tdata["hrv"].notna().sum() > 60:
        df = tdata[["heart_rate", "hrv"]].dropna().copy()
        if len(df) > 100:
            df = df.sort_values("heart_rate")
            # Klouzavý průměr HRV po HR
            df["hr_bin"] = (df["heart_rate"] // 5) * 5
            binned = df.groupby("hr_bin").agg(
                hrv_mean=("hrv", "mean"),
                count=("hrv", "count"),
            )
            binned = binned[binned["count"] >= 3]
            if len(binned) >= 4:
                # Najdi bod kde HRV klesne pod 50% maxima
                hrv_max = binned["hrv_mean"].max()
                threshold = hrv_max * 0.5
                below = binned[binned["hrv_mean"] < threshold]
                if not below.empty:
                    aet_hr = int(below.index[0])
                    # Physiological cap: AeT/VT1 cannot exceed 80% of MaxHR.
                    # Noisy optical HR data can create fake breakpoints
                    # at threshold levels that are physiologically impossible.
                    if aet_hr > (ATHLETE_MAX_HR * 0.80):
                        return None
                    # Stejná ochrana proti šumu jako u speed-based větve
                    if 130 <= aet_hr <= (ATHLETE_MAX_HR * 0.85):
                        return aet_hr

    # Fallback: HR vs Speed linearity breakpoint
    active = tdata.loc[
        (tdata["is_active"] == True)
        & tdata["heart_rate"].notna()
        & (tdata["heart_rate"] > 0)
        & tdata["speed"].notna()
        & (tdata["speed"] > 0.5)  # > 1.8 km/h – keeps steep-climb cyclist records
    ].copy()
    if len(active) < 100:
        return None

    # ── Gradient filter (±2 %) ────────────────────────────────────────────
    # On uphills/downhills the HR vs Speed relationship breaks: HR is elevated
    # or suppressed by gravity independently of aerobic output.  Keeping only
    # near-flat segments ensures the linearity breakpoint reflects true AeT.
    if "altitude" in active.columns and "distance" in active.columns:
        _alt  = pd.to_numeric(active["altitude"],  errors="coerce")
        _dist = pd.to_numeric(active["distance"],  errors="coerce")
        _delta_alt  = _alt.diff()
        _delta_dist = _dist.diff()
        _gradient_pct = (_delta_alt / _delta_dist.replace(0, np.nan)) * 100.0
        # Keep rows where gradient is within ±2 % or cannot be computed
        _flat_mask = _gradient_pct.abs() <= 2.0
        active = active[_flat_mask.fillna(True)]
        if len(active) < 100:
            return None

    active = active.sort_values("speed")
    n = len(active)
    n_segments = min(10, n // 20)
    if n_segments < 4:
        return None

    seg_size = n // n_segments
    slopes = []
    hr_mids = []
    for i in range(n_segments - 1):
        s1 = active.iloc[i * seg_size : (i + 1) * seg_size]
        s2 = active.iloc[(i + 1) * seg_size : (i + 2) * seg_size]
        dhr = s2["heart_rate"].mean() - s1["heart_rate"].mean()
        dsp = s2["speed"].mean() - s1["speed"].mean()
        # Require a meaningful absolute speed delta to avoid division by
        # near-zero values (constant-pace segments, sensor jitter).
        # abs() so that a slight speed *decrease* between segments doesn't
        # silently produce a negative / ∞ slope.
        if abs(dsp) > 0.05:
            slopes.append(dhr / dsp)
            hr_mids.append(s2["heart_rate"].mean())

    if len(slopes) < 3:
        return None

    # Najdi index kde slope nejvíc roste (breakpoint)
    slope_diffs = [slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)]
    max_diff_idx = int(np.argmax(slope_diffs))
    aet_hr = int(hr_mids[max_diff_idx])

    # Physiological cap: AeT/VT1 cannot exceed 80% of MaxHR.
    # Noisy optical HR sensors may produce fake breakpoints above this limit.
    if aet_hr > (ATHLETE_MAX_HR * 0.80):
        return None

    # Sanity check
    # Spodní limit 130 bpm (pod Z2 nemůže být AeT) a horní limit 85 % MaxHR
    if aet_hr < 130 or aet_hr > (ATHLETE_MAX_HR * 0.85):
        return None  # šum nebo falešná detekce

    return aet_hr


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE T – CLIMB SCORE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_climb_score(master: pd.DataFrame) -> pd.DataFrame:
    """
    avg_gradient_pct = (ascent_m / (distance_km * 1000)) * 100
    Kategorie: Flat (<1%), Rolling (1-3%), Hilly (3-6%), Mountainous (>6%).
    """
    master = master.copy()
    ascent = pd.to_numeric(master.get("ascent_m", pd.Series(dtype=float)), errors="coerce").fillna(0)
    dist_km = pd.to_numeric(master.get("distance_km", pd.Series(dtype=float)), errors="coerce").fillna(0)
    dist_m = dist_km * 1000.0

    gradient = (ascent / dist_m.replace(0, np.nan)) * 100
    master.loc[:, "avg_gradient_pct"] = gradient.round(2)

    def _categorise(g):
        if pd.isna(g):
            return ""
        if g < 1.0:
            return "Flat"
        elif g < 3.0:
            return "Rolling"
        elif g < 6.0:
            return "Hilly"
        else:
            return "Mountainous"

    master.loc[:, "climb_category"] = master["avg_gradient_pct"].apply(_categorise)
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE U – REAL DFA-alpha1  (neurokit2-powered threshold detection)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dfa_alpha1_for_activity(
    activity_id: str, tdata: pd.DataFrame,
) -> dict:
    """
    Real DFA-alpha1 computation using neurokit2.

    Extracts R-R intervals from FIT file, computes DFA-alpha1 in sliding
    windows of 200 beats, maps to concurrent HR, and detects:
      - AeT (VT1): HR where alpha1 crosses 0.75
      - AnT (VT2): HR where alpha1 crosses 0.50

    Falls back to proxy (Module S) if R-R data insufficient.

    Returns dict with keys: aet_hr_dfa, ant_hr_dfa, dfa_quality
    """
    result = {"aet_hr_dfa": None, "ant_hr_dfa": None, "dfa_quality": "none"}

    if not HAS_NEUROKIT:
        proxy = dfa_alpha1_proxy_for_activity(tdata)
        if proxy is not None:
            result["aet_hr_dfa"] = proxy
            result["dfa_quality"] = "proxy"
        return result

    # Extract R-R intervals from FIT
    rr_s = extract_rr_for_activity(activity_id)
    if len(rr_s) < DFA_WINDOW_BEATS:
        proxy = dfa_alpha1_proxy_for_activity(tdata)
        if proxy is not None:
            result["aet_hr_dfa"] = proxy
            result["dfa_quality"] = "proxy"
        return result

    # Clean R-R intervals
    rr_ms = clean_rr_intervals(rr_s)
    if len(rr_ms) < DFA_WINDOW_BEATS:
        proxy = dfa_alpha1_proxy_for_activity(tdata)
        if proxy is not None:
            result["aet_hr_dfa"] = proxy
            result["dfa_quality"] = "proxy"
        return result

    # ── Windowed DFA-alpha1 computation ───────────────────────────────────
    hr_alpha_pairs: list[tuple[float, float]] = []

    for start in range(0, len(rr_ms) - DFA_WINDOW_BEATS + 1, DFA_SLIDE_BEATS):
        window = rr_ms[start : start + DFA_WINDOW_BEATS]

        # Skip windows with too much variance (artifact-contaminated)
        cv = np.std(window) / np.mean(window)
        if cv > 0.25:
            continue

        # Average HR for this window
        avg_rr = np.mean(window)
        avg_hr = 60000.0 / avg_rr

        try:
            alpha1, _ = nk.fractal_dfa(window, windows=DFA_BOX_SIZES)
            if np.isfinite(alpha1) and 0.0 < alpha1 < 2.0:
                hr_alpha_pairs.append((avg_hr, alpha1))
        except Exception:
            continue

    if len(hr_alpha_pairs) < 10:
        proxy = dfa_alpha1_proxy_for_activity(tdata)
        if proxy is not None:
            result["aet_hr_dfa"] = proxy
            result["dfa_quality"] = "proxy"
        return result

    # ── Sort by HR and smooth ─────────────────────────────────────────────
    pairs = sorted(hr_alpha_pairs, key=lambda x: x[0])
    hrs = np.array([p[0] for p in pairs])
    alphas = np.array([p[1] for p in pairs])

    kernel = min(5, max(3, len(alphas) // 3))
    alphas_smooth = uniform_filter1d(alphas, size=kernel) if kernel >= 3 else alphas

    # ── Find AeT: alpha1 crosses 0.75 downward ───────────────────────────
    aet_hr = None
    for i in range(len(alphas_smooth) - 1):
        if (alphas_smooth[i] >= DFA_AET_THRESHOLD
                and alphas_smooth[i + 1] < DFA_AET_THRESHOLD):
            denom = alphas_smooth[i + 1] - alphas_smooth[i]
            if denom != 0:
                frac = (DFA_AET_THRESHOLD - alphas_smooth[i]) / denom
                aet_hr = hrs[i] + frac * (hrs[i + 1] - hrs[i])
            break

    # ── Find AnT: alpha1 crosses 0.50 downward ───────────────────────────
    ant_hr = None
    for i in range(len(alphas_smooth) - 1):
        if (alphas_smooth[i] >= DFA_ANT_THRESHOLD
                and alphas_smooth[i + 1] < DFA_ANT_THRESHOLD):
            denom = alphas_smooth[i + 1] - alphas_smooth[i]
            if denom != 0:
                frac = (DFA_ANT_THRESHOLD - alphas_smooth[i]) / denom
                ant_hr = hrs[i] + frac * (hrs[i + 1] - hrs[i])
            break

    # Sanity checks
    # Physiological cap: AeT/VT1 cannot exceed 80% of MaxHR.
    if aet_hr is not None and aet_hr > (ATHLETE_MAX_HR * 0.80):
        aet_hr = None
    if aet_hr is not None and (aet_hr < 100 or aet_hr > ATHLETE_MAX_HR):
        aet_hr = None
    if ant_hr is not None and (ant_hr < 100 or ant_hr > ATHLETE_MAX_HR):
        ant_hr = None
    if aet_hr is not None and ant_hr is not None and ant_hr <= aet_hr:
        ant_hr = None  # AnT must be higher than AeT

    result["aet_hr_dfa"] = int(aet_hr) if aet_hr else None
    result["ant_hr_dfa"] = int(ant_hr) if ant_hr else None
    result["dfa_quality"] = "real"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE V – RESPIRATION RATE FROM HRV  (RSA-based)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_resp_from_rr(activity_id: str) -> Optional[float]:
    """
    Compute respiration rate from R-R intervals using Respiratory Sinus
    Arrhythmia (RSA).

    The HF component (0.15–0.50 Hz) of R-R interval variability corresponds
    to respiratory modulation of heart rate.  Peak frequency in this band
    × 60 = breaths/min.

    Returns breaths/min or None.
    """
    if not HAS_NEUROKIT:
        return None

    rr_s = extract_rr_for_activity(activity_id)
    if len(rr_s) < 120:  # need ≥ ~2 min of data
        return None

    rr_ms = clean_rr_intervals(rr_s)
    if len(rr_ms) < 120:
        return None

    try:
        # Create time axis from cumulative R-R
        rr_times = np.cumsum(rr_ms) / 1000.0  # seconds

        # Resample to uniform 4 Hz
        fs = 4.0
        f_interp = interp1d(rr_times, rr_ms, kind="cubic", fill_value="extrapolate")
        t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / fs)
        rr_uniform = f_interp(t_uniform)

        if len(rr_uniform) < 64:
            return None

        # Detrend
        rr_detrended = rr_uniform - np.mean(rr_uniform)

        # Welch PSD
        nperseg = min(256, len(rr_detrended) // 2)
        if nperseg < 32:
            return None
        frequencies, psd = welch(rr_detrended, fs=fs, nperseg=nperseg)

        # Find peak in respiratory band (0.15–0.50 Hz = 9–30 breaths/min)
        resp_mask = (frequencies >= 0.15) & (frequencies <= 0.50)
        if not np.any(resp_mask):
            return None

        resp_psd = psd[resp_mask]
        resp_freqs = frequencies[resp_mask]
        peak_idx = int(np.argmax(resp_psd))
        resp_rate_bpm = resp_freqs[peak_idx] * 60.0

        # Sanity check
        if 6.0 <= resp_rate_bpm <= 40.0:
            return round(resp_rate_bpm, 1)
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE W – TATI (TIME ABOVE THRESHOLD IMPULSE) & CRITICAL HEART RATE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_critical_hr_model(master: pd.DataFrame) -> pd.DataFrame:
    """
    Critical Heart Rate (CHR) and TATI (Time Above Threshold Impulse).

    Monod-Scherrer model adapted for HR:
      time_lim = TATI / (HR_avg − CHR)

    CHR = highest HR sustainable for ≥30 min (85th percentile of avg_hr
          in long steady-state activities).
    TATI = accumulated bpm·min above CHR per activity (replaces the former
           'w_prime' metric – W' in Watts is a power-domain concept and
           computing it from HR alone is methodically incorrect).

    Adds columns: critical_hr, tati_score
    """
    master = master.copy()

    # Eligible: cardio with valid avg_hr
    cardio_mask = master["sport"].str.contains(
        "|".join(CARDIO_SPORTS), case=False, na=False
    )
    avg_hr = pd.to_numeric(master["avg_hr"], errors="coerce")
    dur_min = pd.to_numeric(master["duration_minutes"], errors="coerce")

    eligible = master.loc[cardio_mask & avg_hr.notna() & (dur_min >= 20)].copy()

    if len(eligible) < WPRIME_MIN_ACTIVITIES:
        master["critical_hr"] = np.nan
        master["tati_score"] = np.nan
        return master

    # Estimate CHR: 85th percentile of avg_hr in long (≥30 min) steady-state
    # efforts.  Using the max() was flawed: a single all-out 30-min race
    # permanently inflates CHR and causes W' to be undercalculated.
    # 85th percentile retains the "best sustainable" intent while ignoring the
    # top 15 % of outlier race efforts.
    # Hard cap at 85 % MaxHR as a physiological sanity bound.
    elig_hr = pd.to_numeric(eligible["avg_hr"], errors="coerce")
    elig_dur = pd.to_numeric(eligible["duration_minutes"], errors="coerce")
    long_30_mask = elig_dur >= 30

    if long_30_mask.any() and elig_hr[long_30_mask].notna().any():
        chr_hr = float(elig_hr[long_30_mask].quantile(0.85))
        # Hard physiological cap: nobody's CHR should exceed 85 % MaxHR
        chr_hr = min(chr_hr, 0.85 * ATHLETE_MAX_HR)
    else:
        chr_hr = 0.85 * ATHLETE_MAX_HR

    chr_hr = round(float(chr_hr))

    # Zone boundaries for W' estimation
    hrr = ATHLETE_MAX_HR - ATHLETE_RHR
    z4_lo = round(0.82 * hrr + ATHLETE_RHR)
    z5_lo = round(0.90 * hrr + ATHLETE_RHR)
    avg_z4_hr = (z4_lo + z5_lo) / 2.0
    avg_z5_hr = (z5_lo + ATHLETE_MAX_HR) / 2.0

    # TATI per activity = time above CHR × excess HR
    z4_min = pd.to_numeric(
        master.get("time_in_z4", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    z5_min = pd.to_numeric(
        master.get("time_in_z5", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)

    tati_score = (
        z4_min * max(0, avg_z4_hr - chr_hr)
        + z5_min * max(0, avg_z5_hr - chr_hr)
    )

    master["critical_hr"] = chr_hr
    master["tati_score"] = tati_score.round(1)

    valid_tati = (tati_score > 0).sum()
    log.info(
        "Critical HR model: CHR=%d bpm, TATI computed for %d activities",
        chr_hr, valid_tati,
    )
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE EXTENDED PER-ACTIVITY METRICS  (Durability, DFA, Fueling, …)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_extended_per_activity(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rozšířený výpočet per-aktivity metrik:
    Durability + DFA-alpha1 (real via neurokit2 or proxy) vyžadují vteřinová data.
    Respiration Rate from RSA + W'/Critical HR vyžadují FIT R-R data.
    Fueling, Fluid Loss, TTE, EPOC, Climb Score pracují přímo s master.
    """
    master = master.copy()

    # ── Masové metriky (nevyžadují vteřinová data) ────────────────────────
    print("  [M] Fueling Model (fat/carb kcal)...")
    master = compute_fueling(master)

    print("  [N] Fluid Loss Estimate...")
    master = compute_fluid_loss(master)

    print("  [O] Temp-Effect Model...")
    master = compute_temp_effect(master)

    print("  [Q] TTE (Time to Exhaustion estimate)...")
    master = compute_tte(master)

    print("  [R] EPOC & Oxygen Debt...")
    master = compute_epoc(master)

    print("  [T] Climb Score...")
    master = compute_climb_score(master)

    print("  [W] W' & Critical Heart Rate...")
    master = compute_critical_hr_model(master)

    # ── Per-activity metriky vyžadující vteřinová data + FIT R-R ──────────
    for col in ("durability_pct", "aet_hr_dfa", "ant_hr_dfa",
                "dfa_quality", "resp_rate_rsa"):
        if col not in master.columns:
            master[col] = np.nan if col != "dfa_quality" else ""

    # Keep legacy column for backward compat
    if "aet_hr_proxy" not in master.columns:
        master["aet_hr_proxy"] = np.nan

    if not os.path.exists(TRAINING_CSV):
        print("  [WARN] High-res training data not found – skipping per-activity.")
        return master

    # Eligible: cardio, ≥ 20 min
    is_cardio = master["sport"].str.contains(
        "|".join(CARDIO_SPORTS), case=False, na=False
    )
    long_enough = master["duration_minutes"] >= MIN_DURATION_DRIFT
    eligible = master.loc[is_cardio & long_enough].copy()

    if eligible.empty:
        return master

    aid_set = set(eligible["activity_id"].astype(str).tolist())
    n_elig = len(aid_set)
    dfa_mode = "neurokit2 (real)" if HAS_NEUROKIT else "proxy"
    print(f"  [P+U+V] Durability, DFA-alpha1 ({dfa_mode}), Resp RSA: "
          f"{n_elig} aktivit...")

    tdata_map = load_training_data_bulk(aid_set)

    dur_results: dict = {}
    aet_results: dict = {}
    ant_results: dict = {}
    quality_results: dict = {}
    resp_results: dict = {}
    proxy_results: dict = {}

    for i, (idx, row) in enumerate(eligible.iterrows(), 1):
        aid = str(row["activity_id"])
        tdata = tdata_map.get(aid, pd.DataFrame())

        # Durability
        sport = str(row["sport"]).lower()
        dur = durability_for_activity(tdata, sport)
        dur_results[idx] = dur

        # DFA-alpha1 (real or proxy)
        dfa = compute_dfa_alpha1_for_activity(aid, tdata)
        aet_results[idx] = dfa["aet_hr_dfa"]
        ant_results[idx] = dfa["ant_hr_dfa"]
        quality_results[idx] = dfa["dfa_quality"]

        # Legacy proxy column (always computed for comparison)
        proxy_results[idx] = dfa_alpha1_proxy_for_activity(tdata) if dfa["dfa_quality"] == "real" else dfa["aet_hr_dfa"]

        # Respiration Rate from RSA
        resp = compute_resp_from_rr(aid)
        resp_results[idx] = resp

        if i % 50 == 0 or i == n_elig:
            print(
                f"    [{i}/{n_elig}]  {row['date']}  "
                f"dfa={dfa['dfa_quality']}  aet={dfa['aet_hr_dfa']}  "
                f"ant={dfa['ant_hr_dfa']}  resp={resp}"
            )

    # Bulk column updates
    if dur_results:
        master.loc[list(dur_results.keys()), "durability_pct"] = list(dur_results.values())
        master.loc[list(aet_results.keys()), "aet_hr_dfa"] = list(aet_results.values())
        master.loc[list(ant_results.keys()), "ant_hr_dfa"] = list(ant_results.values())
        master.loc[list(quality_results.keys()), "dfa_quality"] = list(quality_results.values())
        master.loc[list(resp_results.keys()), "resp_rate_rsa"] = list(resp_results.values())
        master.loc[list(proxy_results.keys()), "aet_hr_proxy"] = list(proxy_results.values())

    dur_count = master["durability_pct"].notna().sum()
    dfa_real = (master["dfa_quality"] == "real").sum()
    dfa_proxy = (master["dfa_quality"] == "proxy").sum()
    resp_count = master["resp_rate_rsa"].notna().sum()
    tati_count = (pd.to_numeric(master["tati_score"], errors="coerce") > 0).sum()
    print(f"  ✓ Durability: {dur_count}  |  DFA real: {dfa_real}  proxy: {dfa_proxy}")
    print(f"  ✓ Resp RSA: {resp_count}  |  TATI > 0: {tati_count} aktivit")

    return master


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-CHECK: verify_analytics()
# ═══════════════════════════════════════════════════════════════════════════════
def verify_analytics(
    daily: pd.DataFrame,
    master: pd.DataFrame,
) -> None:
    """
    Produkční self-check:
    0. Produkční režim potvrzení.
    1. Ověření readiness_score v rozsahu 0–100.
    2. Nejlepší max_hrr_60s ze všech aktivit.
    3. Top den s nejvyšším whoop_strain (z 21).
    4. Průměrný sleep_performance_pct.
    """
    print()
    print("=" * 64)
    print("  SELF-CHECK  –  verify_analytics()")
    print("=" * 64)

    # 0. Production mode
    print(f"  📊 PRODUKČNÍ REŽIM: Zpracována kompletní databáze.")
    print(f"     Aktivit v master: {len(master)}  |  Denních záznamů: {len(daily)}")

    # 0b. Karvonen zóny – verifikace nových konstant (MaxHR=199, RHR=41, HRR=158)
    _hrr = ATHLETE_MAX_HR - ATHLETE_RHR
    _zone_labels = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    _zone_pcts   = ATHLETE_ZONE_PCTS
    print()
    print(f"  ── KARVONEN ZÓNY  (MaxHR={ATHLETE_MAX_HR}, RHR={ATHLETE_RHR}, HRR={_hrr}) ──────────────")
    for _i, _lbl in enumerate(_zone_labels):
        _lo = round(_zone_pcts[_i]     * _hrr + ATHLETE_RHR)
        _hi = round(_zone_pcts[_i + 1] * _hrr + ATHLETE_RHR)
        _bar = "■" * int((_zone_pcts[_i + 1] - _zone_pcts[_i]) * 50)
        print(f"    {_lbl}: {_lo:3d}–{_hi:3d} bpm  [{_bar}]")
    print()

    # 1. Readiness score range
    if "readiness_score" in daily.columns:
        r = daily["readiness_score"].dropna()
        if not r.empty:
            rmin, rmax = r.min(), r.max()
            ok = rmin >= 0 and rmax <= 100
            symbol = "✓" if ok else "✗"
            print(f"  {symbol} Readiness Score rozsah: {rmin:.1f} – {rmax:.1f}  (očekáváno 0–100)")
        else:
            print("  ⚠ Readiness Score: žádné platné hodnoty")
    else:
        print("  ✗ Readiness Score: sloupec nenalezen")

    # 2. Nejlepší max_hrr_60s
    if "max_hrr_60s" in master.columns:
        hrr_valid = master["max_hrr_60s"].dropna()
        if not hrr_valid.empty:
            best_idx = hrr_valid.idxmax()
            best_val = hrr_valid.max()
            best_row = master.loc[best_idx]
            print(
                f"  🏆 Nejlepší Max HRR 60s: {best_val:.1f} bpm  "
                f"({best_row['date']}  {best_row['sport']}  "
                f"{best_row['duration_minutes']:.0f} min)"
            )
        else:
            print("  ⚠ Max HRR 60s: žádné platné hodnoty")
    else:
        print("  ✗ max_hrr_60s: sloupec nenalezen v master")

    # 3. Top den s nejvyšším whoop_strain
    if "whoop_strain" in daily.columns:
        ws = daily["whoop_strain"].dropna()
        if not ws.empty:
            top_idx = ws.idxmax()
            top_val = ws.max()
            d = top_idx.date() if hasattr(top_idx, "date") else top_idx
            print(f"  🔥 Top Whoop Strain den: {d}  →  {top_val:.2f} / 21.00")
        else:
            print("  ⚠ Whoop Strain: žádné platné hodnoty")
    else:
        print("  ✗ whoop_strain: sloupec nenalezen")

    # 4. Průměrný sleep_performance_pct
    if "sleep_performance_pct" in daily.columns:
        sp = daily["sleep_performance_pct"].dropna()
        if not sp.empty:
            print(f"  😴 Průměrný Sleep Performance: {sp.mean():.1f} %  "
                  f"(min {sp.min():.1f} %, max {sp.max():.1f} %)")
        else:
            print("  ⚠ Sleep Performance: žádné platné hodnoty")
    else:
        print("  ✗ sleep_performance_pct: sloupec nenalezen")

    # 5. HRV CV
    if "hrv_cv_pct" in daily.columns:
        hcv = daily["hrv_cv_pct"].dropna()
        if not hcv.empty:
            last_cv = hcv.iloc[-1]
            label = "✓ stabilní" if last_cv < 10 else ("⚠ zvýšený" if last_cv < 15 else "✗ vysoký stres")
            print(f"  💓 HRV CV (7 dní): {last_cv:.1f} %  (Ideál < 10 %)  [{label}]")
        else:
            print("  ⚠ HRV CV: žádné platné hodnoty")
    else:
        print("  ✗ hrv_cv_pct: sloupec nenalezen")

    # 6. Polarization 80/20
    if "polarization_low_pct" in daily.columns:
        pl = daily["polarization_low_pct"].dropna()
        ph = daily["polarization_high_pct"].dropna()
        if not pl.empty:
            low_v  = pl.iloc[-1]
            high_v = ph.iloc[-1]
            print(f"  🎯 Polarizace (14 dní): {low_v:.0f} % Low Intensity / {high_v:.0f} % High Intensity")
        else:
            print("  ⚠ Polarizace: žádné platné hodnoty")
        # Z3 Junk Miles
        if "z3_junk_pct" in daily.columns:
            z3j = daily["z3_junk_pct"].dropna()
            if not z3j.empty:
                z3_v = z3j.iloc[-1]
                z3_warn = " ⚠ SNIŽ Z3 (>15 %)" if z3_v > 15 else (" ✓ ok" if z3_v <= 5 else "")
                print(f"       Z3 'Junk Miles' (14 dní): {z3_v:.1f} %{z3_warn}")
        if "polarization_efficiency" in daily.columns:
            pe = daily["polarization_efficiency"].dropna()
            if not pe.empty:
                pe_v = pe.iloc[-1]
                pe_label = "✓ vysoká" if pe_v >= 80 else ("⚠ snížená" if pe_v >= 60 else "🔴 nízká")
                print(f"       Polarizační efektivita: {pe_v:.1f} %  [{pe_label}]")
    else:
        print("  ✗ polarization: sloupce nenalezeny")

    # 7. Best VAM
    if "vam_m_per_h" in master.columns:
        vam_valid = master["vam_m_per_h"].dropna()
        if not vam_valid.empty:
            best_idx = vam_valid.idxmax()
            best_val = vam_valid.max()
            best_row = master.loc[best_idx]
            print(
                f"  ⛰️  Nejlepší VAM: {best_val:.0f} m/h  "
                f"({best_row['date']}  {best_row['sport']}  "
                f"{best_row['duration_minutes']:.0f} min)"
            )
        else:
            print("  ⚠ VAM: žádné aktivity s ascent > 50 m")
    else:
        print("  ✗ vam_m_per_h: sloupec nenalezen v master")

    # 8. Top-3 Strain days
    if "strain" in daily.columns:
        top3 = daily["strain"].dropna().nlargest(3)
        if not top3.empty:
            print("  Top 3 nejnebezpečnější dny (Strain):")
            for dt, val in top3.items():
                d = dt.date() if hasattr(dt, "date") else dt
                print(f"    {d}  →  Strain = {val:.1f}")

    # ── NEW: Risk (ACWR + Ramp Rate) ─────────────────────────────────────
    print()
    print("  ── RISK ──────────────────────────────────────────────────────")
    if "acwr" in daily.columns:
        acwr_last = daily["acwr"].dropna()
        if not acwr_last.empty:
            a = acwr_last.iloc[-1]
            if a < 0.8:
                status = "NÍZKÉ (undertrained)"
            elif a <= 1.3:
                status = "✓ SWEET SPOT"
            elif a <= 1.5:
                status = "⚠ ZVÝŠENÉ"
            else:
                status = "🔴 DANGER ZONE"
            print(f"  ACWR: {a:.2f}  [{status}]")
    if "ctl_ramp_rate" in daily.columns:
        ramp = daily["ctl_ramp_rate"].dropna()
        if not ramp.empty:
            r = ramp.iloc[-1]
            warn = " ⚠ BURN-OUT RISK" if r > CTL_RAMP_WARN else ""
            print(f"  CTL Ramp Rate (7d): {r:+.1f}{warn}")

    # ── NEW: Metabolism ──────────────────────────────────────────────────
    print()
    print("  ── METABOLISM (posledních 7 dní) ─────────────────────────────")
    if "fat_g" in master.columns and "carb_g" in master.columns:
        recent = master.loc[master["date"].astype(str) >= str(
            (pd.Timestamp.now() - pd.Timedelta(days=7)).date()
        )]
        fat_g_sum = pd.to_numeric(recent["fat_g"], errors="coerce").sum()
        carb_g_sum = pd.to_numeric(recent["carb_g"], errors="coerce").sum()
        print(f"  Spáleno tuků:  {fat_g_sum:.0f} g  ({fat_g_sum * KCAL_PER_G_FAT:.0f} kcal)")
        print(f"  Spáleno cukrů: {carb_g_sum:.0f} g  ({carb_g_sum * KCAL_PER_G_CARB:.0f} kcal)")
    else:
        print("  ⚠ Fueling data nenalezena")

    # ── NEW: Physio (Durability + EPOC + DFA + Resp + W') ───────────────
    print()
    print("  ── PHYSIO (v7.0 R-R Engine) ──────────────────────────────────")
    if "durability_pct" in master.columns:
        dur_valid = master["durability_pct"].dropna()
        if not dur_valid.empty:
            avg_dur = dur_valid.mean()
            print(f"  Průměrná Durability: {avg_dur:+.1f} %  "
                  f"(n={len(dur_valid)} aktivit > 2h)")
        else:
            print("  ⚠ Durability: žádné aktivity > 2 h")
    if "epoc_score" in master.columns:
        epoc_valid = master["epoc_score"].dropna()
        if not epoc_valid.empty:
            last_epoc = epoc_valid.iloc[-1]
            print(f"  Poslední EPOC score: {last_epoc:.0f}")
    # DFA-alpha1 (real vs proxy)
    if "dfa_quality" in master.columns:
        dfa_real = (master["dfa_quality"] == "real").sum()
        dfa_proxy = (master["dfa_quality"] == "proxy").sum()
        print(f"  DFA-alpha1: {dfa_real} real (neurokit2) | {dfa_proxy} proxy (fallback)")
    if "aet_hr_dfa" in master.columns:
        aet_valid = master["aet_hr_dfa"].dropna()
        if not aet_valid.empty:
            last_aet = aet_valid.iloc[-1]
            print(f"  Poslední AeT (DFA α1=0.75): {last_aet:.0f} bpm")
    if "ant_hr_dfa" in master.columns:
        ant_valid = master["ant_hr_dfa"].dropna()
        if not ant_valid.empty:
            last_ant = ant_valid.iloc[-1]
            print(f"  Poslední AnT (DFA α1=0.50): {last_ant:.0f} bpm")
    if "resp_rate_rsa" in master.columns:
        resp_valid = master["resp_rate_rsa"].dropna()
        if not resp_valid.empty:
            avg_resp = resp_valid.mean()
            print(f"  Průměrná Resp Rate (RSA): {avg_resp:.1f} breaths/min  "
                  f"(n={len(resp_valid)})")
    if "critical_hr" in master.columns:
        chr_val = master["critical_hr"].dropna()
        if not chr_val.empty:
            print(f"  Critical Heart Rate: {chr_val.iloc[0]:.0f} bpm")
    if "tati_score" in master.columns:
        wp = pd.to_numeric(master["tati_score"], errors="coerce")
        wp_valid = wp[wp > 0]
        if not wp_valid.empty:
            print(f"  TATI max: {wp_valid.max():.0f} bpm·min  "
                  f"(n={len(wp_valid)} aktivit s anaerobní prací)")
    if "time_at_threshold_min" in master.columns:
        tte_max = pd.to_numeric(master["time_at_threshold_min"], errors="coerce").max()
        if pd.notna(tte_max) and tte_max > 0:
            print(f"  TTE max (Z4+Z5): {tte_max:.1f} min")

    # ── NEW: Coach Advice ────────────────────────────────────────────────
    print()
    print("  ── COACH ADVICE ──────────────────────────────────────────────")
    if "acwr" in daily.columns:
        acwr_last = daily["acwr"].dropna()
        if not acwr_last.empty:
            a = acwr_last.iloc[-1]
            if a > 1.5:
                print(f"  ⚠ Pozor, tvé ACWR je v červené zóně ({a:.1f}) "
                      f"– sniž objem, nebo hrozí zranění.")
            elif a > 1.3:
                print(f"  ⚠ ACWR zvýšené ({a:.1f}) – buď opatrný s nárůstem zátěže.")
    if "ctl_ramp_rate" in daily.columns:
        ramp_last = daily["ctl_ramp_rate"].dropna()
        if not ramp_last.empty and ramp_last.iloc[-1] > CTL_RAMP_WARN:
            print(f"  ⚠ CTL ramp příliš strmý ({ramp_last.iloc[-1]:+.1f}/týden) – hrozí burn-out!")

    print("=" * 64)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     athlete_analytics.py – Virtual Lab Suite v7.0          ║")
    print("║     PRODUKČNÍ REŽIM – R-R Physiology Engine                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── load data ─────────────────────────────────────────────────────────
    master = load_master()
    activities = load_activities()
    hrv_df = load_hrv()
    health_df = load_health()
    sleep_df = load_sleep()

    print(f"  Aktivit:  {len(master)}  ({master['date'].min()} → {master['date'].max()})")
    print(f"  HRV dní:  {len(hrv_df)}   |   Health dní: {len(health_df)}   |   Sleep dní: {len(sleep_df)}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  A + B: Cardiac Drift & Max HRR 60s  (full database)
    # ══════════════════════════════════════════════════════════════════════
    print("[A+B] Cardiac Drift & Max HRR 60s")
    master = compute_per_activity_metrics(master)

    # ══════════════════════════════════════════════════════════════════════
    #  J: VAM (Velocità Ascensionale Media)
    # ══════════════════════════════════════════════════════════════════════
    print("[J] VAM (Velocità Ascensionale Media)")
    master = compute_vam(master)

    # ══════════════════════════════════════════════════════════════════════
    #  K-T: Extended per-activity metrics (Fueling, Fluid, EPOC, Climb, …)
    # ══════════════════════════════════════════════════════════════════════
    print("[K-W] Extended per-activity metrics + R-R Physiology")
    master = compute_extended_per_activity(master)

    # ══════════════════════════════════════════════════════════════════════
    #  Daily aggregations
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("[CTL/ATL/TSB] Denní TRIMP + exponenciální průměry")
    daily = build_daily_trimp(master)
    daily = compute_ctl_atl_tsb(daily)

    # ══════════════════════════════════════════════════════════════════════
    #  C: Monotony, Strain & Whoop Strain
    # ══════════════════════════════════════════════════════════════════════
    print("[C] Training Monotony, Strain & Whoop Logarithmic Strain")
    daily = compute_monotony_strain(daily)

    # ══════════════════════════════════════════════════════════════════════
    #  D: Efficiency Index
    # ══════════════════════════════════════════════════════════════════════
    print("[D] Efficiency Index (TRIMP / km)")
    eff_daily = compute_efficiency_index(master, activities)
    if not eff_daily.empty:
        daily = daily.join(eff_daily, how="left")
    else:
        daily.loc[:, "daily_efficiency"] = np.nan
        daily.loc[:, "ef_trend"] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  E: Bio-Readiness Score
    # ══════════════════════════════════════════════════════════════════════
    print("[E] Bio-Readiness Score")
    daily = compute_readiness(daily, hrv_df)

    # ── Max HRR 60s daily average (from master) ──────────────────────────
    hrr_valid = master.loc[master["max_hrr_60s"].notna()].copy() \
        if "max_hrr_60s" in master.columns else pd.DataFrame()
    if not hrr_valid.empty:
        hrr_valid = hrr_valid.assign(date=pd.to_datetime(hrr_valid["date"]))
        hrr_daily = (
            hrr_valid.groupby("date")["max_hrr_60s"]
            .mean()
            .to_frame("max_hrr_60s_avg")
        )
        daily = daily.join(hrr_daily, how="left")
    else:
        daily.loc[:, "max_hrr_60s_avg"] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  F: Pure Recovery Score (HRV + RHR + Spánek) + Sleep Performance
    # ══════════════════════════════════════════════════════════════════════
    print("[F] Pure Recovery Score + Sleep Performance")
    daily = compute_pure_recovery(daily, hrv_df, health_df, sleep_df)

    # ══════════════════════════════════════════════════════════════════════
    #  G: Illness Warning
    # ══════════════════════════════════════════════════════════════════════
    print("[G] Illness Warning (multi-indikátor)")
    daily = compute_illness_warning(daily)

    # ══════════════════════════════════════════════════════════════════════
    #  K: ACWR + CTL Ramp Rate
    # ══════════════════════════════════════════════════════════════════════
    print("[K] ACWR + CTL Ramp Rate")
    daily = compute_acwr(daily)

    # ══════════════════════════════════════════════════════════════════════
    #  L: External vs Internal Load (fatigue index)
    # ══════════════════════════════════════════════════════════════════════
    print("[L] Fatigue Index (External vs Internal Load)")
    daily = compute_fatigue_index(daily)

    # ══════════════════════════════════════════════════════════════════════
    #  H: Coach Advice
    # ══════════════════════════════════════════════════════════════════════
    print("[H] Coach Advice")
    daily = compute_coach_advice(daily)

    # ══════════════════════════════════════════════════════════════════════
    #  I: Polarization Score (80/20)
    # ══════════════════════════════════════════════════════════════════════
    print("[I] Polarization Score (80/20)")
    daily = compute_polarization(daily, master)

    # ══════════════════════════════════════════════════════════════════════
    #  Aggregate per-activity metrics into daily
    # ══════════════════════════════════════════════════════════════════════
    # EPOC / recovery_tax daily sums
    for agg_col in ("epoc_score", "recovery_tax_min", "fat_g", "carb_g",
                     "fat_kcal", "carb_kcal", "fluid_loss_L"):
        if agg_col in master.columns:
            _m = master[["date", agg_col]].copy()
            _m = _m.assign(date=pd.to_datetime(_m["date"]))
            _m[agg_col] = pd.to_numeric(_m[agg_col], errors="coerce")
            _agg = _m.groupby("date")[agg_col].sum().to_frame(f"{agg_col}_daily")
            daily = daily.join(_agg, how="left")

    # ══════════════════════════════════════════════════════════════════════
    #  SAVE athlete_readiness.csv
    # ══════════════════════════════════════════════════════════════════════
    out_cols = [
        "CTL", "ATL", "TSB",
        "monotony", "strain", "whoop_strain",
        "daily_efficiency", "ef_trend",
        "readiness_score",
        "max_hrr_60s_avg",
        # Recovery & bio
        "pure_recovery_score",
        "rhr_day", "hrv_last_night", "hrv_weekly_avg",
        "sleep_score_day", "sleep_duration_min",
        "sleep_need_min", "sleep_performance_pct",
        "hrv_cv_pct",
        "polarization_low_pct", "polarization_high_pct",
        "stress_flag_count", "illness_warning", "stress_flags",
        # Load management (v6.0)
        "acwr", "ctl_ramp_rate", "ctl_ramp_warning",
        "fatigue_index",
        # Metabolism (v6.0)
        "fat_kcal_daily", "carb_kcal_daily",
        "fat_g_daily", "carb_g_daily",
        "fluid_loss_L_daily",
        # EPOC (v6.0)
        "epoc_score_daily", "recovery_tax_min_daily",
        "coach_advice",
    ]
    out = daily[[c for c in out_cols if c in daily.columns]].copy()
    round_map = {
        "CTL": 2, "ATL": 2, "TSB": 2,
        "monotony": 2, "strain": 1, "whoop_strain": 2,
        "daily_efficiency": 4, "ef_trend": 4,
        "readiness_score": 1, "max_hrr_60s_avg": 1,
        "pure_recovery_score": 1,
        "rhr_day": 0, "hrv_last_night": 1, "hrv_weekly_avg": 1,
        "sleep_score_day": 0, "sleep_duration_min": 0,
        "sleep_need_min": 0, "sleep_performance_pct": 1,
        "hrv_cv_pct": 1,
        "polarization_low_pct": 1, "polarization_high_pct": 1,
        "acwr": 2, "ctl_ramp_rate": 2,
        "fatigue_index": 3,
        "fat_kcal_daily": 0, "carb_kcal_daily": 0,
        "fat_g_daily": 0, "carb_g_daily": 0,
        "fluid_loss_L_daily": 2,
        "epoc_score_daily": 1, "recovery_tax_min_daily": 0,
    }
    out = out.round({k: v for k, v in round_map.items() if k in out.columns})
    out.index = out.index.date
    out.index.name = "date"
    out.to_csv(OUTPUT_CSV)
    print(f"\n  → Uloženo: {OUTPUT_CSV}  ({len(out)} řádků)")

    # ══════════════════════════════════════════════════════════════════════
    #  WRITE BACK to master_high_res_summary.csv
    # ══════════════════════════════════════════════════════════════════════
    master_out = master.copy()
    _new_master_cols = (
        "cardiac_drift", "max_hrr_60s", "vam_m_per_h",
        "fat_kcal", "carb_kcal", "fat_g", "carb_g",
        "fluid_loss_L", "heat_flag",
        "epoc_score", "recovery_tax_min", "time_at_threshold_min",
        "tte_z4z5_min",
        "durability_pct", "aet_hr_proxy",
        "avg_gradient_pct", "climb_category",
        # v7.0 – R-R Physiology Engine
        "aet_hr_dfa", "ant_hr_dfa", "dfa_quality",
        "resp_rate_rsa",
        "critical_hr", "tati_score",
    )
    for col in _new_master_cols:
        if col not in master_out.columns:
            master_out = master_out.assign(**{col: np.nan})
    # drop legacy columns
    master_out = master_out.drop(columns=["hrr_1min", "dynamic_hrr", "w_prime_bpm_min"], errors="ignore")
    master_out.to_csv(MASTER_CSV, index=False)
    print(f"  → Zpětný zápis: {MASTER_CSV}  ({len(_new_master_cols)} metrik)")

    # ══════════════════════════════════════════════════════════════════════
    #  CONSOLE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    latest = daily.iloc[-1]
    ctl, atl, tsb = latest["CTL"], latest["ATL"], latest["TSB"]
    rec = recommend(tsb)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"  Fitness (CTL): {ctl:.1f}  │  Únava (ATL): {atl:.1f}  │  Forma (TSB): {tsb:.1f}")
    print(f"  Doporučení: {rec}")

    if "readiness_score" in daily.columns:
        last_rs = daily["readiness_score"].dropna()
        if not last_rs.empty:
            rs = last_rs.iloc[-1]
            rs_date = last_rs.index[-1]
            rs_d = rs_date.date() if hasattr(rs_date, "date") else rs_date
            print(f"  Readiness Score ({rs_d}): {rs:.1f} / 100")

    if "pure_recovery_score" in daily.columns:
        last_pr = daily["pure_recovery_score"].dropna()
        if not last_pr.empty:
            pr = last_pr.iloc[-1]
            pr_date = last_pr.index[-1]
            pr_d = pr_date.date() if hasattr(pr_date, "date") else pr_date
            print(f"  Pure Recovery ({pr_d}): {pr:.1f} / 100")

    if "whoop_strain" in daily.columns:
        last_ws = daily["whoop_strain"].dropna()
        if not last_ws.empty:
            ws = last_ws.iloc[-1]
            print(f"  Whoop Strain (dnes): {ws:.2f} / 21.00")

    if "illness_warning" in daily.columns:
        illness_days = daily["illness_warning"].sum()
        print(f"  Illness Warning dní: {int(illness_days)}")

    if "coach_advice" in daily.columns:
        last_advice = daily["coach_advice"].dropna().iloc[-1] if not daily["coach_advice"].dropna().empty else ""
        if last_advice:
            print(f"  Coach: {last_advice[:80]}")

    if "monotony" in daily.columns:
        last_m = daily["monotony"].dropna()
        last_s = daily["strain"].dropna()
        if not last_m.empty:
            print(f"  Monotony: {last_m.iloc[-1]:.2f}  │  Strain: {last_s.iloc[-1]:.1f}")

    print("╚══════════════════════════════════════════════════════════════╝")

    # ══════════════════════════════════════════════════════════════════════
    #  SELF-CHECK
    # ══════════════════════════════════════════════════════════════════════
    verify_analytics(daily, master)


if __name__ == "__main__":
    main()
