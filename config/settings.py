"""
Garmin Training Analytics – Centralized Configuration
======================================================
All athlete parameters, file paths, and thresholds in one place.
Edit this file to match YOUR physiology – never hardcode values
in individual scripts.
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS (relative to project root)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
FIT_DIR         = DATA_DIR / "fit"
SUMMARIES_DIR   = DATA_DIR / "summaries"
PROCESSED_DIR   = DATA_DIR / "processed"
REPORTS_DIR     = PROJECT_ROOT / "reports"
LOGS_DIR        = PROJECT_ROOT / "logs"

# ============================================================
# ATHLETE PROFILE
# ============================================================
MAX_HR          = 199          # Maximum heart rate (bpm)
RESTING_HR      = 41           # Resting heart rate (bpm)

# Karvonen HR-Reserve zone boundaries (fraction of HRR)
# Z1: 50-60%, Z2: 60-72%, Z3: 72-82%, Z4: 82-90%, Z5/Z5+: 90-100%
ZONE_PCTS       = [0.50, 0.60, 0.72, 0.82, 0.90, 1.00]
ZONE_LABELS     = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z5+"]
ZONE_2_CAP      = 155          # Talk-Test ceiling (bpm) ≈ 72 % HRR

# ============================================================
# TRAINING LOAD MODEL (Banister / PMC)
# ============================================================
CTL_DAYS        = 42           # Chronic Training Load window
ATL_DAYS        = 7            # Acute Training Load window
CTL_RAMP_WARN   = 8.0          # CTL ramp rate → burnout warning

# ACWR (Acute : Chronic Workload Ratio)
ACWR_ACUTE_DAYS   = 7
ACWR_CHRONIC_DAYS = 28

# TRIMP constants (male Banister model)
TRIMP_K1        = 0.64
TRIMP_K2        = 1.92

# ============================================================
# DEDUPLICATION (master_rebuild)
# ============================================================
DEDUP_WINDOW_MIN    = 30       # minutes tolerance for duplicate detection
HR_DENSITY_THRESHOLD = 0.90    # 90 % HR coverage = reliable file
ANT_DEVICE_TYPE_HR  = 120      # ANT+ chest strap device type
INTEGRITY_DIFF_PCT  = 0.25     # 25 % max allowable summary diff

# ============================================================
# CARDIAC DRIFT & RECOVERY
# ============================================================
WARMUP_SECONDS         = 600   # 10 min warm-up skip for drift
MIN_DURATION_DRIFT_MIN = 20    # min activity duration (min) for drift
MAX_REALISTIC_HRR      = 90    # bpm drop > 90 = sensor glitch
MIN_STARTING_HR        = 110   # recovery only from sufficient load
MAX_3S_JUMP            = 30    # bpm jump > 30 in 3 s = artefact
CARDIAC_DRIFT_MAX_ALT  = 30.0  # metres – max altitude range for drift

# ============================================================
# READINESS / ILLNESS WARNING
# ============================================================
HRV_DROP_THRESHOLD     = 0.10  # 10 % below weekly avg → warning
HIGH_RHR_THRESHOLD     = 46    # bpm above → flag
LOW_SLEEP_SCORE        = 60    # sleep score below → flag
SHORT_SLEEP_MINUTES    = 360   # < 6 h total → flag
ILLNESS_FLAG_COUNT     = 3     # simultaneous flags → illness alert

# ============================================================
# EFFICIENCY & ANALYTICS
# ============================================================
EF_WINDOW              = 30    # Efficiency factor trend window (days)
MONOTONY_WINDOW        = 7     # Training monotony window (days)

# Fueling model – fat / carb split per zone
FAT_PCT_BY_ZONE        = {"Z1": 0.80, "Z2": 0.60, "Z3": 0.40, "Z4": 0.10, "Z5": 0.00}
KCAL_PER_MIN_BY_ZONE   = {"Z1": 6, "Z2": 8, "Z3": 10, "Z4": 12, "Z5": 14}

# Durability
DURABILITY_MIN_DURATION_MIN = 120   # minimum 2 h activity for durability

# DFA alpha-1
DFA_WINDOW_BEATS       = 200
DFA_AET_THRESHOLD      = 0.75  # α1 at aerobic threshold
DFA_ANT_THRESHOLD      = 0.50  # α1 at anaerobic threshold

# ============================================================
# SPEED THRESHOLDS
# ============================================================
DEFAULT_SPEED_THRESHOLD_MS  = 0.3     # m/s – general
CYCLING_SPEED_THRESHOLD_MS  = 0.833   # m/s – 3.0 km/h

# ============================================================
# GARMIN SYNC
# ============================================================
INITIAL_BACKFILL_DAYS  = 180   # Days of history on first sync

# ============================================================
# CSV FILE NAMES (inside SUMMARIES_DIR)
# ============================================================
CSV_ACTIVITIES             = "activities.csv"
CSV_HRV                    = "hrv.csv"
CSV_DAILY_HEALTH           = "daily_health.csv"
CSV_SLEEP                  = "sleep.csv"
CSV_TRAINING_READINESS     = "training_readiness.csv"
CSV_VO2_MAX                = "vo2_max.csv"
CSV_HEART_RATE_SUMMARY     = "heart_rate_summary.csv"
CSV_HEART_RATE_DETAILS     = "heart_rate_details.csv"
CSV_MOVEMENT               = "movement.csv"
CSV_INTENSITY              = "intensity.csv"
CSV_TRAINING_STATUS        = "training_status.csv"
CSV_LOAD_FOCUS             = "load_focus.csv"
CSV_LACTATE_THRESHOLD      = "lactate_threshold.csv"

CSV_HIGH_RES_TRAINING      = "high_res_training_data.csv"
CSV_HIGH_RES_SUMMARY       = "high_res_summary.csv"
CSV_MASTER_TRAINING        = "master_high_res_training_data.csv"
CSV_MASTER_SUMMARY         = "master_high_res_summary.csv"
CSV_ATHLETE_READINESS      = "athlete_readiness.csv"
CSV_METADATA_CACHE         = "metadata_cache.json"
