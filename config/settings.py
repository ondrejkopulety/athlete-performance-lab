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
STRAVA_FIT_DIR  = FIT_DIR / "strava_originals"
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

# ============================================================
# BOTANICAL STOP ANALYSIS  –  Single Source of Truth
# ============================================================

# Minimum Confidence Score for a stop to appear in ANY pipeline output
# (green_stops_report.csv, botanical_hotspots_ranked.csv, performance CSV, map).
# Change here to adjust the filter universally.
MIN_CONFIDENCE_THRESHOLD: float  = 15.0

# DBSCAN clustering radius – stops within this distance merge into one hotspot.
# Used by detect_botanical_hotspots.py; defined here so both the mapper and
# any future scripts share an identical value.
CLUSTER_RADIUS_M: int            = 300

# A GPS speed below this threshold means the rider is "stopped"
STOP_SPEED_THRESHOLD_KMH: float  = 2.0

# A stop must last at least this many seconds to be analysed
MIN_STOP_DURATION_S: int         = 600         # 10 minutes

# Consecutive "stopped" rows can be separated by at most this many seconds
# (bridges Auto-Pause / Smart-Recording gaps) and still count as one stop
MAX_AUTO_PAUSE_GAP_SEC: int      = 900         # 15 minutes

# Length of pre/post-stop analysis window for physiological markers
ANALYSIS_WINDOW_S: int           = 15 * 60     # 15 minutes

# HR settle window used at start/end of each stop for Marker A
HR_SETTLE_WINDOW_S: int          = 3 * 60      # 3 minutes

# Set False to skip exclusion-zone filtering for quick / offline runs
ENABLE_EXCLUSION_ZONES: bool     = True

# Locations to completely exclude from all outputs (home, work, known false-positives).
# Format:  "Label": (latitude, longitude, radius_metres)
# Edit coordinates to your real locations; add/remove entries freely.
EXCLUDED_LOCATIONS: dict[str, tuple[float, float, float]] = {
    "Ignored_Spot_1": (49.2099417, 16.1466983, 200),
    "Ignored_Spot_2": (49.2061342, 16.1556133, 200),
    "Ignored_Spot_3": (49.2127633, 16.1470508, 200),
    "Ignored_Spot_4": (49.1540772, 16.0799922, 200),
}

# Sports considered "cardio" (cycling + running variants).
# Used to filter activities before stop-detection in all three pipeline scripts.
CARDIO_SPORTS: frozenset[str] = frozenset({
    "cycling", "gravel_cycling", "mountain_biking",
    "road_cycling", "indoor_cycling", "virtual_cycling",
    "e_bike", "bmx", "cyclocross", "track_cycling",
    "running", "trail_running", "treadmill_running",
    "track_running", "ultra_running",
})
