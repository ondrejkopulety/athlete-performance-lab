"""
Garmin Training Analytics – Centralized Configuration
======================================================
All athlete parameters, file paths, and thresholds in one place.
Edit this file to match YOUR physiology – never hardcode values
in individual scripts.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# PROJECT PATHS (relative to project root)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

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

# Hardcoded heart rate zones (bpm) – measured from lactate tests / personal experience.
# Each zone is (lower_bound_inclusive, upper_bound_inclusive).
# Edit these values whenever your fitness changes; no formula involved.
ZONES: dict[str, tuple[int, int]] = {
    "Z1": (100, 136),
    "Z2": (137, 155),
    "Z3": (156, 171),
    "Z4": (172, 183),
    "Z5": (184, 199),
}
ZONE_LABELS     = list(ZONES.keys())   # ["Z1", "Z2", "Z3", "Z4", "Z5"]
ZONE_2_CAP      = 155                  # Talk-Test ceiling (bpm) – top of Z2

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

# Klidový tep se posuzuje RELATIVNĚ k vlastnímu baseline, ne proti pevnému
# číslu. Absolutní práh nefunguje: RHR se v průběhu sezóny posouvá s formou
# i s věkem, takže hodnota nastavená jednou brzy leží uprostřed běžného
# rozptylu a vlajka pak hoří zhruba obden, čímž ztrácí výpovědní hodnotu.
# Elevace o 5 bpm nad baseline je běžně užívaný marker únavy či nemoci.
RHR_BASELINE_DAYS      = 14    # okno pro klouzavý baseline klidového tepu
RHR_ELEVATION_BPM      = 5     # o kolik bpm nad baseline → varovná vlajka

# Dlouhé okno slouží k jinému účelu: škálování tréninkové zátěže. TRIMP
# potřebuje vědět, jaká je aktuální ÚROVEŇ klidového tepu, ne jestli je
# dnešek mimo. Krátké okno by do zátěže vneslo denní šum ze spánku a
# alkoholu. Medián místo průměru – odolnější vůči jednotlivé špatné noci.
RHR_BASELINE_LONG_DAYS = 90
RHR_BASELINE_LONG_MIN  = 30    # minimum měření v okně, jinak fallback na RESTING_HR
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
INITIAL_BACKFILL_DAYS  = 3     # Days of history on first sync (low to avoid rate-limit ban)

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

# Místa vyloučená ze všech výstupů (domov, práce, známé falešné poplachy).
#
# Souřadnice žijí v .env, ne tady: jsou to reálné adresy s přesností na
# metry a repozitář může být veřejný. Kód v repu, data mimo něj.
#
# Formát v .env (středníkem oddělené položky, popisek je volitelný):
#   EXCLUDED_LOCATIONS=domov:50.0755,14.4378,200;prace:50.0880,14.4210,200
def _parse_excluded_locations(raw: str) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for i, entry in enumerate(p.strip() for p in raw.split(";")):
        if not entry:
            continue
        label, _, coords = entry.rpartition(":")
        parts = [c.strip() for c in coords.split(",")]
        if len(parts) != 3:
            continue  # poškozený zápis raději ignoruj, než aby spadl import
        try:
            lat, lon, radius = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            continue
        out[label.strip() or f"Ignored_Spot_{i + 1}"] = (lat, lon, radius)
    return out


EXCLUDED_LOCATIONS: dict[str, tuple[float, float, float]] = _parse_excluded_locations(
    os.getenv("EXCLUDED_LOCATIONS", "")
)

# Sports considered "cardio" (cycling + running variants).
# Used to filter activities before stop-detection in all three pipeline scripts.
CARDIO_SPORTS: frozenset[str] = frozenset({
    "cycling", "gravel_cycling", "mountain_biking",
    "road_cycling", "indoor_cycling", "virtual_cycling",
    "e_bike", "bmx", "cyclocross", "track_cycling",
    "running", "trail_running", "treadmill_running",
    "track_running", "ultra_running",
})

# ============================================================
# DATABASE
# ============================================================
DATABASE_URL: str = os.getenv("DATABASE_URL") or (
    "postgresql+psycopg://"
    f"{os.getenv('POSTGRES_USER', 'garmin')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'garmin')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'garmin')}"
)

# ============================================================
# ANALYTICS VERSIONING & INCREMENTAL WINDOW
# ============================================================
# Bump when a per-activity formula changes → vynutí přepočet activity_metrics
# u všech aktivit (řádky s nižší verzí se považují za zastaralé).
#   2 = TRIMP z klidového tepu platného k datu, LTHR z terénních dat,
#       oprava parametru DFA-alpha1
ACTIVITY_METRICS_VERSION: int = 2

# Bump when a daily formula changes → vynutí full rebuild daily_metrics.
#   2 = 90denní baseline klidového tepu, lthr_estimate
DAILY_METRICS_VERSION: int = 2

# ============================================================
# PRAHOVÝ TEP Z TERÉNNÍCH DAT (LTHR)
# ============================================================
# Praktický odhad prahu: 0.95 × nejlepší 20minutový průměr tepu.
# Na reálných datech dává 172 bpm, což na bpm sedí s laktátovým testem.
# Na rozdíl od DFA-alpha1 funguje nad všemi aktivitami, ne jen nad těmi
# s hrudním pásem. Slouží jen jako REFERENCE – zóny v ZONES mají přednost,
# protože měřená hodnota je víc než odhad.
LTHR_TEST_MINUTES      = 20    # délka úsilí, ze kterého se práh odhaduje
LTHR_FACTOR            = 0.95  # převod 20min výkonu na hodinový práh
# 180 dní, ne 90: odhad je DOLNÍ mez – odráží jen nejtvrdší úsilí, které
# v okně opravdu proběhlo. Za 90 dní se snadno stane, že žádné maximální
# úsilí nebylo, a práh pak vypadá, že spadl (naměřeno 164 vs 172 bpm).
# Při 180 dnech vychází 172 bpm, tedy shoda s laktátovým testem.
LTHR_WINDOW_DAYS       = 180   # okno, ve kterém se hledá nejlepší úsilí
LTHR_BEST_WINDOWS_MIN  = [20, 30, 60]  # která okna počítat per-activity

# Kolik dní historie načíst před prvním "dirty" dnem, aby rolling okna
# (monotony 7d, ACWR 7/28d, polarizace 14d, HRV z-score 30d, strain kvantil 30d)
# měla plný kontext. Musí být >= nejdelší rolling okno; 45 dává rezervu.
# CTL/ATL (EMA) se neseeduje warm-upem, ale uloženou hodnotou z předchozího dne.
LOOKBACK_DAYS: int = 45

# ============================================================
# API
# ============================================================
API_CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("API_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
SYNC_CRON_HOUR: int | None = (
    int(os.environ["SYNC_CRON_HOUR"]) if os.getenv("SYNC_CRON_HOUR", "").strip() else None
)

# ============================================================
# METRIC METADATA  –  glosář pro AI trenéra (/coach/context)
# ============================================================
# Jediná definice významu metrik. LLM bez tohohle neví, že u hrv_cv_pct je
# nižší lepší, nebo že ACWR má sweet spot uprostřed, ne na kraji.
#
#   unit      – jednotka (pro formulaci odpovědi)
#   direction – higher_is_better | lower_is_better | sweet_spot | higher_is_fresher
#   note      – jak se metrika počítá / co znamená
#   bands     – interpretační pásma (volitelné)
METRIC_META: dict[str, dict] = {
    # ── Tréninková zátěž (PMC) ─────────────────────────────────────────────
    "ctl": {
        "unit": "TRIMP/den", "direction": "higher_is_better",
        "note": f"Chronic Training Load – exponenciální průměr TRIMP za {CTL_DAYS} dní "
                "(alpha=1/N). Proxy pro fitness.",
    },
    "atl": {
        "unit": "TRIMP/den", "direction": "neutral",
        "note": f"Acute Training Load – exponenciální průměr TRIMP za {ATL_DAYS} dní. "
                "Proxy pro aktuální únavu.",
    },
    "tsb": {
        "unit": "TRIMP/den", "direction": "higher_is_fresher",
        "note": "Training Stress Balance = CTL − ATL z PŘEDCHOZÍHO dne. "
                "Posun o den záměrně: reprezentuje ranní formu před dnešním tréninkem.",
        "bands": {
            "< -30": "vysoké riziko zranění, nutný odpočinek",
            "-30 až -10": "optimální tréninková zátěž",
            "-10 až 5": "udržovací režim",
            "> 10": "čerstvost, připravenost na výkon",
        },
    },
    "trimp": {
        "unit": "TRIMP", "direction": "neutral",
        "note": "Banisterův TRIMP za den (součet přes aktivity). "
                "Hiking/walking má koeficient 0.6, aby dlouhé Z1 túry nenafoukly PMC.",
    },
    # ── Riziko ─────────────────────────────────────────────────────────────
    "acwr": {
        "unit": "poměr", "direction": "sweet_spot", "sweet_spot": [0.8, 1.3],
        "note": f"Acute:Chronic Workload Ratio – {ACWR_ACUTE_DAYS}d průměr / "
                f"{ACWR_CHRONIC_DAYS}d průměr, počítáno z EPOC-vážené TRIMP. Neclipováno.",
        "bands": {
            "< 0.8": "undertrained, ztráta formy",
            "0.8 až 1.3": "sweet spot",
            "1.3 až 1.5": "zvýšené riziko",
            "> 1.5": "danger zone, hrozí zranění",
        },
    },
    "ctl_ramp_rate": {
        "unit": "CTL/týden", "direction": "sweet_spot", "sweet_spot": [0, CTL_RAMP_WARN],
        "note": f"Týdenní přírůstek CTL (CTL − CTL před 7 dny). "
                f"Nad {CTL_RAMP_WARN} hrozí burn-out.",
    },
    "illness_warning": {
        "unit": "bool", "direction": "lower_is_better",
        "note": f"True když je současně aktivních >= {ILLNESS_FLAG_COUNT} varovných vlajek "
                "(pokles HRV, vysoký RHR, špatný spánek, vysoký strain při nízké regeneraci).",
    },
    # ── Kvalita tréninku ───────────────────────────────────────────────────
    "monotony": {
        "unit": "poměr", "direction": "lower_is_better",
        "note": f"Foster monotony – mean/std denního TRIMP v {MONOTONY_WINDOW}d okně, "
                "stropováno na 4.0. Vysoká monotonie = málo variability = riziko.",
        "bands": {"< 1.5": "dobrá variabilita", "1.5 až 2.0": "zvýšená", "> 2.0": "vysoká, přidej lehký/volný den"},
    },
    "strain": {
        "unit": "TRIMP", "direction": "neutral",
        "note": "Foster strain = monotony × součet TRIMP za 7 dní.",
    },
    "whoop_strain": {
        "unit": "0–21", "direction": "neutral",
        "note": "Logaritmický strain 21 × (1 − exp(−0.005 × denní TRIMP)).",
    },
    "polarization_low_pct": {
        "unit": "%", "direction": "sweet_spot", "sweet_spot": [75, 85],
        "note": "Podíl času v Z1+Z2 za 14 dní ze VŠECH zón. Seilerův cíl je ~80 %.",
    },
    "polarization_high_pct": {
        "unit": "%", "direction": "sweet_spot", "sweet_spot": [15, 25],
        "note": "Podíl času v Z4+Z5 za 14 dní ze všech zón. Seilerův cíl je ~20 %.",
    },
    "z3_junk_pct": {
        "unit": "%", "direction": "lower_is_better",
        "note": "Podíl času v Z3 ('junk miles') za 14 dní – šedá zóna, "
                "ani regenerace ani vědomá intenzita.",
        "bands": {"<= 5": "ok", "5 až 15": "zvýšené", "> 15": "sniž Z3"},
    },
    "fatigue_index": {
        "unit": "poměr", "direction": "lower_is_better",
        "note": "Dnešní efektivita (TRIMP/km) / 7denní průměr. > 1.0 = organismus "
                "reaguje hůř na stejnou práci = skrytá únava.",
    },
    # ── Regenerace a biometrie ─────────────────────────────────────────────
    "readiness_score": {
        "unit": "0–100", "direction": "higher_is_better",
        "note": "Bio-Readiness. Moderní mód (jsou-li HRV i spánek): "
                "0.30×HRV z-score + 0.30×sleep_score + 0.40×normalizované TSB. "
                "Legacy mód (bez biometrie): pouze funkce TSB.",
    },
    "pure_recovery_score": {
        "unit": "0–100", "direction": "higher_is_better",
        "note": "0.40×HRV (vs 7denní baseline) + 0.30×RHR (vs 14denní baseline) "
                "+ 0.30×sleep_score. Nezávislé na tréninkové zátěži.",
        "bands": {">= 80": "výborná", "60 až 80": "dobrá", "40 až 60": "snížená", "< 40": "nízká, sniž intenzitu"},
    },
    "hrv_last_night": {
        "unit": "ms", "direction": "higher_is_better",
        "note": "Noční průměr RMSSD z Garminu. Absolutní hodnota je individuální – "
                "vždy porovnávej s hrv_weekly_avg, ne s populačními normami.",
    },
    "hrv_cv_pct": {
        "unit": "%", "direction": "lower_is_better",
        "note": "7denní koeficient variace HRV (std/mean z hrubého RMSSD). "
                "Vysoká variabilita HRV = nestabilní autonomní systém.",
        "bands": {"< 10": "stabilní", "10 až 15": "zvýšený", "> 15": "vysoký stres"},
    },
    "rhr_day": {
        "unit": "bpm", "direction": "lower_is_better",
        "note": "Klidový tep z Garminu. Absolutní hodnota je individuální – "
                "vždy porovnávej s rhr_baseline_14d, ne s populačními normami.",
    },
    "rhr_baseline_14d": {
        "unit": "bpm", "direction": "neutral",
        "note": f"Klouzavý {RHR_BASELINE_DAYS}denní průměr klidového tepu. "
                "Referenční hodnota, vůči které se posuzuje elevace.",
    },
    "rhr_baseline_90d": {
        "unit": "bpm", "direction": "lower_is_better",
        "note": f"Klouzavý {RHR_BASELINE_LONG_DAYS}denní MEDIÁN klidového tepu. "
                "Reprezentuje aktuální úroveň, ne denní odchylku, a vstupuje "
                "do výpočtu TRIMP jako spodní hranice tepové rezervy.",
    },
    "lthr_estimate": {
        "unit": "bpm", "direction": "higher_is_better",
        "note": f"Odhad prahového tepu z terénních dat: {LTHR_FACTOR} × nejlepší "
                f"{LTHR_TEST_MINUTES}minutový průměr tepu za posledních "
                f"{LTHR_WINDOW_DAYS} dní. Je to REFERENCE, ne zdroj zón – "
                "nastavené zóny pocházejí z laktátového testu a mají přednost. "
                "POZOR: je to dolní mez. Odráží jen nejtvrdší úsilí, které v okně "
                "opravdu proběhlo, takže po období bez intenzity klesne, aniž by "
                "se práh skutečně zhoršil. Pokles interpretuj až spolu s tím, "
                "jestli v daném období vůbec nějaké maximální úsilí bylo.",
    },
    "rhr_elevation_bpm": {
        "unit": "bpm", "direction": "lower_is_better",
        "note": f"O kolik je dnešní klidový tep nad vlastním "
                f"{RHR_BASELINE_DAYS}denním baseline. Baseline se počítá "
                "z předchozích dní, aby si zvýšená hodnota nezvedala vlastní referenci.",
        "bands": {
            f"< {RHR_ELEVATION_BPM}": "běžný rozptyl",
            f">= {RHR_ELEVATION_BPM}": "elevace – únava, nemoc nebo nedostatek spánku",
        },
    },
    "avg_stress_day": {
        "unit": "0–100", "direction": "lower_is_better",
        "note": "Celodenní průměrný Garmin stres skóre.",
    },
    "sleep_score_day": {
        "unit": "0–100", "direction": "higher_is_better",
        "note": f"Garmin sleep score. Pod {LOW_SLEEP_SCORE} se aktivuje varovná vlajka.",
    },
    "sleep_duration_min": {
        "unit": "min", "direction": "higher_is_better",
        "note": f"Celková délka spánku. Pod {SHORT_SLEEP_MINUTES} min (6 h) varovná vlajka.",
    },
    "sleep_need_min": {
        "unit": "min", "direction": "neutral",
        "note": "Odhad potřeby spánku = 450 min (7.5 h) + 0.5 × TRIMP předchozího dne, "
                "strop 780 min (13 h).",
    },
    "sleep_performance_pct": {
        "unit": "%", "direction": "higher_is_better",
        "note": "sleep_duration_min / sleep_need_min × 100 (Whoop-style).",
    },
    # ── Per-activity fyziologie ────────────────────────────────────────────
    "cardiac_drift": {
        "unit": "%", "direction": "lower_is_better",
        "note": "Aerobní decoupling Pa:HR. Po 10min rozjezdu se aktivita dělí na "
                "poloviny podle času; EF = power/HR (kolo) nebo speed/HR (běh). "
                "Drift = (EF1 − EF2) / EF1 × 100.",
        "bands": {"< 5": "dobrá aerobní odolnost", "> 5": "decoupling, únava nebo horko"},
    },
    "max_hrr_60s": {
        "unit": "bpm", "direction": "higher_is_better",
        "note": "Maximální pokles tepu za 60 s (resample na 1 s, filtry proti "
                "senzorovým glitchům). Vyšší = lepší parasympatická reaktivace.",
    },
    "durability_pct": {
        "unit": "%", "direction": "higher_is_better",
        "note": "Změna efektivity mezi 1. a 2. polovinou aktivity delší než 2 h. "
                "Záporné = pokles výkonu = únava.",
    },
    "vam_m_per_h": {
        "unit": "m/h", "direction": "higher_is_better",
        "note": "Velocità Ascensionale Media = převýšení / čas do kopce. "
                "Jen pro aktivity s průměrným gradientem > 4 %.",
    },
    "aet_hr_dfa": {
        "unit": "bpm", "direction": "higher_is_better",
        "note": f"Aerobní práh z DFA-alpha1 = {DFA_AET_THRESHOLD} (neurokit2 nad R-R "
                "intervaly). Fallback proxy z linearity HR vs rychlost.",
    },
    "ant_hr_dfa": {
        "unit": "bpm", "direction": "higher_is_better",
        "note": f"Anaerobní práh z DFA-alpha1 = {DFA_ANT_THRESHOLD}.",
    },
    "resp_rate_rsa": {
        "unit": "dechů/min", "direction": "neutral",
        "note": "Dechová frekvence z respirační sinusové arytmie (Welch PSD "
                "nad R-R intervaly, pásmo 0.15–0.50 Hz).",
    },
    "epoc_score": {
        "unit": "body", "direction": "neutral",
        "note": "Proxy kyslíkového dluhu = minuty v Z4 × 2 + minuty v Z5 × 5.",
    },
    "recovery_tax_hours": {
        "unit": "h", "direction": "lower_is_better",
        "note": "Odhad hodin snížené kapacity = min(96, 0.08 × TRIMP^1.2).",
    },
    "tati_score": {
        "unit": "bpm·min", "direction": "neutral",
        "note": "Time Above Threshold Impulse – akumulovaná práce nad Critical HR "
                "(Monod-Scherrer adaptovaný na tep).",
    },
}
