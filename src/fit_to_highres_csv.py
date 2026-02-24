"""
fit_to_highres_csv.py  –  Garmin AI Trainer · High-Resolution FIT Processor
=============================================================================
**Čistá extrakční vrstva** – zpracuje .fit soubory a emituje pouze data,
která fyzicky leží uvnitř FIT souboru (record, session, hrv zprávy).

Výstup:
  • data/summaries/high_res_training_data.csv   (každý záznam ≈ 1 s aktivity)
  • data/summaries/high_res_summary.csv          (každý řádek = 1 aktivita)

Žádné načítání externích CSV (daily_health, hrv, activities, sleep).
Veškerá analytická logika (stress, coach_advice, recovery) → athlete_analytics.py.

Parametry atleta
----------------
MAX_HR          : 199 bpm
BASELINE_RHR    : 41  bpm
ZONE_2_CAP      : 155 bpm  (Talk-Test práh ≈ 72 % HRR při RHR 41)
"""

from __future__ import annotations

import csv
import glob
import logging
import math
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Optional

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from fitparse import FitFile

from config.settings import (
    MAX_HR, RESTING_HR as BASELINE_RHR, ZONE_2_CAP,
    ZONE_PCTS, ZONE_LABELS,
    DEFAULT_SPEED_THRESHOLD_MS, CYCLING_SPEED_THRESHOLD_MS,
    TRIMP_K1, TRIMP_K2,
)

# Sporty považované za cyklistiku (pro vyšší speed threshold)
CYCLING_SPORTS: frozenset[str] = frozenset({
    "cycling", "gravel_cycling", "mountain_biking",
    "road_cycling", "indoor_cycling", "virtual_cycling",
    "e_bike", "bmx", "cyclocross", "track_cycling",
})

# Cesty
FIT_FOLDER       = "data/fit"
SUMMARY_FOLDER   = "data/summaries"
HIGHRES_CSV      = os.path.join(SUMMARY_FOLDER, "high_res_training_data.csv")
SUMMARY_CSV      = os.path.join(SUMMARY_FOLDER, "high_res_summary.csv")

HIGHRES_COLS = [
    "activity_id", "timestamp", "date", "heart_rate", "speed",
    "distance", "altitude", "cadence", "power", "temperature",
    "vertical_oscillation", "stance_time",
    "respiratory_rate", "hrv",
    "is_active", "hr_zone", "trimp_increment",
]

SUMMARY_COLS = [
    "activity_id", "date", "activity_name", "sport",
    "duration_minutes", "total_trimp", "avg_hr", "max_hr",
    "time_in_z1", "time_in_z2", "time_in_z3", "time_in_z4", "time_in_z5",
    "zone2_cap_used", "records_count",
    # Fyzické metriky
    "distance_km", "ascent_m", "descent_m",
    "avg_speed_kmh", "max_speed_kmh", "calories",
    # Senzory
    "avg_cadence", "max_cadence", "avg_temp", "max_temp",
    # Výkonnostní (Garmin)
    "training_effect_aerobic", "training_effect_anaerobic", "vo2_max",
    # Stoupání (pro VAM)
    "uphill_minutes",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/fit_to_csv.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("garmin_trainer")





# ─────────────────────────────────────────────────────────────────────────────
# KARVONEN ZÓNY
# ─────────────────────────────────────────────────────────────────────────────

def compute_zones(rhr: int, max_hr: int = MAX_HR) -> list[tuple[int, int, str]]:
    """Vrátí [(lo_bpm, hi_bpm, label), ...] pro Z1–Z5."""
    hrr = max_hr - rhr
    zones = []
    for i in range(len(ZONE_LABELS) - 1):
        lo = round(ZONE_PCTS[i] * hrr + rhr)
        hi = round(ZONE_PCTS[i + 1] * hrr + rhr)
        zones.append((lo, hi, ZONE_LABELS[i]))
    return zones


def classify_zone(hr: Optional[int], zones: list[tuple[int, int, str]]) -> str:
    if hr is None:
        return ""
    for lo, hi, label in zones:
        if lo <= hr <= hi:
            return label
    if hr > zones[-1][1]:
        return "Z5+"
    return "Z1" if hr < zones[0][0] else ""


# ─────────────────────────────────────────────────────────────────────────────
# TRIMP (Banister)
# ─────────────────────────────────────────────────────────────────────────────

def trimp_increment(hr: Optional[int], duration_s: float, rhr: int, max_hr: int = MAX_HR) -> float:
    """
    TRIMP = t_min × HR_ratio × k1 × e^(k2 × HR_ratio)
    HR_ratio = (HR − RHR) / (MaxHR − RHR)

    Konstanty k1=0.64 a k2=1.92 jsou specifické pro MUŽE (Banister et al., 1991).
    Pro ženy použijte k1=0.86, k2=1.67.
    Ponecháno jako in-line hodnoty; pro snadnou změnu v budoucnu
    je lze extrahovat do konfigurační konstanty.
    """
    if hr is None or hr <= rhr or duration_s <= 0:
        return 0.0
    hrr = max_hr - rhr
    if hrr <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, (hr - rhr) / hrr))
    return (duration_s / 60.0) * ratio * TRIMP_K1 * math.exp(TRIMP_K2 * ratio)


# ─────────────────────────────────────────────────────────────────────────────
# RMSSD Z HRV ZPRÁV FIT SOUBORU
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmssd(rr_intervals_s: list[float]) -> Optional[float]:
    if len(rr_intervals_s) < 2:
        return None
    diffs = [
        ((rr_intervals_s[i + 1] - rr_intervals_s[i]) * 1000) ** 2
        for i in range(len(rr_intervals_s) - 1)
    ]
    return math.sqrt(mean(diffs))


def extract_rr_from_fit(fitfile: FitFile) -> list[float]:
    rr: list[float] = []
    try:
        for msg in fitfile.get_messages("hrv"):
            intervals = msg.get_values().get("time")
            if intervals is None:
                continue
            if isinstance(intervals, (list, tuple)):
                rr.extend(v for v in intervals if v is not None and v > 0)
            elif isinstance(intervals, (int, float)) and intervals > 0:
                rr.append(float(intervals))
    except Exception as exc:
        log.debug("HRV zprávy nelze přečíst: %s", exc)
    return rr


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLACE CHYBĚJÍCÍCH HODNOT
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_none(
    values: list,
    timestamps: Optional[list] = None,
    max_gap_seconds: float = 15.0,
    max_gap: int = 10,
) -> list:
    """
    Lineární interpolace None mezer v číselném seznamu.

    Pokud jsou zadány ``timestamps`` (seznam datetime objektů paralelní s
    ``values``), rozhoduje o interpolaci **časový rozdíl** mezi sousedními
    platnými hodnotami: mezery delší než ``max_gap_seconds`` se ponechají jako
    None.  Tím se správně zpracovávají soubory s Garmin Smart Recording, kde
    10 řádků snadno trvá přes minutu (např. 10 × 12 s = 120 s) a interpolace
    přes celý úsek by falzifikovala rychlost a nadmořskou výšku.

    Bez ``timestamps`` se použije záložní row-count limit ``max_gap``.
    """
    result = list(values)
    n = len(result)
    i = 0
    while i < n:
        if result[i] is None:
            j = i + 1
            while j < n and result[j] is None:
                j += 1
            # Determine whether to interpolate this gap
            should_interpolate = False
            if timestamps is not None and i > 0 and j < n:
                ts_left  = timestamps[i - 1]
                ts_right = timestamps[j]
                if isinstance(ts_left, datetime) and isinstance(ts_right, datetime):
                    gap_secs = (ts_right - ts_left).total_seconds()
                    should_interpolate = gap_secs <= max_gap_seconds
                else:
                    # timestamps present but not datetimes – fall back to row count
                    should_interpolate = (j - i) <= max_gap
            else:
                should_interpolate = (j - i) <= max_gap

            if should_interpolate:
                left  = result[i - 1] if i > 0 else None
                right = result[j]     if j < n else None
                for k in range(i, j):
                    if left is not None and right is not None:
                        t = (k - i + 1) / (j - i + 1)
                        result[k] = left + t * (right - left)
                    elif left is not None:
                        result[k] = left
                    elif right is not None:
                        result[k] = right
            # else: gap too long → leave as None (GPS dropout / long pause)
            i = j
        else:
            i += 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARSOVÁNÍ ID Z NÁZVU SOUBORU
# ─────────────────────────────────────────────────────────────────────────────

_ID_PATTERNS = [
    re.compile(r"activity_(\d+)\.fit$",   re.IGNORECASE),
    re.compile(r"^(\d+)_ACTIVITY\.fit$",  re.IGNORECASE),
    re.compile(r"(\d+)\.fit$",            re.IGNORECASE),
]

def extract_activity_id(file_path: str) -> str:
    base = os.path.basename(file_path)
    for pat in _ID_PATTERNS:
        m = pat.search(base)
        if m:
            return m.group(1)
    return base.replace(".fit", "")


# ─────────────────────────────────────────────────────────────────────────────
# ZPRACOVÁNÍ JEDNOHO FIT SOUBORU
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def parse_fit_file(
    file_path: str,
    highres_writer: csv.DictWriter,
) -> Optional[dict]:
    """
    Zpracuje jeden FIT soubor – čistá extrakce dat z FIT struktury.
    Zapíše záznamy do highres_writer (streaming, nízká RAM).
    Vrátí summary dict nebo None při chybě.
    """
    activity_id = extract_activity_id(file_path)
    act_date: Optional[date] = None
    act_name  = ""
    sport     = ""

    try:
        fitfile = FitFile(file_path)
    except Exception as exc:
        log.error("Nelze otevřít %s: %s", file_path, exc)
        return None

    # Načti všechny 'record' zprávy (typicky < 15 000 řádků = bezpečné)
    raw_records: list[dict] = [m.get_values() for m in fitfile.get_messages("record")]

    if not raw_records:
        log.warning("[%s] Žádné 'record' zprávy – přeskakuji.", activity_id)
        return None

    # Ultra-endurance safety: warn when activity is suspiciously long.
    # 12 h × 3600 s/h = 43 200 records (1 Hz recording, worst case).
    # Smart Recording uses fewer points, so 43 200 is a very conservative cap.
    _MAX_RECORDS_12H = 43_200
    if len(raw_records) > _MAX_RECORDS_12H:
        log.warning(
            "[%s] Ultra-endurance aktivita: %d záznamů (>12 h @ 1 Hz). "
            "Zpracovávám, ale RAM může být vyšší.",
            activity_id, len(raw_records),
        )

    # Session-level metriky (fyzické, výkonnostní)
    session_distance: Optional[float] = None
    session_ascent: Optional[float] = None
    session_descent: Optional[float] = None
    session_avg_speed: Optional[float] = None
    session_max_speed: Optional[float] = None
    session_calories: Optional[int] = None
    session_avg_cadence: Optional[int] = None
    session_max_cadence: Optional[int] = None
    session_avg_temp: Optional[float] = None
    session_max_temp: Optional[float] = None
    session_te_aerobic: Optional[float] = None
    session_te_anaerobic: Optional[float] = None
    session_vo2_max: Optional[float] = None

    # Sport + datum + fyzické/výkonnostní metriky ze 'session' zprávy
    for msg in fitfile.get_messages("session"):
        vals = msg.get_values()
        s = vals.get("sport")
        if s:
            sport = str(s)
        ss = vals.get("sub_sport")
        if ss:
            sport = f"{sport}/{ss}"
        if act_date is None:
            ts = vals.get("start_time")
            if isinstance(ts, datetime):
                act_date = ts.date()

        # Fyzické metriky ze session
        session_distance = _safe_float(vals.get("total_distance"))
        session_ascent   = _safe_float(vals.get("total_ascent"))
        session_descent  = _safe_float(vals.get("total_descent"))
        # Rychlost – preferuj enhanced verze (m/s), pak normální, pak vypočti
        session_avg_speed = _safe_float(vals.get("enhanced_avg_speed")) or _safe_float(vals.get("avg_speed"))
        session_max_speed = _safe_float(vals.get("enhanced_max_speed")) or _safe_float(vals.get("max_speed"))
        # Záloha: vypočti avg z distance/time
        if not session_avg_speed and session_distance:
            timer = _safe_float(vals.get("total_timer_time"))
            if timer and timer > 0:
                session_avg_speed = session_distance / timer  # m/s
        session_calories  = _safe_int(vals.get("total_calories"))
        # Kadence – Garmin ukládá jako avg_running_cadence (běh) nebo avg_cadence (kolo)
        session_avg_cadence = (_safe_int(vals.get("avg_cadence"))
                               or _safe_int(vals.get("avg_running_cadence")))
        session_max_cadence = (_safe_int(vals.get("max_cadence"))
                               or _safe_int(vals.get("max_running_cadence")))
        session_avg_temp    = _safe_float(vals.get("avg_temperature"))
        session_max_temp    = _safe_float(vals.get("max_temperature"))
        # Výkonnostní metriky
        session_te_aerobic   = _safe_float(vals.get("total_training_effect"))
        session_te_anaerobic = _safe_float(vals.get("total_anaerobic_training_effect"))
        # VO2max – hledej ve více polích (žádná náhrada odpechováním – ta je fyziologicky nesmyslná)
        session_vo2_max = None
        for vo2_key in ("vo2_max", "estimated_vo2_max", "vo2max"):
            v = _safe_float(vals.get(vo2_key))
            if v is not None and v > 0:
                # Garmin scale-factor sanity check.
                # Some firmware versions emit a raw fixed-point integer
                # (e.g. ~900 000) instead of the human-readable value.
                # Applying val / 65536 * 3.5 recovers the real ml/kg/min figure.
                # If the parsed value is already in the realistic VO2max range
                # (10–100 ml/kg/min), use it directly.
                if v > 200:
                    # Raw large integer – apply Garmin multiplier
                    v_scaled = v / 65536 * 3.5
                    if 10.0 <= v_scaled <= 100.0:
                        v = v_scaled
                elif v < 10:
                    # Suspiciously small – attempt multiplier (per Garmin spec)
                    v_scaled = v / 65536 * 3.5
                    v = v_scaled if 10.0 <= v_scaled <= 100.0 else None
                session_vo2_max = v
                break
        break

    # VO2max záloha z user_profile zpráv (pokud session zpráva neobsahovala)
    if session_vo2_max is None:
        for _up_msg in fitfile.get_messages("user_profile"):
            _up_vals = _up_msg.get_values()
            for _fid in (0, 1):
                _v = _safe_float(_up_vals.get(_fid))
                if _v is not None and 10.0 <= _v <= 100.0:
                    session_vo2_max = _v
                    break
            if session_vo2_max is not None:
                break

    # Záloha data z prvního záznamu
    if act_date is None:
        ts0 = raw_records[0].get("timestamp")
        if isinstance(ts0, datetime):
            act_date = ts0.date()

    # Klidový tep: pouze baseline (externí data se nenačítají)
    rhr: int = BASELINE_RHR
    zones = compute_zones(rhr)
    z2_cap_used = ZONE_2_CAP

    # Dynamický speed threshold podle sportu
    # Rozděl sport string "cycling/gravel_cycling" na hlavní + sub-sport
    _sport_parts = [p.strip().lower() for p in sport.split("/") if p.strip()]
    _is_cycling = bool(_sport_parts and any(p in CYCLING_SPORTS for p in _sport_parts))
    speed_threshold = CYCLING_SPEED_THRESHOLD_MS if _is_cycling else DEFAULT_SPEED_THRESHOLD_MS
    if _is_cycling:
        log.debug("[%s] Cycling sport detected (%s) → speed threshold %.3f m/s",
                  activity_id, sport, speed_threshold)

    # HRV z FIT (RMSSD z R-R intervalů)
    rr_intervals   = extract_rr_from_fit(fitfile)
    workout_rmssd  = compute_rmssd(rr_intervals) if rr_intervals else None

    # Interpolace altitude + speed (preferuj enhanced_speed → speed)
    # ── Pandas-based interpolation (C-optimized, replaces Python while-loop) ─
    altitudes_series = pd.Series([_safe_float(r.get("altitude") or r.get("enhanced_altitude")) for r in raw_records]).infer_objects(copy=False)
    speeds_series = pd.Series([_safe_float(r.get("enhanced_speed") or r.get("speed")) for r in raw_records]).infer_objects(copy=False)

    # Použijeme lineární interpolaci s limitem 15 záznamů (což odpovídá cca 15 vteřinám)
    altitudes_series = altitudes_series.interpolate(method='linear', limit=15)
    speeds_series = speeds_series.interpolate(method='linear', limit=15)

    altitudes = altitudes_series.tolist()
    speeds = speeds_series.tolist()

    # ── Paused intervals (event messages – timer stop/start) ─────────────────
    # Build a list of (pause_start, resume_ts) tuples so records that fall
    # within a paused window are correctly marked is_active=False, even when
    # speed is non-zero (Smart Recording may carry over the last speed value).
    _paused_intervals: list[tuple[datetime, datetime]] = []
    _pause_start_ts: Optional[datetime] = None
    try:
        for _ev_msg in fitfile.get_messages("event"):
            _ev_vals = _ev_msg.get_values()
            _ev_name  = str(_ev_vals.get("event",      "")).lower()
            _ev_type  = str(_ev_vals.get("event_type", "")).lower()
            _ev_ts    = _ev_vals.get("timestamp")
            if not isinstance(_ev_ts, datetime):
                continue
            if _ev_name == "timer" and _ev_type in ("stop", "stop_disable", "stop_disable_all"):
                _pause_start_ts = _ev_ts
            elif _ev_name == "timer" and _ev_type in ("start", "start_data"):
                if _pause_start_ts is not None:
                    _paused_intervals.append((_pause_start_ts, _ev_ts))
                    _pause_start_ts = None
    except Exception:
        pass  # event zprávy jsou nepovinné – pokud chybí, fallback na speed > 0

    def _in_pause(ts: Optional[datetime]) -> bool:
        """True pokud timestamp padne do detekovaného přestávkového intervalu."""
        if ts is None or not _paused_intervals:
            return False
        return any(p0 <= ts <= p1 for p0, p1 in _paused_intervals)

    # Akumulátory
    hr_values: list[int] = []
    zone_times: dict[str, float] = defaultdict(float)
    total_trimp = 0.0
    records_count = 0
    prev_ts: Optional[datetime] = None
    active_seconds: float = 0.0        # čas is_active=True (pro správný výpočet duration)
    uphill_seconds: float = 0.0        # čas stoupání s pohybem (pro VAM)
    prev_alt: Optional[float] = None   # předchozí nadmořská výška pro detekci stoupání

    for idx, raw in enumerate(raw_records):
        ts = raw.get("timestamp")
        ts_dt: Optional[datetime] = ts if isinstance(ts, datetime) else None
        if ts_dt is None and isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass

        seg_s = max(0.0, min(120.0, (ts_dt - prev_ts).total_seconds())) if (ts_dt and prev_ts) else 1.0
        prev_ts = ts_dt

        hr   = _safe_int(raw.get("heart_rate"))
        spd  = speeds[idx]
        dist = _safe_float(raw.get("distance"))
        alt  = altitudes[idx]
        cad  = _safe_int(raw.get("cadence"))
        pwr  = _safe_int(raw.get("power"))
        temp = _safe_float(raw.get("temperature"))
        vert_osc   = _safe_float(raw.get("vertical_oscillation"))
        stance_t   = _safe_float(raw.get("stance_time"))
        resp_rate  = _safe_float(raw.get("respiratory_rate") or raw.get("respiration_rate"))
        # HRV: some devices embed per-record HRV (beat-to-beat interval in ms)
        hrv_val    = _safe_float(raw.get("hrv") or raw.get("heart_rate_variability"))

        # Zone must be computed before is_active so the HR-zone fallback works.
        zone = classify_zone(hr, zones)
        # A record is active when:
        #  (a) speed is above threshold (normal forward movement), OR
        #  (b) speed is low/zero but HR is in Z2 or higher
        #      (steep climb, technical stop, track change in cycling).
        # Pause detection wins in both cases.
        _above_speed = spd is not None and spd > speed_threshold
        # Exclude both Z1 and empty string: classify_zone returns "" when HR
        # is missing, which must NOT be treated as an active high-HR zone.
        _above_z1    = zone is not None and zone not in ("Z1", "")
        is_active = (_above_speed or _above_z1) and not _in_pause(ts_dt)

        t_inc = trimp_increment(hr, seg_s, rhr)
        if hr is not None:
            hr_values.append(hr)
        # Only accumulate TRIMP and zone time during active movement;
        # stationary/paused intervals (rest between sets, traffic lights)
        # must not inflate the training load metrics.
        if is_active:
            total_trimp += t_inc
            if zone:
                zone_times[zone] += seg_s / 60.0

        # Active-time accumulator (used for duration_min – avoids bag-activity inflation)
        if is_active:
            active_seconds += seg_s

        # Uphill-time accumulator (used for VAM in athlete_analytics.py)
        if (is_active
                and alt is not None and prev_alt is not None
                and alt > prev_alt
                and spd is not None and spd > 0):
            uphill_seconds += seg_s
        prev_alt = alt

        highres_writer.writerow({
            "activity_id":     activity_id,
            "timestamp":       ts_dt.isoformat() if ts_dt else "",
            "date":            (ts_dt.date().isoformat() if ts_dt else (act_date.isoformat() if act_date else "")),
            "heart_rate":      hr if hr is not None else "",
            "speed":           f"{spd:.4f}" if spd is not None else "",
            "distance":        f"{dist:.2f}" if dist is not None else "",
            "altitude":        f"{alt:.2f}" if alt is not None else "",
            "cadence":         cad if cad is not None else "",
            "power":           pwr if pwr is not None else "",
            "temperature":     f"{temp:.1f}" if temp is not None else "",
            "vertical_oscillation": f"{vert_osc:.2f}" if vert_osc is not None else "",
            "stance_time":     f"{stance_t:.1f}" if stance_t is not None else "",
            "respiratory_rate": f"{resp_rate:.1f}" if resp_rate is not None else "",
            "hrv":             f"{hrv_val:.1f}" if hrv_val is not None else "",
            "is_active":       is_active,
            "hr_zone":         zone,
            "trimp_increment": f"{t_inc:.5f}",
        })
        records_count += 1

    # Summary výpočty
    avg_hr_val  = round(mean(hr_values)) if hr_values else None
    # Clamp max HR to MAX_HR – device glitches occasionally report
    # physiologically impossible values (220+ bpm).
    max_hr_val  = min(max(hr_values), MAX_HR) if hr_values else None

    # Duration from active records only (prevents "bag activity" inflation where e.g.
    # a forgotten soccer watch accumulates 5 h but only 6 TRIMP).
    # Priority: active_seconds > zone_times sum > raw record count (last resort).
    zone_min = sum(zone_times.values())
    if active_seconds > 0:
        duration_min = active_seconds / 60.0
    elif zone_min > 0:
        duration_min = zone_min
    else:
        duration_min = records_count / 60.0

    # NOTE: hrv_recovery_index removed – robust per-activity HRV is computed
    # by athlete_analytics.py (max_hrr_60s_for_activity / workout RMSSD).

    # ── Převody jednotek (session) ──────────────────────────────────────────
    distance_km    = round(session_distance / 1000.0, 3) if session_distance is not None else None
    avg_speed_kmh  = round(session_avg_speed * 3.6, 2)   if session_avg_speed is not None else None
    # Vždy ignorujeme session_max_speed z Garminu – hodinky mohou propsat
    # GPS glitch přímo do session hlavičky.  Počítáme robustní max z record dat.
    record_speeds = [s for s in speeds if s is not None and s > 0]
    if record_speeds:
        # 3sekundové mediánové vyhlazení + 99. percentil → spolehlivě odstraní GPS glitche
        spds = pd.Series(record_speeds)
        real_max_ms = spds.rolling(3, min_periods=1).median().quantile(0.99)
        max_speed_kmh = round(real_max_ms * 3.6, 2)
    else:
        max_speed_kmh = None

    # ── GPS sanity caps (sport-aware) ───────────────────────────────────────
    # Physical upper limits: values above these are sensor artefacts / GPS jumps.
    _sport_lower = (sport or "").lower()
    _is_cycling_sport = any(p in _sport_lower for p in ("cycling", "biking"))
    _is_slow_sport    = any(p in _sport_lower for p in (
        "running", "trail", "soccer", "football", "hiking", "walking",
        "fitness", "gym", "swim",
    ))
    if _is_cycling_sport:
        _max_spd_cap = 100.0   # downhill cycling can hit 80+ km/h
        _avg_spd_cap = 60.0
    elif _is_slow_sport:
        _max_spd_cap = 35.0
        _avg_spd_cap = 30.0
    else:
        _max_spd_cap = 45.0
        _avg_spd_cap = 40.0

    if max_speed_kmh is not None and max_speed_kmh > _max_spd_cap:
        log.warning("[%s] max_speed_kmh=%.1f exceeds limit %.1f for sport '%s' → nullified",
                    activity_id, max_speed_kmh, _max_spd_cap, sport)
        max_speed_kmh = None

    if avg_speed_kmh is not None and avg_speed_kmh > _avg_spd_cap:
        log.warning("[%s] avg_speed_kmh=%.1f exceeds limit %.1f for sport '%s' → nullified",
                    activity_id, avg_speed_kmh, _avg_spd_cap, sport)
        avg_speed_kmh = None
        distance_km = None  # implied speed nesmyslný → vzdálenost také nespolehlivá

    # Kontrolní průměrná rychlost z GPS distance / active duration:
    # pokud se liší od session avg a taktéž překročuje limit, vzdálenost vyhodíme.
    if distance_km is not None and duration_min > 0:
        _implied_avg = (distance_km / (duration_min / 60.0))  # km/h
        if _implied_avg > _avg_spd_cap:
            log.warning("[%s] implied avg speed %.1f km/h (%.3f km / %.1f min) exceeds "
                        "limit %.1f for sport '%s' → distance_km nullified",
                        activity_id, _implied_avg, distance_km, duration_min,
                        _avg_spd_cap, sport)
            distance_km = None

    ascent_m       = round(session_ascent, 1) if session_ascent is not None else None
    descent_m      = round(session_descent, 1) if session_descent is not None else None

    # Logování
    # NOTE: HRR (Heart Rate Recovery) was removed – the per-activity rolling
    # max_hrr_60s computed in athlete_analytics.py is more robust.
    is_match = any(w in act_name.lower() for w in
                   ["fotbal", "soccer", "football", "zapas", "match", "game"])
    z5_load  = zone_times.get("Z5", 0.0) + zone_times.get("Z5+", 0.0)
    extras   = ""
    if is_match:
        extras += "  [FOTBALOVÝ ZÁPAS]"
    if z5_load > 5:
        extras += f"  [VYSOKÁ Z5 ZÁTĚŽ: {z5_load:.1f} min]"

    log.info(
        "[%s] %s | %-14s | trvání=%5.1f min | TRIMP=%6.1f | Z2=%.1f min | Z5=%.1f min%s",
        activity_id,
        act_date.isoformat() if act_date else "????-??-??",
        sport[:14],
        duration_min,
        total_trimp,
        zone_times.get("Z2", 0.0),
        z5_load,
        extras,
    )

    # ── Synthetic TE fallback (Karvonen-recalibrated) ─────────────────────
    # Používáme nově přepočítaný TRIMP (MAX_HR=199, HRR=158) a čas v nových
    # zónách jako záložní výpočet, pokud Garmin nativní TE chybí v FIT souboru.
    #
    # Aerobic TE  ~ 5 × (1 − e^(−TRIMP / 50))   ← saturuje na 5.0 při ~200 TRIMP
    # Anaerobic TE ~ 5 × (1 − e^(−Z4Z5_min / 20)) ← saturuje na 5.0 při ~80 min Z4/Z5
    if session_te_aerobic is None and total_trimp > 0:
        session_te_aerobic = round(
            min(5.0, 5.0 * (1.0 - math.exp(-total_trimp / 50.0))), 1
        )
        log.debug("[%s] Synthetic Aerobic TE = %.1f (TRIMP=%.1f)", activity_id,
                  session_te_aerobic, total_trimp)

    z4z5_min = zone_times.get("Z4", 0.0) + z5_load
    if session_te_anaerobic is None and z4z5_min > 0:
        session_te_anaerobic = round(
            min(5.0, 5.0 * (1.0 - math.exp(-z4z5_min / 20.0))), 1
        )
        log.debug("[%s] Synthetic Anaerobic TE = %.1f (Z4+Z5=%.1f min)", activity_id,
                  session_te_anaerobic, z4z5_min)

    return {
        "activity_id":        activity_id,
        "date":               (act_date.isoformat() if act_date else ""),
        "activity_name":      act_name,
        "sport":              sport,
        "duration_minutes":   round(duration_min, 2),
        "total_trimp":        round(total_trimp, 2),
        "avg_hr":             avg_hr_val if avg_hr_val is not None else "",
        "max_hr":             max_hr_val if max_hr_val is not None else "",
        "time_in_z1":         round(zone_times.get("Z1",  0.0), 2),
        "time_in_z2":         round(zone_times.get("Z2",  0.0), 2),
        "time_in_z3":         round(zone_times.get("Z3",  0.0), 2),
        "time_in_z4":         round(zone_times.get("Z4",  0.0), 2),
        "time_in_z5":         round(z5_load, 2),
        "zone2_cap_used":     z2_cap_used,
        "records_count":      records_count,
        # Fyzické metriky
        "distance_km":        distance_km if distance_km is not None else "",
        "ascent_m":           ascent_m if ascent_m is not None else "",
        "descent_m":          descent_m if descent_m is not None else "",
        "avg_speed_kmh":      avg_speed_kmh if avg_speed_kmh is not None else "",
        "max_speed_kmh":      max_speed_kmh if max_speed_kmh is not None else "",
        "calories":           session_calories if session_calories is not None else "",
        # Senzory
        "avg_cadence":        session_avg_cadence if session_avg_cadence is not None else "",
        "max_cadence":        session_max_cadence if session_max_cadence is not None else "",
        "avg_temp":           session_avg_temp if session_avg_temp is not None else "",
        "max_temp":           session_max_temp if session_max_temp is not None else "",
        # Výkonnostní
        "training_effect_aerobic":   session_te_aerobic if session_te_aerobic is not None else "",
        "training_effect_anaerobic": session_te_anaerobic if session_te_anaerobic is not None else "",
        "vo2_max":            session_vo2_max if session_vo2_max is not None else "",
        # Stoupání (pro VAM)
        "uphill_minutes":     round(uphill_seconds / 60.0, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL-SAFE VARIANT: parse FIT file without a shared writer
# ─────────────────────────────────────────────────────────────────────────────

class _RowCollector:
    """
    Mimics the csv.DictWriter.writerow interface used by parse_fit_file.
    Accumulates row dicts into an in-memory list instead of writing to disk.
    Passed to parse_fit_file in worker processes where a shared file handle
    cannot exist.
    """
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def writerow(self, row: dict) -> None:
        self.rows.append(dict(row))


def parse_fit_to_memory(
    file_path: str,
) -> Optional[tuple[dict, list[dict]]]:
    """
    Parallel-safe variant of parse_fit_file.

    Opens the FIT file, processes all records, and returns
    (summary_dict, highres_rows) instead of writing to a shared CSV writer.
    Returns None if parsing fails (mirrors parse_fit_file's error contract).

    Intended for use with multiprocessing.Pool.imap so each worker is
    fully independent with no shared I/O state.
    """
    collector = _RowCollector()
    summary = parse_fit_file(file_path, collector)
    if summary is None:
        return None
    return summary, collector.rows


# ─────────────────────────────────────────────────────────────────────────────
# HLAVNÍ FUNKCE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("Garmin AI Trainer – FIT → CSV  (MAX_HR=%d bpm, baseline_RHR=%d bpm)",
             MAX_HR, BASELINE_RHR)
    log.info("=" * 70)

    os.makedirs(SUMMARY_FOLDER, exist_ok=True)

    # Zobraz dnešní zóny jako referenci (baseline RHR)
    log.info("Karvonen zóny (baseline RHR=%d bpm):", BASELINE_RHR)
    for lo, hi, label in compute_zones(BASELINE_RHR):
        suffix = "  ← Z2 cap" if label == "Z2" else ""
        log.info("  %s: %3d – %3d bpm%s", label, lo, hi, suffix)

    fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
    if not fit_files:
        log.error("Žádné .fit soubory v %s", FIT_FOLDER)
        return

    log.info("Nalezeno %d FIT souborů. Zahajuji zpracování...", len(fit_files))

    summary_rows: list[dict] = []
    processed = failed = 0

    # Streaming zápis do high_res_training_data.csv (memory-safe)
    with open(HIGHRES_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HIGHRES_COLS)
        writer.writeheader()
        for i, fit_path in enumerate(fit_files, 1):
            log.info("─── [%d/%d] %s", i, len(fit_files), os.path.basename(fit_path))
            result = parse_fit_file(fit_path, writer)
            if result is not None:
                summary_rows.append(result)
                processed += 1
            else:
                failed += 1
            if i % 10 == 0:
                fh.flush()   # periodický flush → nízká RAM

    # Summary CSV
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as sf:
        ws = csv.DictWriter(sf, fieldnames=SUMMARY_COLS)
        ws.writeheader()
        ws.writerows(summary_rows)

    log.info("=" * 70)
    log.info("DOKONČENO  Zpracováno: %d aktivit  |  Chyb: %d", processed, failed)
    if summary_rows:
        trimps = [r["total_trimp"] for r in summary_rows if r["total_trimp"]]
        log.info("Celkový TRIMP:          %.1f", sum(trimps))
        log.info("Průměrný TRIMP/aktivita:%.1f", mean(trimps) if trimps else 0)
    log.info("High-res CSV : %s", HIGHRES_CSV)
    log.info("Summary CSV  : %s", SUMMARY_CSV)
    log.info("")
    log.info("Upgrade dokončen: Extrahováno %d nových metrik u %d aktivit.", 14, processed)
    log.info("=" * 70)


if __name__ == "__main__":
    main()