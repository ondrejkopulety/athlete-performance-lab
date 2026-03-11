#!/usr/bin/env python3
"""
src/core/stop_analysis.py – Shared Stop-Detection Primitives
=============================================================
Single Source of Truth for the botanical analytics pipeline.

Exports
-------
Data structures:
    Record          – one second of FIT/CSV data
    StopSegment     – a merged low-speed segment

Canonical algorithm:
    detect_stops()  – gap-tolerant stop finder with GPS blackout rejection

Shared helpers:
    _safe_float, _safe_int, _semicircles_to_deg, _speed_ms_to_kmh
    _haversine_m, _is_in_excluded_zone
    _is_cardio  (alias: _is_cycling)
    extract_activity_id  (alias: _extract_activity_id)
    parse_fit_records

All tunable thresholds are imported from config.settings – never harcode
them here.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from datetime import datetime
from typing import Optional, NamedTuple

import numpy as np
import pandas as pd
from fitparse import FitFile

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import (
    MIN_STOP_DURATION_S,
    STOP_SPEED_THRESHOLD_KMH,
    MAX_AUTO_PAUSE_GAP_SEC,
    EXCLUDED_LOCATIONS,
    ENABLE_EXCLUSION_ZONES,
    CARDIO_SPORTS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class Record(NamedTuple):
    """One time-stamped sample from a FIT file or the master high-res CSV."""
    timestamp:        datetime
    heart_rate:       Optional[float]
    speed_ms:         Optional[float]          # m/s
    distance_m:       Optional[float]          # cumulative metres
    altitude_m:       Optional[float]
    cadence:          Optional[float]          # rpm
    power:            Optional[float]          # watts
    respiratory_rate: Optional[float]          # breaths / min
    lat:              Optional[float]          # WGS-84 degrees
    lon:              Optional[float]          # WGS-84 degrees


class StopSegment(NamedTuple):
    """A single merged low-speed segment detected by detect_stops()."""
    start_idx:  int
    end_idx:    int
    duration_s: float
    start_time: datetime
    end_time:   datetime


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    """Cast *v* to float; return None on failure or NaN."""
    try:
        f = float(v) if v is not None else None
        return None if (f is not None and np.isnan(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    """Cast *v* to int; return None on failure."""
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _semicircles_to_deg(sc: Optional[float]) -> Optional[float]:
    """Convert Garmin semicircles to WGS-84 degrees."""
    if sc is None:
        return None
    return sc * (180.0 / 2 ** 31)


def _speed_ms_to_kmh(v: Optional[float]) -> float:
    """Convert m/s to km/h; default 0.0 when *v* is None."""
    if v is None:
        return 0.0
    return v * 3.6


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance between two WGS-84 points in metres."""
    r = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _is_in_excluded_zone(lat: float, lon: float) -> bool:
    """
    Return True when (lat, lon) falls inside any entry in EXCLUDED_LOCATIONS.

    Respects the ENABLE_EXCLUSION_ZONES toggle from config.settings.
    """
    if not ENABLE_EXCLUSION_ZONES:
        return False
    for _name, (elat, elon, eradius) in EXCLUDED_LOCATIONS.items():
        if _haversine_m(lat, lon, elat, elon) <= eradius:
            return True
    return False


def _is_cardio(sport_str: str) -> bool:
    """
    Return True if *sport_str* is a cardio activity (cycling / running / bike).
    Uses the CARDIO_SPORTS frozenset from config.settings and a few keyword
    fallbacks for sport strings that don't match the canonical names exactly.
    """
    s = sport_str.lower().replace(" ", "_")
    for cs in CARDIO_SPORTS:
        if cs in s:
            return True
    return "cycl" in s or "bik" in s or "run" in s


# Backwards-compatibility alias used by some older call-sites
_is_cycling = _is_cardio


def extract_activity_id(file_path: str) -> str:
    """Extract a numeric activity-ID string from a FIT file name."""
    patterns = [
        re.compile(r"activity_(\d+)\.fit$", re.IGNORECASE),
        re.compile(r"^(\d+)_ACTIVITY\.fit$", re.IGNORECASE),
        re.compile(r"(\d+)\.fit$", re.IGNORECASE),
    ]
    base = os.path.basename(file_path)
    for pat in patterns:
        m = pat.search(base)
        if m:
            return m.group(1)
    return base.replace(".fit", "")


# Backwards-compatibility alias (hotspots script used leading underscore)
_extract_activity_id = extract_activity_id


# ─────────────────────────────────────────────────────────────────────────────
# FIT FILE PARSER  (shared, canonical version)
# ─────────────────────────────────────────────────────────────────────────────

def parse_fit_records(
    file_path: str,
) -> tuple[str, str, datetime | None, list[Record]]:
    """
    Parse a FIT file and return (activity_id, sport, start_datetime, [Record, …]).

    Returns an empty records list when the file cannot be opened or contains
    no record messages.  Never raises – all exceptions are caught and logged.
    """
    activity_id = extract_activity_id(file_path)
    sport: str = ""
    start_dt: datetime | None = None

    try:
        fitfile = FitFile(file_path)
    except Exception as exc:
        log.error("Cannot open %s: %s", file_path, exc)
        return activity_id, sport, None, []

    # Session message → sport + start_time
    try:
        for msg in fitfile.get_messages("session"):
            vals = msg.get_values()
            s = vals.get("sport")
            if s:
                sport = str(s)
            ss = vals.get("sub_sport")
            if ss:
                sport = f"{sport}/{ss}"
            ts = vals.get("start_time")
            if isinstance(ts, datetime):
                start_dt = ts
            break
    except Exception:
        pass

    # Record messages
    records: list[Record] = []
    try:
        record_messages = list(fitfile.get_messages("record"))
    except Exception:
        return activity_id, sport, start_dt, []

    for msg in record_messages:
        try:
            v = msg.get_values()
        except Exception:
            continue

        ts = v.get("timestamp")
        if not isinstance(ts, datetime):
            continue

        # Speed – prefer enhanced_speed
        speed = _safe_float(v.get("enhanced_speed")) or _safe_float(v.get("speed"))

        # GPS – Garmin stores in semicircles OR already in WGS-84 degrees
        raw_lat = _safe_float(v.get("position_lat"))
        raw_lon = _safe_float(v.get("position_long"))
        lat = _semicircles_to_deg(raw_lat) if raw_lat and abs(raw_lat) > 180 else raw_lat
        lon = _semicircles_to_deg(raw_lon) if raw_lon and abs(raw_lon) > 180 else raw_lon

        # Respiration – stored under several key names across firmware versions
        resp = (
            _safe_float(v.get("respiration_rate"))
            or _safe_float(v.get("respiratory_rate"))
            or _safe_float(v.get("enhanced_respiration_rate"))
        )

        records.append(
            Record(
                timestamp=ts,
                heart_rate=_safe_float(v.get("heart_rate")),
                speed_ms=speed,
                distance_m=_safe_float(v.get("distance")),
                altitude_m=(
                    _safe_float(v.get("enhanced_altitude"))
                    or _safe_float(v.get("altitude"))
                ),
                cadence=_safe_float(v.get("cadence")),
                power=_safe_float(v.get("power")),
                respiratory_rate=resp,
                lat=lat,
                lon=lon,
            )
        )

    return activity_id, sport, start_dt, records


# ─────────────────────────────────────────────────────────────────────────────
# STOP DETECTION  (canonical, gap-tolerant implementation with Wall-Clock & Debounce)
# ─────────────────────────────────────────────────────────────────────────────

# Local threshold for Wall-Clock Gap Detection (not in settings.py to avoid breaking changes)
WALL_CLOCK_GAP_THRESHOLD_S: int = 300  # 5 minutes – gaps > this between ANY rows = implicit stop

# GPS drift speed threshold: rows with speed < this during a "moving gap" are considered drift
GPS_DRIFT_SPEED_KMH: float = 3.5  # Slightly above STOP_SPEED_THRESHOLD_KMH to catch drift


def _find_implicit_stops_from_gaps(records: list[Record]) -> list[tuple[int, int, float]]:
    """
    Detect implicit stops from large wall-clock gaps between consecutive records.

    Auto-Pause stops recording entirely during long stops, leaving a time gap
    between rows with no data.  A gap > WALL_CLOCK_GAP_THRESHOLD_S between ANY
    two consecutive records is treated as an implicit stop.

    Returns list of (row_before_gap, row_after_gap, gap_duration_s).
    """
    implicit_stops: list[tuple[int, int, float]] = []
    if len(records) < 2:
        return implicit_stops

    for i in range(1, len(records)):
        prev_ts = records[i - 1].timestamp
        curr_ts = records[i].timestamp
        gap_s = (curr_ts - prev_ts).total_seconds()

        if gap_s > WALL_CLOCK_GAP_THRESHOLD_S:
            # Verify this is a genuine stop via odometer check (not GPS blackout while moving)
            dist_prev = records[i - 1].distance_m
            dist_curr = records[i].distance_m
            if (
                dist_prev is not None
                and dist_curr is not None
                and not pd.isna(dist_prev)
                and not pd.isna(dist_curr)
                and gap_s > 0
            ):
                gap_speed_kmh = ((dist_curr - dist_prev) / 1000.0) / (gap_s / 3600.0)
                if gap_speed_kmh > 6.0:
                    # Athlete was moving – this is a GPS/sensor blackout, not a stop
                    log.debug(
                        "Implicit gap rejected (moving): %.0f m in %.1f min = %.1f km/h",
                        dist_curr - dist_prev, gap_s / 60.0, gap_speed_kmh,
                    )
                    continue

            implicit_stops.append((i - 1, i, gap_s))
            log.debug(
                "Implicit stop from wall-clock gap: %.1f min (%s – %s)",
                gap_s / 60.0, prev_ts, curr_ts,
            )

    return implicit_stops


def _merge_overlapping_segments(
    segments: list[tuple[int, int, datetime, datetime]],
    records: list[Record],
    debounce_gap_s: float,
) -> list[tuple[int, int, datetime, datetime]]:
    """
    Merge segments if the wall-clock gap between consecutive segments is ≤ debounce_gap_s
    (GPS drift tolerance / segment debouncing).

    Each segment is (start_idx, end_idx, start_time, end_time).
    Adjacent segments are merged if:
        - The end of segment A to the start of segment B is ≤ debounce_gap_s AND
        - The "moving" rows between them have low/drift-level speeds
    """
    if not segments:
        return []

    # Sort by start index
    segments = sorted(segments, key=lambda s: s[0])
    merged: list[tuple[int, int, datetime, datetime]] = []

    current_start_idx, current_end_idx, current_start_time, current_end_time = segments[0]

    for seg in segments[1:]:
        next_start_idx, next_end_idx, next_start_time, next_end_time = seg

        # Wall-clock gap between end of current and start of next
        gap_s = (next_start_time - current_end_time).total_seconds()

        # Check if we should merge (gap is small OR the rows between are drift-speed)
        should_merge = False

        if gap_s <= 0:
            # Overlapping or contiguous – always merge
            should_merge = True
        elif gap_s <= debounce_gap_s:
            # Gap is within tolerance – check if intermediate rows are drift/noise
            # Look at rows between current_end_idx and next_start_idx
            if next_start_idx - current_end_idx <= 1:
                # Adjacent indices – merge
                should_merge = True
            else:
                # Check speeds of intermediate rows
                intermediate_indices = range(current_end_idx + 1, next_start_idx)
                high_speed_count = 0
                total_intermediate = 0
                for idx in intermediate_indices:
                    if idx < len(records):
                        total_intermediate += 1
                        speed_kmh = _speed_ms_to_kmh(records[idx].speed_ms)
                        if speed_kmh > GPS_DRIFT_SPEED_KMH:
                            high_speed_count += 1

                # Merge if most intermediate rows are low-speed (drift/noise)
                if total_intermediate == 0 or high_speed_count / total_intermediate < 0.5:
                    should_merge = True

        if should_merge:
            # Extend current segment
            current_end_idx = max(current_end_idx, next_end_idx)
            current_end_time = max(current_end_time, next_end_time)
        else:
            # Finalize current, start new
            merged.append((current_start_idx, current_end_idx, current_start_time, current_end_time))
            current_start_idx, current_end_idx = next_start_idx, next_end_idx
            current_start_time, current_end_time = next_start_time, next_end_time

    merged.append((current_start_idx, current_end_idx, current_start_time, current_end_time))
    return merged


def detect_stops(records: list[Record]) -> list[StopSegment]:
    """
    Find all stop segments ≥ MIN_STOP_DURATION_S with comprehensive gap handling.

    Algorithm (v2.1 – Wall-Clock Gap Detection + Segment Debouncing)
    -----------------------------------------------------------------
    1. **Wall-Clock Gap Detection**: Scan for large time gaps (> 300s) between
       ANY consecutive FIT records.  These are treated as implicit stops even
       if no "stopped" rows were recorded (handles Auto-Pause completely).

    2. **Speed-Based Detection**: Collect all record indices where
       speed < STOP_SPEED_THRESHOLD_KMH and group consecutive stopped rows.

    3. **Segment Debouncing**: Merge segments if the "moving" gap between them
       has ≤ MAX_AUTO_PAUSE_GAP_SEC wall-clock time AND the intermediate rows
       are mostly drift-level speeds (< 3.5 km/h).  This prevents GPS drift
       from fracturing one long stop into multiple short ones.

    4. **Wall-Clock Duration**: Duration = (end_time - start_time), so any
       unrecorded Auto-Pause interval is naturally included.

    5. **GPS Blackout Rejection**: Discard apparent stops where odometer shows
       > 6 km/h average motion (GPS blackout while riding, not a real stop).

    Parameters are read from config.settings at import time:
        STOP_SPEED_THRESHOLD_KMH, MIN_STOP_DURATION_S, MAX_AUTO_PAUSE_GAP_SEC
    """
    if not records:
        return []

    raw_segments: list[tuple[int, int, datetime, datetime]] = []

    # ── 1. Wall-Clock Gap Detection (implicit stops from Auto-Pause gaps) ────
    implicit_stops = _find_implicit_stops_from_gaps(records)
    for row_before, row_after, gap_s in implicit_stops:
        # Use the row BEFORE the gap as the anchor (stable GPS location)
        # The stop spans from that row's timestamp to the row AFTER the gap
        start_time = records[row_before].timestamp
        end_time = records[row_after].timestamp
        if gap_s >= MIN_STOP_DURATION_S:
            raw_segments.append((row_before, row_after, start_time, end_time))

    # ── 2. Speed-Based Detection (explicit stopped rows) ─────────────────────
    stopped_indices: list[int] = [
        i for i, r in enumerate(records)
        if _speed_ms_to_kmh(r.speed_ms) < STOP_SPEED_THRESHOLD_KMH
    ]

    if stopped_indices:
        # Group by gap tolerance using cumsum trick
        stopped_ts = pd.Series(
            pd.to_datetime(
                [records[i].timestamp for i in stopped_indices],
                errors="coerce",
            )
        )
        gap_seconds = stopped_ts.diff().dt.total_seconds().fillna(0.0)
        # Only break into new group if gap > MAX_AUTO_PAUSE_GAP_SEC
        group_ids = (gap_seconds > MAX_AUTO_PAUSE_GAP_SEC).cumsum().values
        idx_arr = np.array(stopped_indices)

        for gid in np.unique(group_ids):
            grp_record_indices = idx_arr[group_ids == gid]
            start_idx = int(grp_record_indices[0])
            end_idx = int(grp_record_indices[-1])
            start_time = records[start_idx].timestamp
            end_time = records[end_idx].timestamp
            raw_segments.append((start_idx, end_idx, start_time, end_time))

    if not raw_segments:
        return []

    # ── 3. Segment Debouncing: merge nearby segments (GPS drift tolerance) ───
    merged_segments = _merge_overlapping_segments(
        raw_segments, records, float(MAX_AUTO_PAUSE_GAP_SEC)
    )

    # ── 4. Build StopSegments with duration filter & GPS blackout rejection ──
    stops: list[StopSegment] = []
    for start_idx, end_idx, start_time, end_time in merged_segments:
        duration_s = (end_time - start_time).total_seconds()

        if duration_s < MIN_STOP_DURATION_S:
            continue

        # GPS blackout rejection: average speed > 6 km/h = sensor blackout while moving
        dist_start = records[start_idx].distance_m
        dist_end = records[end_idx].distance_m
        if (
            dist_start is not None
            and dist_end is not None
            and not pd.isna(dist_start)
            and not pd.isna(dist_end)
            and duration_s > 0
        ):
            gap_speed_kmh = ((dist_end - dist_start) / 1000.0) / (duration_s / 3600.0)
            if gap_speed_kmh > 6.0:
                log.debug(
                    "GPS blackout discarded: %.0f m in %.1f min = %.1f km/h (%s – %s)",
                    dist_end - dist_start,
                    duration_s / 60.0,
                    gap_speed_kmh,
                    start_time,
                    end_time,
                )
                continue

        stops.append(
            StopSegment(
                start_idx=start_idx,
                end_idx=end_idx,
                duration_s=duration_s,
                start_time=start_time,
                end_time=end_time,
            )
        )

    # Remove duplicates (same start_time) and sort
    seen_start_times: set[datetime] = set()
    unique_stops: list[StopSegment] = []
    for stop in sorted(stops, key=lambda s: (s.start_time, -s.duration_s)):
        if stop.start_time not in seen_start_times:
            unique_stops.append(stop)
            seen_start_times.add(stop.start_time)

    return unique_stops
