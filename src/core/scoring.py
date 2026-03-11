"""
src/core/scoring.py – Single Source of Truth for Confidence Scoring
=====================================================================
Senior Sports Data Scientist · Physiologist · Python Expert

Unified physiological scoring logic for the botanical analytics pipeline.
Imported by BOTH detect_botanical_stops.py (reporter) AND
detect_botanical_hotspots.py (mapper) to guarantee byte-identical scores.

Markers:
  A) Resting tachycardia during stop (HR trend inside pause window)
  B) Cardiovascular decoupling (HR/Speed ratio pre vs. post stop)
  C) Stoner Pace – cadence drop + coasting increase post stop
  D) Respiration rate increase post stop  (graceful: returns 0 if absent)

Terrain Gradient Penalty  (global, pipeline-wide):
  If the post-stop analysis window has a mean uphill gradient > 1.5 %,
  the HR spike is mechanically explained by the climb → confidence × 0.20
  (i.e. an 80 % slash).  This rule is enforced here ONCE so both scripts
  can never diverge again.

Public API:
    compute_confidence(records, stop) -> tuple[float, dict]

    • confidence (float) – final score clamped to [0 … 100]
    • details   (dict)  – per-marker scores + descriptions + gradient info
                          (plug directly into the reporter's CSV columns)
"""

from __future__ import annotations

from datetime import timedelta
from statistics import mean
from typing import Optional

import pandas as pd

from config.settings import ANALYSIS_WINDOW_S, HR_SETTLE_WINDOW_S
from src.core.stop_analysis import Record, StopSegment, _speed_ms_to_kmh

# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT PENALTY CONSTANTS  (single definition for the whole pipeline)
# ─────────────────────────────────────────────────────────────────────────────

GRADIENT_PENALTY_THRESHOLD_PCT: float = 1.5   # any post-stop slope above this …
GRADIENT_PENALTY_FACTOR:        float = 0.20  # … keeps only 20 % of confidence


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL WINDOW HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_window_records(
    records: list[Record],
    center_idx: int,
    window_s: float,
    direction: str,  # "before" | "after"
) -> list[Record]:
    """Return records within *window_s* seconds before or after *center_idx*."""
    ref_time = records[center_idx].timestamp
    result: list[Record] = []

    if direction == "before":
        for k in range(center_idx - 1, -1, -1):
            if (ref_time - records[k].timestamp).total_seconds() > window_s:
                break
            result.append(records[k])
        result.reverse()
    else:  # "after"
        for k in range(center_idx + 1, len(records)):
            if (records[k].timestamp - ref_time).total_seconds() > window_s:
                break
            result.append(records[k])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MARKER A  –  Resting Tachycardia in the Pause
# ─────────────────────────────────────────────────────────────────────────────

def marker_a_resting_tachycardia(
    records: list[Record], stop: StopSegment
) -> tuple[float, str]:
    """
    Compare mean HR in the first HR_SETTLE_WINDOW_S seconds of the stop
    vs. the last HR_SETTLE_WINDOW_S seconds.

    Normal recovery → HR falls.  Anomaly → HR stays flat or rises.

    Returns (score 0-100, description_string).
    """
    stop_records = records[stop.start_idx : stop.end_idx + 1]
    if not stop_records:
        return 0.0, "N/A"

    first_cutoff = stop.start_time + timedelta(seconds=HR_SETTLE_WINDOW_S)
    last_cutoff  = stop.end_time   - timedelta(seconds=HR_SETTLE_WINDOW_S)

    first_hrs = [
        r.heart_rate for r in stop_records
        if r.timestamp <= first_cutoff and r.heart_rate is not None
        and not pd.isna(r.heart_rate)
    ]
    last_hrs = [
        r.heart_rate for r in stop_records
        if r.timestamp >= last_cutoff and r.heart_rate is not None
        and not pd.isna(r.heart_rate)
    ]

    if not first_hrs or not last_hrs:
        return 0.0, "Insufficient HR data"

    avg_first = mean(first_hrs)
    avg_last  = mean(last_hrs)
    delta = avg_last - avg_first

    if pd.isna(delta):
        return 0.0, "Insufficient HR data"

    desc = f"Start: {avg_first:.0f} bpm → End: {avg_last:.0f} bpm (Δ{delta:+.0f})"

    if delta <= 0:   return 0.0,   desc
    if delta < 5:    return 15.0,  desc
    if delta < 10:   return 35.0,  desc
    if delta < 15:   return 55.0,  desc
    if delta < 20:   return 70.0,  desc
    if delta < 30:   return 85.0,  desc
    return 100.0, desc


# ─────────────────────────────────────────────────────────────────────────────
# MARKER B  –  Cardiovascular Decoupling (HR / Speed ratio)
# ─────────────────────────────────────────────────────────────────────────────

def marker_b_decoupling(
    records: list[Record], stop: StopSegment
) -> tuple[float, str]:
    """
    Compare mean HR/Speed(km/h) over the ANALYSIS_WINDOW_S before the stop
    vs. after the stop.  A spike in this ratio post-stop indicates decoupling.

    Returns (score 0-100, description_string "±X.X %").
    """
    before = get_window_records(records, stop.start_idx, ANALYSIS_WINDOW_S, "before")
    after  = get_window_records(records, stop.end_idx,   ANALYSIS_WINDOW_S, "after")

    def _ratio(recs: list[Record]) -> Optional[float]:
        vals = []
        for r in recs:
            s = _speed_ms_to_kmh(r.speed_ms)
            if r.heart_rate is not None and not pd.isna(r.heart_rate) and s > 5.0:
                vals.append(r.heart_rate / s)
        return mean(vals) if vals else None

    rb = _ratio(before)
    ra = _ratio(after)

    if rb is None or ra is None or rb == 0:
        return 0.0, "Insufficient data"

    pct = ((ra - rb) / rb) * 100.0
    if pd.isna(pct):
        return 0.0, "Insufficient data"

    desc = f"{pct:+.1f} %"

    if pct <= 3:   return 0.0,   desc
    if pct <= 8:   return 20.0,  desc
    if pct <= 15:  return 45.0,  desc
    if pct <= 25:  return 70.0,  desc
    if pct <= 40:  return 85.0,  desc
    return 100.0, desc


# ─────────────────────────────────────────────────────────────────────────────
# MARKER C  –  Stoner Pace (Cadence Drop + Coasting)
# ─────────────────────────────────────────────────────────────────────────────

def marker_c_cadence_drop(
    records: list[Record], stop: StopSegment
) -> tuple[float, str]:
    """
    Compare average cadence and coasting percentage before vs. after stop.
    A drop in cadence combined with increased coasting is anomalous.

    Returns (score 0-100, description_string "Δrpm (coast Δ%)").
    Gracefully returns (0, 'No cadence data') when sensor is absent.
    """
    before = get_window_records(records, stop.start_idx, ANALYSIS_WINDOW_S, "before")
    after  = get_window_records(records, stop.end_idx,   ANALYSIS_WINDOW_S, "after")

    def _avg_cadence(recs: list[Record]) -> Optional[float]:
        vals = [
            r.cadence for r in recs
            if r.cadence is not None and not pd.isna(r.cadence)
            and _speed_ms_to_kmh(r.speed_ms) > 5.0
        ]
        return mean(vals) if vals else None

    def _coasting_pct(recs: list[Record]) -> float:
        """Percentage of moving records where cadence is zero or missing."""
        moving = [r for r in recs if _speed_ms_to_kmh(r.speed_ms) > 5.0]
        if not moving:
            return 0.0
        coasting = sum(1 for r in moving if r.cadence is None or r.cadence < 2)
        return (coasting / len(moving)) * 100.0

    cad_before = _avg_cadence(before)
    cad_after  = _avg_cadence(after)

    if cad_before is None or cad_after is None:
        return 0.0, "No cadence data"

    delta = cad_after - cad_before  # negative means cadence fell
    if pd.isna(delta):
        return 0.0, "No cadence data"

    coast_before = _coasting_pct(before)
    coast_after  = _coasting_pct(after)
    coast_delta  = coast_after - coast_before

    desc = f"{delta:+.0f} rpm (coast {coast_delta:+.0f}%)"

    drop = -delta  # positive value = cadence dropped
    if drop <= 2:    score = 0.0
    elif drop <= 5:  score = 20.0
    elif drop <= 8:  score = 40.0
    elif drop <= 12: score = 60.0
    elif drop <= 18: score = 80.0
    else:            score = 100.0

    # Coasting bonus
    if coast_delta > 10:
        score = min(100.0, score + 15.0)
    elif coast_delta > 5:
        score = min(100.0, score + 8.0)

    return score, desc


# ─────────────────────────────────────────────────────────────────────────────
# MARKER D  –  Respiration Rate Increase
# ─────────────────────────────────────────────────────────────────────────────

def marker_d_respiration(
    records: list[Record], stop: StopSegment
) -> tuple[float, str]:
    """
    Compare mean respiratory rate before vs. after stop.
    An unexpected rise post-stop is physiologically anomalous.

    Returns (score 0-100, description_string).
    Gracefully returns (0, 'No respiration data') when sensor is absent.
    """
    before = get_window_records(records, stop.start_idx, ANALYSIS_WINDOW_S, "before")
    after  = get_window_records(records, stop.end_idx,   ANALYSIS_WINDOW_S, "after")

    def _avg_resp(recs: list[Record]) -> Optional[float]:
        vals = [
            r.respiratory_rate for r in recs
            if r.respiratory_rate is not None
            and not pd.isna(r.respiratory_rate)
            and r.respiratory_rate > 0
        ]
        return mean(vals) if vals else None

    resp_before = _avg_resp(before)
    resp_after  = _avg_resp(after)

    if resp_before is None or resp_after is None:
        return 0.0, "No respiration data"

    delta = resp_after - resp_before
    if pd.isna(delta):
        return 0.0, "No respiration data"

    desc = f"{resp_before:.1f} → {resp_after:.1f} brpm (Δ{delta:+.1f})"

    if delta <= 1.0:  return 0.0,   desc
    if delta <= 3.0:  return 25.0,  desc
    if delta <= 5.0:  return 50.0,  desc
    if delta <= 8.0:  return 75.0,  desc
    return 100.0, desc


# ─────────────────────────────────────────────────────────────────────────────
# TERRAIN GRADIENT PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def calculate_post_stop_gradient(records: list[Record]) -> tuple[float, float, float]:
    """
    Derive the uphill gradient percentage for a post-stop analysis window.

    Filters GPS micro-noise by ignoring altitude steps ≤ 0.5 m.
    Requires at least 10 m of horizontal distance to return a meaningful value.

    Returns:
        (gradient_pct, ascent_m, distance_m)
        gradient_pct — average slope as a percentage (ascent / distance × 100)
        ascent_m     — cumulative elevation gain (filtered)
        distance_m   — total horizontal distance covered in the window
    """
    if not records or len(records) < 2:
        return 0.0, 0.0, 0.0

    altitudes: list[float] = []
    distances: list[float] = []
    for r in records:
        if r.altitude_m is not None and r.distance_m is not None:
            if not (pd.isna(r.altitude_m) or pd.isna(r.distance_m)):
                altitudes.append(r.altitude_m)
                distances.append(r.distance_m)

    if len(altitudes) < 2 or len(distances) < 2:
        return 0.0, 0.0, 0.0

    alt_series  = pd.Series(altitudes)
    dist_series = pd.Series(distances)

    alt_diff = alt_series.diff()
    ascent_m = alt_diff[alt_diff > 0.5].sum()
    if pd.isna(ascent_m):
        ascent_m = 0.0

    window_dist_m = dist_series.max() - dist_series.min()

    gradient_pct = (ascent_m / window_dist_m) * 100.0 if window_dist_m > 10.0 else 0.0

    return round(gradient_pct, 2), round(ascent_m, 1), round(window_dist_m, 1)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE CONFIDENCE SCORE  –  Single Source of Truth
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(
    records: list[Record],
    stop: StopSegment,
) -> tuple[float, dict]:
    """
    Compute the final weighted Confidence Score for one stop.

    Weights (graceful degradation – missing markers redistributed):
        A: 35 %  B: 30 %  C: 20 %  D: 15 %

    Terrain Gradient Penalty (global, pipeline-wide):
        post-stop gradient > 1.5 %  →  confidence × 0.20  (80 % slash)
        because the HR / cadence response is mechanically caused by the climb.

    Args:
        records : full list of Record objects for the activity
        stop    : StopSegment describing the pause boundaries

    Returns:
        confidence (float) – final score clamped to [0, 100]
        details    (dict)  – individual marker results + gradient metadata,
                             ready to unpack directly into CSV report columns:
                               score_a / desc_a
                               score_b / desc_b
                               score_c / desc_c
                               score_d / desc_d
                               raw_confidence
                               gradient_pct
                               gradient_desc
                               gradient_penalty_applied
    """
    # ── Individual markers ────────────────────────────────────────────────────
    score_a, desc_a = marker_a_resting_tachycardia(records, stop)
    score_b, desc_b = marker_b_decoupling(records, stop)
    score_c, desc_c = marker_c_cadence_drop(records, stop)
    score_d, desc_d = marker_d_respiration(records, stop)

    # Graceful degradation: only include markers with real data
    has_cadence     = desc_c != "No cadence data"
    has_respiration = desc_d != "No respiration data"

    weights: dict[str, float] = {"A": 35.0, "B": 30.0}
    scores:  dict[str, float] = {"A": score_a, "B": score_b}

    if has_cadence:
        weights["C"] = 20.0
        scores["C"]  = score_c

    if has_respiration:
        weights["D"] = 15.0
        scores["D"]  = score_d

    total_w = sum(weights.values())
    raw_confidence = (
        sum(scores[k] * (weights[k] / total_w) for k in weights)
        if total_w > 0 else 0.0
    )

    # ── Terrain Gradient Penalty ──────────────────────────────────────────────
    post_records = get_window_records(
        records, stop.end_idx, ANALYSIS_WINDOW_S, "after"
    )
    gradient_pct, ascent_m, window_dist_m = calculate_post_stop_gradient(post_records)

    gradient_penalty_applied = gradient_pct > GRADIENT_PENALTY_THRESHOLD_PCT
    confidence = (
        raw_confidence * GRADIENT_PENALTY_FACTOR
        if gradient_penalty_applied
        else raw_confidence
    )
    confidence = max(0.0, min(100.0, confidence))

    # ── Gradient description ──────────────────────────────────────────────────
    if gradient_pct > 0:
        gradient_desc = f"{gradient_pct:.1f}% ({ascent_m:.0f}m / {window_dist_m:.0f}m)"
    else:
        gradient_desc = "Flat / No data"
    if gradient_penalty_applied:
        gradient_desc += " [PENALTY]"

    details: dict = {
        "score_a":                  score_a,
        "desc_a":                   desc_a,
        "score_b":                  score_b,
        "desc_b":                   desc_b,
        "score_c":                  score_c,
        "desc_c":                   desc_c,
        "score_d":                  score_d,
        "desc_d":                   desc_d,
        "raw_confidence":           round(raw_confidence, 1),
        "gradient_pct":             gradient_pct,
        "gradient_desc":            gradient_desc,
        "gradient_penalty_applied": gradient_penalty_applied,
    }

    return round(confidence, 1), details
