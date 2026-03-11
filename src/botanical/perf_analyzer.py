#!/usr/bin/env python3
"""
botanical_performance_split.py – Athletic Performance Before/After Botanical Stops
===================================================================================
STRICTLY ISOLATED SUB-PROJECT (v2.0 – Advanced Metrics)

Analyzes athletic performance metrics BEFORE and AFTER botanical stops
(green stops) to identify physiological changes around these events.

This script does NOT modify any existing core files. It imports advanced
algorithms from athlete_analytics.py for proper physiological calculations.

Inputs
------
* data/processed/green_stops_report.csv   – botanical stop events
* data/summaries/master_high_res_training_data.csv – high-res training data

Output
------
* data/processed/botanical_performance_comparison.csv

Metrics Calculated (per segment):
---------------------------------
- Duration, Distance, Avg Speed, Avg HR
- Avg Power (cycling only)
- Efficiency Factor (Power/HR for cycling, Speed/HR for running)
- TRIMP & TRIMP/hr
- Cardiac Drift (%) – imported from athlete_analytics
- Max 60s HR Drop (bpm) – imported from athlete_analytics

Author: Senior Python Data Engineer
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import (
    MAX_HR as ATHLETE_MAX_HR,
    RESTING_HR as ATHLETE_RHR,
    TRIMP_K1,
    TRIMP_K2,
    PROCESSED_DIR,
    SUMMARIES_DIR,
)

# ── Import advanced algorithms from athlete_analytics ─────────────────────
from src.analytics.athlete_analytics import (
    cardiac_drift_for_activity,
    max_hrr_60s_for_activity,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GREEN_STOPS_CSV = os.path.join(PROCESSED_DIR, "green_stops_report.csv")
TRAINING_CSV = os.path.join(SUMMARIES_DIR, "master_high_res_training_data.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "botanical_performance_comparison.csv")

# Minimum confidence threshold for stop inclusion
MIN_CONFIDENCE_SCORE = 15.0

# Chunk size for memory-efficient CSV reading
CHUNK_SIZE = 500_000


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def load_green_stops(min_confidence: float = MIN_CONFIDENCE_SCORE) -> pd.DataFrame:
    """
    Load green stops report and filter by confidence score.
    
    Returns DataFrame with columns:
        Activity ID, Stop Time (datetime), Duration (min), etc.
    """
    if not os.path.exists(GREEN_STOPS_CSV):
        log.error(f"Green stops file not found: {GREEN_STOPS_CSV}")
        return pd.DataFrame()
    
    df = pd.read_csv(GREEN_STOPS_CSV, low_memory=False)
    log.info(f"Loaded {len(df)} green stops from {GREEN_STOPS_CSV}")
    
    # Filter by confidence score
    df["Confidence Score (%)"] = pd.to_numeric(df["Confidence Score (%)"], errors="coerce")
    df = df[df["Confidence Score (%)"] >= min_confidence].copy()
    log.info(f"After confidence filter (>= {min_confidence}%): {len(df)} stops")
    
    # Convert Activity ID to string for consistent matching
    df["Activity ID"] = df["Activity ID"].astype(str)
    
    # Parse Date and Stop Time to create full datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Duration (min)"] = pd.to_numeric(df["Duration (min)"], errors="coerce").fillna(0)
    
    # Parse Stop Time (format: HH:MM:SS) and combine with Date
    def parse_stop_datetime(row) -> Optional[datetime]:
        try:
            date = row["Date"]
            stop_time_str = str(row["Stop Time"])
            time_parts = stop_time_str.split(":")
            if len(time_parts) == 3:
                h, m, s = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
                return datetime(
                    date.year, date.month, date.day,
                    h, m, int(s), int((s % 1) * 1_000_000)
                )
        except Exception:
            pass
        return None
    
    df["stop_start_dt"] = df.apply(parse_stop_datetime, axis=1)
    df["stop_end_dt"] = df.apply(
        lambda r: r["stop_start_dt"] + timedelta(minutes=r["Duration (min)"])
        if pd.notna(r["stop_start_dt"]) and pd.notna(r["Duration (min)"])
        else None,
        axis=1
    )
    
    # Drop rows where datetime parsing failed
    df = df.dropna(subset=["stop_start_dt", "stop_end_dt"])
    log.info(f"After datetime parsing: {len(df)} valid stops")
    
    return df


def load_training_data_bulk(activity_ids: set) -> Dict[str, pd.DataFrame]:
    """
    Single-pass chunked read of TRAINING_CSV for a set of activity IDs.
    
    Returns dict {activity_id: DataFrame} – memory efficient for large files.
    """
    if not os.path.exists(TRAINING_CSV):
        log.error(f"Training data file not found: {TRAINING_CSV}")
        return {}
    
    frames_by_id: Dict[str, List[pd.DataFrame]] = {aid: [] for aid in activity_ids}
    total_rows = 0
    
    log.info(f"Loading training data for {len(activity_ids)} activities (chunked)...")
    
    for chunk in pd.read_csv(TRAINING_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk.assign(activity_id=chunk["activity_id"].astype(str))
        filtered = chunk.loc[chunk["activity_id"].isin(activity_ids)]
        if filtered.empty:
            continue
        total_rows += len(filtered)
        for aid, grp in filtered.groupby("activity_id"):
            frames_by_id[str(aid)].append(grp)
    
    # Concatenate frames per activity
    result = {}
    for aid, frames in frames_by_id.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df.assign(timestamp=pd.to_datetime(df["timestamp"], errors="coerce"))
        df = df.sort_values("timestamp").reset_index(drop=True)
        result[aid] = df
    
    log.info(f"Loaded {total_rows} rows for {len(result)} activities")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS FUNCTIONS (Using athlete_analytics.py algorithms)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_segment_metrics(
    df: pd.DataFrame,
    sport: str,
    segment_name: str = "segment"
) -> Dict[str, float]:
    """
    Calculate advanced performance metrics for a DataFrame segment.
    
    Metrics calculated:
        - Duration (min)
        - Distance (km)
        - Avg Speed (km/h)
        - Avg HR (bpm)
        - Avg Power (W) – only for cycling
        - Efficiency Factor:
            * Cycling: Avg Power / Avg HR
            * Running: Avg Speed (m/s) / Avg HR
        - TRIMP (Banister formula)
        - TRIMP/hr (normalized for comparability)
        - Cardiac Drift (%) – using athlete_analytics algorithm
        - Max 60s HR Drop (bpm) – using athlete_analytics algorithm
    
    Args:
        df: High-res training data for the segment
        sport: Sport type string (e.g., 'running', 'cycling', 'road_bike')
        segment_name: Prefix for metric names ('before' or 'after')
    
    Returns dict of metrics.
    """
    result = {
        f"{segment_name}_duration_min": np.nan,
        f"{segment_name}_distance_km": np.nan,
        f"{segment_name}_avg_speed_kmh": np.nan,
        f"{segment_name}_avg_hr_bpm": np.nan,
        f"{segment_name}_avg_power_w": np.nan,
        f"{segment_name}_efficiency_factor": np.nan,
        f"{segment_name}_trimp": np.nan,
        f"{segment_name}_trimp_per_hr": np.nan,
        f"{segment_name}_cardiac_drift_pct": np.nan,
        f"{segment_name}_max_hrr_60s": np.nan,
        # HR Zone distribution (normalized percentages and absolute minutes)
        f"{segment_name}_Z1_pct": 0.0,
        f"{segment_name}_Z2_pct": 0.0,
        f"{segment_name}_Z3_pct": 0.0,
        f"{segment_name}_Z4_pct": 0.0,
        f"{segment_name}_Z5_pct": 0.0,
        f"{segment_name}_Z1_min": 0.0,
        f"{segment_name}_Z2_min": 0.0,
        f"{segment_name}_Z3_min": 0.0,
        f"{segment_name}_Z4_min": 0.0,
        f"{segment_name}_Z5_min": 0.0,
    }
    
    if df.empty or len(df) < 2:
        return result
    
    # ── Strict active filter (exclude auto-pauses and inactive records) ───────
    if "is_active" in df.columns:
        df = df[df["is_active"] == True].copy()
    
    if df.empty or len(df) < 2:
        return result
    
    # Normalize sport string
    sport_lower = str(sport).lower() if sport else ""
    is_cycling = "cycling" in sport_lower or "bike" in sport_lower
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.assign(timestamp=pd.to_datetime(df["timestamp"], errors="coerce"))
    
    df = df.dropna(subset=["timestamp"])
    if df.empty or len(df) < 2:
        return result
    
    # ── True Moving Time (TMT) ────────────────────────────────────────────────
    # Calculate actual moving time by summing inter-sample gaps < 30s
    # This filters out pauses and GPS dropouts
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_diffs = df["timestamp"].diff().dt.total_seconds().fillna(1.0)
    # Only sum intervals < 30 seconds (exclude pauses)
    moving_seconds = time_diffs[time_diffs < 30].sum()
    duration_min = moving_seconds / 60.0
    result[f"{segment_name}_duration_min"] = round(duration_min, 2)
    
    if duration_min <= 0:
        return result
    
    # ── Distance (km) ─────────────────────────────────────────────────────────
    if "distance" in df.columns:
        dist_series = pd.to_numeric(df["distance"], errors="coerce")
        dist_valid = dist_series.dropna()
        if len(dist_valid) > 0:
            distance_m = dist_valid.max() - dist_valid.min()
            distance_km = distance_m / 1000.0
            result[f"{segment_name}_distance_km"] = round(distance_km, 3)
    
    # ── Average Speed (km/h) ──────────────────────────────────────────────────
    avg_speed_ms = np.nan
    if "speed" in df.columns:
        speed_series = pd.to_numeric(df["speed"], errors="coerce")
        speed_valid = speed_series[speed_series > 0]
        if len(speed_valid) > 0:
            avg_speed_ms = speed_valid.mean()
            avg_speed_kmh = avg_speed_ms * 3.6
            result[f"{segment_name}_avg_speed_kmh"] = round(avg_speed_kmh, 2)
    
    # ── Average HR (bpm) ──────────────────────────────────────────────────────
    avg_hr = np.nan
    if "heart_rate" in df.columns:
        hr_series = pd.to_numeric(df["heart_rate"], errors="coerce")
        hr_valid = hr_series[(hr_series > 30) & (hr_series < 250)]
        if len(hr_valid) > 0:
            avg_hr = hr_valid.mean()
            result[f"{segment_name}_avg_hr_bpm"] = round(avg_hr, 1)
    
    # ── Average Power (W) – cycling only ──────────────────────────────────────
    avg_power = np.nan
    if is_cycling and "power" in df.columns:
        power_series = pd.to_numeric(df["power"], errors="coerce")
        power_valid = power_series[power_series > 0]
        if len(power_valid) > 0:
            avg_power = power_valid.mean()
            result[f"{segment_name}_avg_power_w"] = round(avg_power, 1)
    
    # ── Efficiency Factor ─────────────────────────────────────────────────────
    # Cycling: EF = Avg Power / Avg HR
    # Running: EF = Avg Speed (m/s) / Avg HR
    if not np.isnan(avg_hr) and avg_hr > 0:
        if is_cycling and not np.isnan(avg_power):
            ef = avg_power / avg_hr
            result[f"{segment_name}_efficiency_factor"] = round(ef, 4)
        elif not is_cycling and not np.isnan(avg_speed_ms):
            ef = avg_speed_ms / avg_hr
            result[f"{segment_name}_efficiency_factor"] = round(ef, 5)
    
    # ── TRIMP calculation (Banister formula) ──────────────────────────────────
    # TRIMP = duration_min * HR_ratio * k1 * exp(k2 * HR_ratio)
    # HR_ratio = (HR - RHR) / (HRmax - RHR)
    if not np.isnan(avg_hr) and ATHLETE_MAX_HR > ATHLETE_RHR:
        hrr = ATHLETE_MAX_HR - ATHLETE_RHR  # Heart Rate Reserve
        hr_ratio = max(0, (avg_hr - ATHLETE_RHR) / hrr)
        hr_ratio = min(hr_ratio, 1.0)  # Cap at 1.0
        
        # Banister TRIMP formula (male coefficients from config)
        trimp = duration_min * hr_ratio * TRIMP_K1 * np.exp(TRIMP_K2 * hr_ratio)
        result[f"{segment_name}_trimp"] = round(trimp, 2)
        
        # TRIMP per hour (normalized for duration comparability)
        if duration_min > 0:
            trimp_per_hr = trimp / (duration_min / 60.0)
            result[f"{segment_name}_trimp_per_hr"] = round(trimp_per_hr, 2)
    
    # ── Cardiac Drift (%) – using athlete_analytics algorithm ────────────────
    try:
        cardiac_drift = cardiac_drift_for_activity(df, sport_lower)
        if cardiac_drift is not None:
            result[f"{segment_name}_cardiac_drift_pct"] = round(cardiac_drift, 2)
    except Exception as e:
        log.debug(f"Cardiac drift calculation failed: {e}")
    
    # ── Max 60s HR Drop (bpm) – using athlete_analytics algorithm ─────────────
    try:
        max_hrr = max_hrr_60s_for_activity(df)
        if max_hrr is not None:
            result[f"{segment_name}_max_hrr_60s"] = round(max_hrr, 1)
    except Exception as e:
        log.debug(f"Max HRR 60s calculation failed: {e}")
    
    # ── HR Zone Distribution (normalized % and absolute minutes) ─────────────
    # Percentage of time in each zone – statistically valid for cross-segment comparison
    # Absolute minutes – useful for understanding total load
    if "hr_zone" in df.columns:
        zone_counts = df["hr_zone"].value_counts(normalize=True) * 100
        for zone in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
            zone_pct = 0.0
            if zone in zone_counts.index:
                zone_pct = zone_counts[zone]
                result[f"{segment_name}_{zone}_pct"] = round(zone_pct, 2)
            # Calculate absolute time in zone: duration * (pct / 100)
            zone_min = duration_min * (zone_pct / 100.0)
            result[f"{segment_name}_{zone}_min"] = round(zone_min, 2)
    
    return result


def split_activity_by_stop(
    activity_df: pd.DataFrame,
    stop_start: datetime,
    stop_end: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split activity data into BEFORE and AFTER the botanical stop.
    
    Args:
        activity_df: High-res training data for the activity
        stop_start: Start datetime of the botanical stop
        stop_end: End datetime of the botanical stop
    
    Returns:
        Tuple of (df_before, df_after)
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(activity_df["timestamp"]):
        activity_df = activity_df.assign(
            timestamp=pd.to_datetime(activity_df["timestamp"], errors="coerce")
        )
    
    # Filter: before stop start, after stop end
    df_before = activity_df[activity_df["timestamp"] < stop_start].copy()
    df_after = activity_df[activity_df["timestamp"] > stop_end].copy()
    
    return df_before, df_after


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def process_botanical_performance() -> pd.DataFrame:
    """
    Main processing function.
    
    1. Load green stops with confidence >= 15%
    2. Load high-res training data for relevant activities
    3. Split each activity by stop timing
    4. Calculate before/after metrics
    5. Calculate deltas
    6. Save results
    """
    log.info("=" * 70)
    log.info("BOTANICAL PERFORMANCE SPLIT ANALYSIS")
    log.info("=" * 70)
    
    # 1. Load green stops
    stops_df = load_green_stops(MIN_CONFIDENCE_SCORE)
    if stops_df.empty:
        log.warning("No valid green stops found. Exiting.")
        return pd.DataFrame()
    
    # 2. Get unique activity IDs
    activity_ids = set(stops_df["Activity ID"].unique())
    log.info(f"Processing {len(activity_ids)} unique activities with botanical stops")
    
    # 3. Load training data for these activities
    training_data = load_training_data_bulk(activity_ids)
    if not training_data:
        log.warning("No training data loaded. Exiting.")
        return pd.DataFrame()
    
    # 4. Process each stop
    results = []
    
    for idx, stop_row in stops_df.iterrows():
        activity_id = str(stop_row["Activity ID"])
        
        if activity_id not in training_data:
            log.debug(f"Activity {activity_id} not found in training data, skipping")
            continue
        
        activity_df = training_data[activity_id]
        stop_start = stop_row["stop_start_dt"]
        stop_end = stop_row["stop_end_dt"]
        
        # Extract sport from the activity data
        sport = ""
        if "sport" in activity_df.columns and len(activity_df) > 0:
            sport = str(activity_df["sport"].iloc[0]) if pd.notna(activity_df["sport"].iloc[0]) else ""
        
        # Split activity
        df_before, df_after = split_activity_by_stop(activity_df, stop_start, stop_end)
        
        # Check minimum data requirements
        if len(df_before) < 5 or len(df_after) < 5:
            log.debug(
                f"Activity {activity_id}: insufficient data "
                f"(before={len(df_before)}, after={len(df_after)}), skipping"
            )
            continue
        
        # Calculate metrics with sport parameter
        before_metrics = calculate_segment_metrics(df_before, sport, "before")
        after_metrics = calculate_segment_metrics(df_after, sport, "after")
        
        # Build result row
        row = {
            "activity_id": activity_id,
            "date": stop_row["Date"].strftime("%Y-%m-%d") if pd.notna(stop_row["Date"]) else "",
            "sport": sport,
            "stop_start": stop_start.strftime("%Y-%m-%d %H:%M:%S"),
            "stop_duration_min": stop_row["Duration (min)"],
            "confidence_score": stop_row["Confidence Score (%)"],
        }
        
        # Add before/after metrics
        row.update(before_metrics)
        row.update(after_metrics)
        
        # ── Calculate deltas (After - Before) ─────────────────────────────────
        
        # Speed delta (km/h)
        if not np.isnan(after_metrics.get("after_avg_speed_kmh", np.nan)) and \
           not np.isnan(before_metrics.get("before_avg_speed_kmh", np.nan)):
            row["delta_speed_kmh"] = round(
                after_metrics["after_avg_speed_kmh"] - before_metrics["before_avg_speed_kmh"], 2
            )
        else:
            row["delta_speed_kmh"] = np.nan
        
        # HR delta (bpm)
        if not np.isnan(after_metrics.get("after_avg_hr_bpm", np.nan)) and \
           not np.isnan(before_metrics.get("before_avg_hr_bpm", np.nan)):
            row["delta_hr_bpm"] = round(
                after_metrics["after_avg_hr_bpm"] - before_metrics["before_avg_hr_bpm"], 1
            )
        else:
            row["delta_hr_bpm"] = np.nan
        
        # Power delta (W) – cycling only
        if not np.isnan(after_metrics.get("after_avg_power_w", np.nan)) and \
           not np.isnan(before_metrics.get("before_avg_power_w", np.nan)):
            row["delta_power_w"] = round(
                after_metrics["after_avg_power_w"] - before_metrics["before_avg_power_w"], 1
            )
        else:
            row["delta_power_w"] = np.nan
        
        # Efficiency Factor delta
        if not np.isnan(after_metrics.get("after_efficiency_factor", np.nan)) and \
           not np.isnan(before_metrics.get("before_efficiency_factor", np.nan)):
            row["delta_efficiency_factor"] = round(
                after_metrics["after_efficiency_factor"] - before_metrics["before_efficiency_factor"], 5
            )
        else:
            row["delta_efficiency_factor"] = np.nan
        
        # TRIMP/hr delta
        if not np.isnan(after_metrics.get("after_trimp_per_hr", np.nan)) and \
           not np.isnan(before_metrics.get("before_trimp_per_hr", np.nan)):
            row["delta_trimp_per_hr"] = round(
                after_metrics["after_trimp_per_hr"] - before_metrics["before_trimp_per_hr"], 2
            )
        else:
            row["delta_trimp_per_hr"] = np.nan
        
        # Cardiac Drift delta (%)
        if not np.isnan(after_metrics.get("after_cardiac_drift_pct", np.nan)) and \
           not np.isnan(before_metrics.get("before_cardiac_drift_pct", np.nan)):
            row["delta_cardiac_drift_pct"] = round(
                after_metrics["after_cardiac_drift_pct"] - before_metrics["before_cardiac_drift_pct"], 2
            )
        else:
            row["delta_cardiac_drift_pct"] = np.nan
        
        # Max 60s HR Drop delta (bpm)
        if not np.isnan(after_metrics.get("after_max_hrr_60s", np.nan)) and \
           not np.isnan(before_metrics.get("before_max_hrr_60s", np.nan)):
            row["delta_max_hrr_60s"] = round(
                after_metrics["after_max_hrr_60s"] - before_metrics["before_max_hrr_60s"], 1
            )
        else:
            row["delta_max_hrr_60s"] = np.nan
        
        # HR Zone distribution deltas (percentage points and absolute minutes)
        for zone in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
            # Percentage delta
            after_pct_key = f"after_{zone}_pct"
            before_pct_key = f"before_{zone}_pct"
            delta_pct_key = f"delta_{zone}_pct"
            after_pct = after_metrics.get(after_pct_key, 0.0)
            before_pct = before_metrics.get(before_pct_key, 0.0)
            if after_pct is not None and before_pct is not None:
                row[delta_pct_key] = round(after_pct - before_pct, 2)
            else:
                row[delta_pct_key] = np.nan
            
            # Absolute minutes delta
            after_min_key = f"after_{zone}_min"
            before_min_key = f"before_{zone}_min"
            delta_min_key = f"delta_{zone}_min"
            after_min = after_metrics.get(after_min_key, 0.0)
            before_min = before_metrics.get(before_min_key, 0.0)
            if after_min is not None and before_min is not None:
                row[delta_min_key] = round(after_min - before_min, 2)
            else:
                row[delta_min_key] = np.nan
        
        results.append(row)
    
    # Create output DataFrame
    if not results:
        log.warning("No results computed. Check input data.")
        return pd.DataFrame()
    
    output_df = pd.DataFrame(results)
    
    # Sort by date
    output_df = output_df.sort_values("date").reset_index(drop=True)
    
    # Log summary statistics
    log.info("=" * 70)
    log.info("ANALYSIS COMPLETE (v2.0 – Advanced Metrics)")
    log.info(f"Total stops analyzed: {len(output_df)}")
    
    if len(output_df) > 0:
        # Sport breakdown
        sport_counts = output_df["sport"].value_counts()
        log.info(f"Sports: {dict(sport_counts)}")
        
        # Delta statistics for all metrics
        for metric, col in [
            ("Speed (km/h)", "delta_speed_kmh"),
            ("HR (bpm)", "delta_hr_bpm"),
            ("Power (W)", "delta_power_w"),
            ("Efficiency Factor", "delta_efficiency_factor"),
            ("TRIMP/hr", "delta_trimp_per_hr"),
            ("Cardiac Drift (%)", "delta_cardiac_drift_pct"),
            ("Max 60s HR Drop (bpm)", "delta_max_hrr_60s"),
            ("Z1 (%)", "delta_Z1_pct"),
            ("Z2 (%)", "delta_Z2_pct"),
            ("Z3 (%)", "delta_Z3_pct"),
            ("Z4 (%)", "delta_Z4_pct"),
            ("Z5 (%)", "delta_Z5_pct"),
            ("Z1 (min)", "delta_Z1_min"),
            ("Z2 (min)", "delta_Z2_min"),
            ("Z3 (min)", "delta_Z3_min"),
            ("Z4 (min)", "delta_Z4_min"),
            ("Z5 (min)", "delta_Z5_min"),
        ]:
            valid = output_df[col].dropna()
            if len(valid) > 0:
                log.info(
                    f"  Δ {metric}: mean={valid.mean():.2f}, "
                    f"min={valid.min():.2f}, max={valid.max():.2f}"
                )
    
    # Save output
    output_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Results saved to: {OUTPUT_CSV}")
    log.info("=" * 70)
    
    return output_df


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result_df = process_botanical_performance()
    
    if not result_df.empty:
        print(f"\n✓ Analysis complete: {len(result_df)} botanical stops processed")
        print(f"  Output: {OUTPUT_CSV}")
    else:
        print("\n✗ No data processed. Check logs for details.")
