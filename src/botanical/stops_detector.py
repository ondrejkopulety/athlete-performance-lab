#!/usr/bin/env python3
"""
src/botanical/stops_detector.py – The "Green Stop" Anomaly Detector v2.1
=========================================================================
Refactored from src/detect_botanical_stops.py.

Analyzes the complete history of cycling FIT files and identifies stops
(≥ 10 min) where physiological markers suggest cannabis use.

All stop-detection logic → src.core.stop_analysis
All physiological scoring → src.core.scoring  (Single Source of Truth)

Output:
  • data/processed/green_stops_report.csv
  • ASCII table in the console
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from fitparse import FitFile

from config.settings import (
    STRAVA_FIT_DIR,
    PROCESSED_DIR,
    SUMMARIES_DIR,
    CSV_MASTER_SUMMARY,
    CSV_MASTER_TRAINING,
    CARDIO_SPORTS,
    LOGS_DIR,
)

from src.core.stop_analysis import (
    Record,
    StopSegment,
    detect_stops,
    parse_fit_records,
    _safe_float,
    _speed_ms_to_kmh,
    _is_cardio,
    _is_cycling,
    extract_activity_id,
)

from src.core.scoring import compute_confidence

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_CSV = os.path.join(str(PROCESSED_DIR), "green_stops_report.csv")
SUMMARY_CSV_PATH = os.path.join(str(SUMMARIES_DIR), CSV_MASTER_SUMMARY)
HIGHRES_CSV_PATH = os.path.join(str(SUMMARIES_DIR), CSV_MASTER_TRAINING)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "green_stops.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("green_stops")


# ─────────────────────────────────────────────────────────────────────────────
# GPS → Google Maps Link
# ─────────────────────────────────────────────────────────────────────────────

def _google_maps_link(records: list[Record], stop: StopSegment) -> str:
    """Return a Google Maps URL from GPS coordinates near the stop."""
    for k in range(stop.start_idx, min(stop.end_idx + 1, len(records))):
        r = records[k]
        if r.lat is not None and r.lon is not None:
            return f"https://www.google.com/maps/search/?api=1&query={r.lat:.6f},{r.lon:.6f}"
    for k in range(stop.start_idx - 1, max(stop.start_idx - 30, -1), -1):
        r = records[k]
        if r.lat is not None and r.lon is not None:
            return f"https://www.google.com/maps/search/?api=1&query={r.lat:.6f},{r.lon:.6f}"
    return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _distance_at(records: list[Record], idx: int) -> float:
    d = records[idx].distance_m
    return d / 1000.0 if d is not None else 0.0


def _total_distance(records: list[Record]) -> float:
    for r in reversed(records):
        if r.distance_m is not None:
            return r.distance_m / 1000.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE ONE ACTIVITY (FIT fallback path)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_activity(file_path: str) -> list[dict]:
    """Analyze a single FIT file. Returns list of report row dicts."""
    activity_id, sport, start_dt, records = parse_fit_records(file_path)

    if not records:
        return []

    if not _is_cardio(sport):
        log.info("  [%s] Sport = '%s' → skipping (not cardio).", activity_id, sport)
        return []

    log.info(
        "  [%s] Sport = '%s', %d records, start = %s",
        activity_id, sport, len(records),
        start_dt.strftime("%Y-%m-%d %H:%M") if start_dt else "?",
    )

    stops = detect_stops(records)
    if not stops:
        return []

    total_dist = _total_distance(records)
    ride_date = start_dt.date() if start_dt else None

    results: list[dict] = []
    for stop in stops:
        confidence, details = compute_confidence(records, stop)
        km_before = _distance_at(records, stop.start_idx)
        km_after = total_dist - _distance_at(records, stop.end_idx)

        results.append({
            "Date": ride_date.strftime("%Y-%m-%d") if ride_date else "?",
            "Activity ID": activity_id,
            "Stop Time": stop.start_time.strftime("%H:%M:%S"),
            "Duration (min)": f"{stop.duration_s / 60:.1f}",
            "Km PŘED (km)": f"{km_before:.1f}",
            "Km PO (km)": f"{km_after:.1f}",
            "HR Trend v pauze": details["desc_a"],
            "HR Score": f"{details['score_a']:.0f}",
            "Decoupling Post-Stop": details["desc_b"],
            "Decoupling Score": f"{details['score_b']:.0f}",
            "Cadence Drop": details["desc_c"],
            "Cadence Score": f"{details['score_c']:.0f}",
            "Respiration": details["desc_d"],
            "Respiration Score": f"{details['score_d']:.0f}",
            "Google Maps Link": _google_maps_link(records, stop),
            "Confidence Score (%)": f"{confidence:.1f}",
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ASCII TABLE FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def print_ascii_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n  ⚠  Žádné podezřelé zastávky nebyly nalezeny.\n")
        return

    display_cols = [
        "Date", "Stop Time", "Duration (min)",
        "Km PŘED (km)", "Km PO (km)",
        "HR Trend v pauze", "Decoupling Post-Stop",
        "Cadence Drop", "Respiration",
        "Confidence Score (%)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    ddf = df[display_cols].copy()

    col_widths = {}
    for col in display_cols:
        max_data = max(len(str(v)) for v in ddf[col]) if len(ddf) > 0 else 0
        col_widths[col] = max(len(col), max_data) + 2

    total_width = sum(col_widths.values()) + len(display_cols) + 1

    print()
    print("🌿" + "═" * (total_width - 1))
    print("  THE GREEN STOP ANOMALY DETECTOR v2.1 – RESULTS")
    print("═" * (total_width + 1))

    header = "│"
    for col in display_cols:
        header += f" {col:^{col_widths[col] - 2}} │"
    print(header)
    print("├" + "┼".join("─" * col_widths[col] for col in display_cols) + "┤")

    for _, row in ddf.iterrows():
        line = "│"
        for col in display_cols:
            val = str(row[col])
            line += f" {val:<{col_widths[col] - 2}} │"
        print(line)

    print("└" + "┴".join("─" * col_widths[col] for col in display_cols) + "┘")

    print()
    print("  📊 Confidence Score Legend:")
    print("     0-20%   → Normal stop (food, navigation)")
    print("     20-45%  → Mildly suspicious")
    print("     45-70%  → Suspicious – physiological anomalies present")
    print("     70-90%  → Highly probable – most markers positive")
    print("     90-100% → Near-certain – all physiological markers firing 🔥")
    print()

    if "Google Maps Link" in df.columns:
        maps = df[["Date", "Stop Time", "Google Maps Link"]].copy()
        maps = maps[maps["Google Maps Link"] != "N/A"]
        if not maps.empty:
            print("  📍 Google Maps Links:")
            for _, r in maps.iterrows():
                print(f"     {r['Date']} {r['Stop Time']} → {r['Google Maps Link']}")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# FAST CSV LOADER (chunked, memory-efficient)
# ─────────────────────────────────────────────────────────────────────────────

def load_weed_activities_fast(
    summary_csv_path: str,
    highres_csv_path: str,
    max_rides: int,
) -> pd.DataFrame:
    """
    Memory-efficient loader for the weed-analytics pipeline.

    1. Reads summary CSV → sorts by date desc → extracts top *max_rides*
       activity IDs into a set of strings.
    2. Reads the high-res training CSV in 250 000-row chunks, keeping only
       rows whose activity_id is in the set.
    3. Concatenates, converts timestamp to datetime, sorts, and returns.
    """
    log.info("📂 Reading summary from %s…", summary_csv_path)
    summary = pd.read_csv(summary_csv_path, low_memory=False)
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
    summary = summary.dropna(subset=["date"]).sort_values("date", ascending=False)

    if "sport" in summary.columns:
        summary = summary[summary["sport"].fillna("").apply(_is_cardio)]

    recent_ids: set[str] = set(summary.head(max_rides)["activity_id"].astype(str))

    if not recent_ids:
        log.warning("No recent cardio activity IDs found in summary CSV.")
        return pd.DataFrame()

    log.info(
        "🔍 Chunked reading %s (filtering %d cardio activity IDs)…",
        highres_csv_path, len(recent_ids),
    )

    needed_cols = [
        "activity_id", "timestamp", "speed", "heart_rate", "cadence",
        "altitude", "position_lat", "position_long", "is_active", "distance",
    ]

    chunks: list[pd.DataFrame] = []
    rows_scanned = 0
    for chunk in pd.read_csv(
        highres_csv_path,
        chunksize=250_000,
        usecols=lambda c: c in needed_cols,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        chunk["activity_id"] = chunk["activity_id"].astype(str)
        filtered = chunk[chunk["activity_id"].isin(recent_ids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        log.warning("No matching rows in high-res CSV (%d rows scanned).", rows_scanned)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(
        "✅ Loaded %d rows for %d activities (scanned %d total rows).",
        len(df), df["activity_id"].nunique(), rows_scanned,
    )
    return df


def _records_from_dataframe(group_df: pd.DataFrame) -> list[Record]:
    """Convert a per-activity DataFrame slice into Record NamedTuples."""
    records: list[Record] = []
    for row in group_df.itertuples(index=False):
        ts = getattr(row, "timestamp", None)
        if pd.isna(ts):
            continue
        records.append(Record(
            timestamp=ts,
            heart_rate=_safe_float(getattr(row, "heart_rate", None)),
            speed_ms=_safe_float(getattr(row, "speed", None)),
            distance_m=_safe_float(getattr(row, "distance", None)),
            altitude_m=_safe_float(getattr(row, "altitude", None)),
            cadence=_safe_float(getattr(row, "cadence", None)),
            power=_safe_float(getattr(row, "power", None)),
            respiratory_rate=_safe_float(getattr(row, "respiratory_rate", None)),
            lat=_safe_float(getattr(row, "position_lat", None)),
            lon=_safe_float(getattr(row, "position_long", None)),
        ))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    MAX_RIDES = 10000
    all_results: list[dict] = []
    n_analyzed = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY PATH: Fast chunked CSV loading (pre-processed master data)
    # ══════════════════════════════════════════════════════════════════════════
    if os.path.isfile(SUMMARY_CSV_PATH) and os.path.isfile(HIGHRES_CSV_PATH):
        log.info("⚡ Fast CSV loading mode (chunked) from master dataset.")

        highres_df = load_weed_activities_fast(
            SUMMARY_CSV_PATH, HIGHRES_CSV_PATH, MAX_RIDES,
        )

        if highres_df.empty:
            log.info("🌿 No data loaded from CSVs.")
            sys.exit(0)

        summary = pd.read_csv(SUMMARY_CSV_PATH, low_memory=False)
        sport_map = dict(zip(
            summary["activity_id"].astype(str),
            summary["sport"].fillna("").astype(str),
        ))
        date_map = dict(zip(
            summary["activity_id"].astype(str),
            pd.to_datetime(summary["date"], errors="coerce"),
        ))

        activity_ids = highres_df["activity_id"].unique()
        for aid in activity_ids:
            sport = sport_map.get(str(aid), "")
            if not _is_cardio(sport):
                continue

            start_dt = date_map.get(str(aid))
            records = _records_from_dataframe(
                highres_df[highres_df["activity_id"] == str(aid)]
            )
            if not records:
                continue

            n_analyzed += 1
            log.info(
                "  [%s] %s, %d records, %s",
                aid, sport, len(records),
                start_dt.strftime("%Y-%m-%d") if pd.notna(start_dt) else "?",
            )

            stops = detect_stops(records)
            if not stops:
                continue

            total_dist = _total_distance(records)
            ride_date = start_dt.date() if pd.notna(start_dt) else None

            for stop in stops:
                confidence, details = compute_confidence(records, stop)
                km_before = _distance_at(records, stop.start_idx)
                km_after = total_dist - _distance_at(records, stop.end_idx)

                all_results.append({
                    "Date": ride_date.strftime("%Y-%m-%d") if ride_date else "?",
                    "Activity ID": aid,
                    "Stop Time": stop.start_time.strftime("%H:%M:%S"),
                    "Duration (min)": f"{stop.duration_s / 60:.1f}",
                    "Km PŘED (km)": f"{km_before:.1f}",
                    "Km PO (km)": f"{km_after:.1f}",
                    "HR Trend v pauze": details["desc_a"],
                    "HR Score": f"{details['score_a']:.0f}",
                    "Decoupling Post-Stop": details["desc_b"],
                    "Decoupling Score": f"{details['score_b']:.0f}",
                    "Cadence Drop": details["desc_c"],
                    "Cadence Score": f"{details['score_c']:.0f}",
                    "Respiration": details["desc_d"],
                    "Respiration Score": f"{details['score_d']:.0f}",
                    "Google Maps Link": _google_maps_link(records, stop),
                    "Confidence Score (%)": f"{confidence:.1f}",
                })

        print(f"\n✅ CSV fast-loading: {n_analyzed} cardio activities analyzed.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK: Direct FIT file parsing (when CSVs are not yet generated)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        log.info("📂 CSV files not found – falling back to direct FIT parsing.")
        strava_dir = str(STRAVA_FIT_DIR)
        if not os.path.isdir(strava_dir):
            log.error("Folder %s does not exist.", strava_dir)
            sys.exit(1)

        fit_files = sorted(glob.glob(os.path.join(strava_dir, "**", "*.fit"), recursive=True))
        if not fit_files:
            log.error("No .fit files in %s.", strava_dir)
            sys.exit(1)

        log.info("Found %d FIT files total.", len(fit_files))

        cycling_rides: list[tuple[datetime, str]] = []
        for fp in fit_files:
            try:
                fitfile = FitFile(fp)
                sport = ""
                start_dt_scan: datetime | None = None
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
                        start_dt_scan = ts
                    break
                if _is_cardio(sport) and start_dt_scan:
                    cycling_rides.append((start_dt_scan, fp))
            except Exception:
                continue

        if not cycling_rides:
            log.error("No cardio activities found.")
            sys.exit(1)

        cycling_rides.sort(key=lambda x: x[0], reverse=True)
        selected = cycling_rides[:MAX_RIDES]
        selected.sort(key=lambda x: x[0])
        print(f"\n✅ Analyzing {len(selected)} most recent cardio rides.\n")

        for i, (dt, fp) in enumerate(selected, 1):
            log.info("[%d/%d] %s – %s", i, len(selected), dt.strftime("%Y-%m-%d"), os.path.basename(fp))
            results = analyze_activity(fp)
            all_results.extend(results)

        n_analyzed = len(selected)

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT (shared by both paths)
    # ══════════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_results)

    if not df.empty:
        df["_sort_conf"] = df["Confidence Score (%)"].astype(float)
        df = df.sort_values("_sort_conf", ascending=False).drop(columns="_sort_conf")
        df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info("\n📁 Report saved: %s", OUTPUT_CSV)

    print_ascii_table(df)

    if not df.empty:
        high_conf = df[df["Confidence Score (%)"].astype(float) >= 50.0]
        log.info(
            "🌿 Total: %d stops analyzed, %d with Confidence ≥ 50%%.",
            len(df), len(high_conf),
        )
    else:
        log.info("🌿 No 10+ min stops found in %d cardio rides.", n_analyzed)


if __name__ == "__main__":
    main()
