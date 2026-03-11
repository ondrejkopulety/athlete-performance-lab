#!/usr/bin/env python3
"""
detect_botanical_hotspots.py – Botanical Hotspot Mapper v2.0
=============================================================
Senior Data Scientist & Geospatial Analyst

Projde KOMPLETNÍ historii cyklistických FIT souborů, najde anomální
pauzy (≥ 10 min) a pomocí DBSCAN (haversine) je shlukne do mapových
"Hotspotů" v okruhu 300 metrů.

Nové ve v2.0:
  • Exclusion Zones – manuální blacklist míst (domov, práce, …)
  • Minimum Confidence Threshold – filtr na kvalitu hotspotů
  • Interactive Folium Map – reports/hotspot_map.html
  • Spot Category – automatická kategorizace (High Probability / Frequent Stop)
  • Reverse Geocoding – dohledání adresy přes geopy

Výstup:
  • data/processed/botanical_hotspots_ranked.csv  (všechny shluky)
  • reports/hotspot_map.html                      (interaktivní mapa)
  • ASCII tabulka Top 10 v konzoli

Závislosti:
  pip install scikit-learn numpy pandas tqdm fitparse folium geopy
"""

from __future__ import annotations

# ── Dependency check ──────────────────────────────────────────────────────────
_MISSING: list[str] = []
for _pkg_name, _import_name in [
    ("scikit-learn", "sklearn"),
    ("numpy",       "numpy"),
    ("pandas",      "pandas"),
    ("tqdm",        "tqdm"),
    ("fitparse",    "fitparse"),
    ("folium",      "folium"),
    ("geopy",       "geopy"),
]:
    try:
        __import__(_import_name)
    except ImportError:
        _MISSING.append(_pkg_name)

if _MISSING:
    raise SystemExit(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║  Chybějící závislosti – nainstaluj prosím:              ║\n"
        "║                                                          ║\n"
        f"║  pip install {' '.join(_MISSING):<44s}║\n"
        "║                                                          ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )

# ── Standard library ─────────────────────────────────────────────────────────
import glob
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import folium
from folium import plugins as folium_plugins
from fitparse import FitFile
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import time as _time

from config.settings import (
    STRAVA_FIT_DIR, PROCESSED_DIR, REPORTS_DIR,
    SUMMARIES_DIR, CSV_MASTER_SUMMARY, CSV_MASTER_TRAINING,
    MIN_CONFIDENCE_THRESHOLD,
    CLUSTER_RADIUS_M,
    MIN_STOP_DURATION_S,
    STOP_SPEED_THRESHOLD_KMH,
    ANALYSIS_WINDOW_S,
    HR_SETTLE_WINDOW_S,
    CARDIO_SPORTS,
    ENABLE_EXCLUSION_ZONES,
    EXCLUDED_LOCATIONS,
)

from src.core.scoring import compute_confidence

from src.core.stop_analysis import (
    Record,
    StopSegment,
    detect_stops,
    _safe_float,
    _semicircles_to_deg,
    _speed_ms_to_kmh,
    _haversine_m,
    _is_in_excluded_zone,
    _is_cardio,
    _is_cycling,
    _extract_activity_id,
    extract_activity_id,
    parse_fit_records,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (shared thresholds imported from config.settings)
# ─────────────────────────────────────────────────────────────────────────────
# MIN_CONFIDENCE_THRESHOLD, MIN_STOP_DURATION_S, STOP_SPEED_THRESHOLD_KMH,
# ANALYSIS_WINDOW_S, HR_SETTLE_WINDOW_S, CARDIO_SPORTS, EXCLUDED_LOCATIONS,
# ENABLE_EXCLUSION_ZONES  →  all imported at the top of this file.

EARTH_RADIUS_KM    = 6371.0
# CLUSTER_RADIUS_M is imported from config.settings – do not redefine here.
DBSCAN_EPS         = CLUSTER_RADIUS_M / 1000.0 / EARTH_RADIUS_KM  # radians
DBSCAN_MIN_SAMPLES = 1           # even a single point forms a cluster

OUTPUT_CSV = os.path.join(str(PROCESSED_DIR), "botanical_hotspots_ranked.csv")
OUTPUT_MAP = os.path.join(str(REPORTS_DIR), "hotspot_map.html")

# Master CSV paths
SUMMARY_CSV_PATH = os.path.join(str(SUMMARIES_DIR), CSV_MASTER_SUMMARY)
HIGHRES_CSV_PATH = os.path.join(str(SUMMARIES_DIR), CSV_MASTER_TRAINING)

# ── Reverse Geocoding ────────────────────────────────────────────────────────
ENABLE_REVERSE_GEOCODING = True     # Set False for offline / fast runs
GEOCODER_TIMEOUT_S       = 5        # Per-request timeout (seconds)
GEOCODER_DELAY_S         = 1.1      # Delay between requests (Nominatim requires ≥ 1 s)

# Backwards-compat alias
CYCLING_SPORTS = CARDIO_SPORTS

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
        logging.FileHandler("logs/botanical_hotspots.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("botanical_hotspots")

# ─────────────────────────────────────────────────────────────────────────────
# CORE PRIMITIVES  (imported from src.core.stop_analysis)
# ─────────────────────────────────────────────────────────────────────────────
# Record, StopSegment, detect_stops (gap-tolerant + GPS-blackout-resistant),
# parse_fit_records, _safe_float, _semicircles_to_deg, _speed_ms_to_kmh,
# _haversine_m, _is_in_excluded_zone, _is_cardio, extract_activity_id
# →  see imports at top of file.

# ─────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL SCORING  →  src.core.scoring  (Single Source of Truth)
# ─────────────────────────────────────────────────────────────────────────────
# compute_confidence(records, stop) → (float, dict)  is imported at the top.
# All marker logic (A, B, C, D), gradient penalty, and graceful degradation
# live exclusively in src/core/scoring.py.  Do NOT redefine them here.


# ─────────────────────────────────────────────────────────────────────────────
# GPS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _stop_gps(records: list[Record], stop: StopSegment) -> tuple[Optional[float], Optional[float]]:
    """
    Return (lat, lon) for the stop location, optimized for DBSCAN stability.

    GPS Stability Strategy (v2.1):
    1. FIRST priority: Last valid GPS coordinate BEFORE the stop (start_idx).
       This is the most stable anchor point before Auto-Pause kicked in.
    2. SECOND priority: GPS at the exact start_idx of the stop.
    3. THIRD priority: First valid GPS within the stopped segment.
    4. LAST resort: Look backwards up to 50 rows before start_idx.

    This ordering ensures consistent clustering in DBSCAN even when GPS
    drifts during the stationary period.
    """
    # Priority 1: Last valid GPS BEFORE the stop (most stable anchor)
    for k in range(stop.start_idx - 1, max(stop.start_idx - 50, -1), -1):
        if records[k].lat is not None and records[k].lon is not None:
            return records[k].lat, records[k].lon

    # Priority 2: GPS at stop start index
    if (
        records[stop.start_idx].lat is not None
        and records[stop.start_idx].lon is not None
    ):
        return records[stop.start_idx].lat, records[stop.start_idx].lon

    # Priority 3: First valid GPS within the stop segment
    for k in range(stop.start_idx + 1, min(stop.end_idx + 1, len(records))):
        if records[k].lat is not None and records[k].lon is not None:
            return records[k].lat, records[k].lon

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def process_activity(file_path: str) -> list[dict]:
    """
    Vrátí list dicts, jeden per pauza, s klíči:
    date, activity_id, stop_time, duration_min, lat, lon, confidence
    """
    activity_id, sport, start_dt, records = parse_fit_records(file_path)
    if not records or not _is_cardio(sport):
        return []

    stops = detect_stops(records)
    results: list[dict] = []
    for stop in stops:
        lat, lon = _stop_gps(records, stop)
        if lat is None or lon is None:
            continue  # ignoruj pauzy bez GPS

        # Exclusion zone check – applied first, before any scoring
        if ENABLE_EXCLUSION_ZONES and _is_in_excluded_zone(lat, lon):
            continue

        confidence, _ = compute_confidence(records, stop)
        results.append({
            "date": start_dt.strftime("%Y-%m-%d") if start_dt else "?",
            "activity_id": activity_id,
            "stop_time": stop.start_time.strftime("%H:%M"),
            "duration_min": round(stop.duration_s / 60.0, 1),
            "lat": lat,
            "lon": lon,
            "confidence": round(confidence, 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def cluster_stops(stops_df: pd.DataFrame) -> pd.DataFrame:
    """
    Přidá sloupec 'cluster' pomocí DBSCAN (haversine, 300 m).
    Vrátí DataFrame s přidaným 'cluster'.
    Řádky s chybějícími GPS souřadnicemi jsou odstraněny před clusteringem.
    """
    stops_df = stops_df.dropna(subset=["lat", "lon"]).copy()
    if stops_df.empty:
        stops_df["cluster"] = pd.Series(dtype=int)
        return stops_df

    coords_rad = np.radians(stops_df[["lat", "lon"]].values)

    db = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = db.fit_predict(coords_rad)
    stops_df = stops_df.copy()
    stops_df["cluster"] = labels
    return stops_df


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE & RANK
# ─────────────────────────────────────────────────────────────────────────────

def _categorize_spot(visit_count: int, avg_conf: float, max_conf: float) -> str:
    """
    Kategorizuje hotspot:
      - "High Probability Spot" → vysoké confidence
      - "Frequent Stop"         → hodně návštěv, ale nízké confidence
      - "Moderate Interest"     → střední kategorie
    """
    if max_conf >= 50.0 and avg_conf >= 35.0:
        return "High Probability Spot"
    if visit_count >= 3 and avg_conf < 30.0:
        return "Frequent Stop"
    return "Moderate Interest"


def _get_ssl_context():
    """Vrátí SSL context – na macOS může chybět systémový cert bundle."""
    import ssl
    import certifi
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return ctx


GEOCODE_CACHE_PATH = os.path.join(str(PROCESSED_DIR), "geocode_cache.json")


def _reverse_geocode_batch(
    coords: list[tuple[float, float]],
) -> list[str]:
    """
    Pro seznam (lat, lon) vrátí nejbližší adresy přes Nominatim.
    Používá lokální JSON cache (data/processed/geocode_cache.json)
    s klíči zaokrouhlenými na 4 des. místa ("49.1234,16.4567").
    Při chybě vrátí prázdný řetězec.
    """
    if not ENABLE_REVERSE_GEOCODING:
        return [""] * len(coords)

    # ── Load cache ────────────────────────────────────────────────────────
    cache: dict[str, str] = {}
    if os.path.isfile(GEOCODE_CACHE_PATH):
        try:
            with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Geocode cache nelze načíst, startuji s prázdnou: %s", exc)

    geolocator = Nominatim(
        user_agent="botanical_hotspot_mapper/2.0",
        timeout=GEOCODER_TIMEOUT_S,
        ssl_context=_get_ssl_context(),
    )

    api_calls = sum(
        1 for lat, lon in coords
        if f"{round(lat, 4)},{round(lon, 4)}" not in cache
    )
    log.info(
        "🌍 Reverse geocoding %d hotspotů (%d cached, %d API calls, ~%.0f s)…",
        len(coords), len(coords) - api_calls, api_calls, api_calls * GEOCODER_DELAY_S,
    )

    results: list[str] = []
    calls_made = 0
    for lat, lon in coords:
        key = f"{round(lat, 4)},{round(lon, 4)}"
        if key in cache:
            results.append(cache[key])
            continue

        # API call – respect rate limit
        if calls_made > 0:
            _time.sleep(GEOCODER_DELAY_S)
        try:
            location = geolocator.reverse(f"{lat}, {lon}", language="cs", exactly_one=True)
            addr = location.address if location else ""
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as exc:
            log.debug("Geocoding chyba pro (%.5f, %.5f): %s", lat, lon, exc)
            addr = ""
        cache[key] = addr
        results.append(addr)
        calls_made += 1

    # ── Save cache ────────────────────────────────────────────────────────
    try:
        os.makedirs(os.path.dirname(GEOCODE_CACHE_PATH), exist_ok=True)
        with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
        log.info("💾 Geocode cache uložena (%d záznamů): %s", len(cache), GEOCODE_CACHE_PATH)
    except OSError as exc:
        log.warning("Nelze uložit geocode cache: %s", exc)

    return results


# Threshold for "high frequency" clusters that bypass confidence filter
HIGH_FREQUENCY_VISIT_COUNT: int = 3


def build_hotspot_ranking(stops_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje shluky → hotspot ranking:
    visit_count, avg_confidence, avg_duration, centroid, dates, google_maps,
    spot_category, location_name

    Smart Filter (v2.1):
    - Clusters with max_conf >= MIN_CONFIDENCE_THRESHOLD pass through.
    - Clusters with visit_count >= HIGH_FREQUENCY_VISIT_COUNT pass through
      even if confidence is low (labeled as "Frequent Stop").
    - Only clusters with BOTH low confidence AND low frequency are dropped.
    """
    groups = stops_df.groupby("cluster")

    rows: list[dict] = []
    for cid, grp in groups:
        centroid_lat = grp["lat"].mean()
        centroid_lon = grp["lon"].mean()
        visit_count  = len(grp)
        avg_conf     = grp["confidence"].mean()
        avg_dur      = grp["duration_min"].mean()
        max_conf     = grp["confidence"].max()

        # ── Smart Filter (v2.1) ──────────────────────────────────────────
        # Keep cluster if it has:
        #   A) High confidence (max_conf >= MIN_CONFIDENCE_THRESHOLD), OR
        #   B) High frequency (visit_count >= HIGH_FREQUENCY_VISIT_COUNT)
        # Drop only if BOTH conditions fail.
        if max_conf < MIN_CONFIDENCE_THRESHOLD and visit_count < HIGH_FREQUENCY_VISIT_COUNT:
            continue   # Low confidence AND low frequency → skip

        dates_list   = sorted(grp["date"].unique())
        # Full Date List – kompletní seznam všech unikátních datumů
        full_dates_str = ", ".join(dates_list)
        # Dates – zkrácený zobrazovací řetězec (posledních 5)
        dates_str    = ", ".join(dates_list[-5:])
        if len(dates_list) > 5:
            dates_str = f"… + {len(dates_list)-5} older | " + dates_str

        maps_url = (
            f"https://www.google.com/maps/search/?api=1"
            f"&query={centroid_lat:.6f},{centroid_lon:.6f}"
        )

        category = _categorize_spot(visit_count, avg_conf, max_conf)

        rows.append({
            "Rank":               0,
            "Visit Count":        visit_count,
            "Avg Confidence (%)": round(avg_conf, 1),
            "Max Confidence (%)": round(max_conf, 1),
            "Avg Duration (min)": round(avg_dur, 1),
            "Centroid Lat":       round(centroid_lat, 6),
            "Centroid Lon":       round(centroid_lon, 6),
            "Spot Category":      category,
            "Location":           "",        # doplní se geocodingem
            "Dates":              dates_str,
            "Full Date List":     full_dates_str,
            "Google Maps":        maps_url,
            "Cluster ID":         cid,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        ["Visit Count", "Avg Confidence (%)"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df["Rank"] = df.index + 1

    # ── Reverse Geocoding ────────────────────────────────────────────────
    coords = list(zip(df["Centroid Lat"], df["Centroid Lon"]))
    addresses = _reverse_geocode_batch(coords)
    df["Location"] = addresses

    col_order = [
        "Rank", "Visit Count", "Avg Confidence (%)", "Max Confidence (%)",
        "Avg Duration (min)", "Spot Category", "Location",
        "Centroid Lat", "Centroid Lon",
        "Dates", "Full Date List", "Google Maps", "Cluster ID",
    ]
    return df[col_order]


# ─────────────────────────────────────────────────────────────────────────────
# ASCII TABLE
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MAP (Folium)
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_color(avg_conf: float) -> str:
    """Zelená → Žlutá → Červená podle Avg Confidence."""
    if avg_conf < 25:
        return "green"
    if avg_conf < 50:
        return "orange"
    if avg_conf < 75:
        return "darkorange"
    return "red"


def export_folium_map(df: pd.DataFrame) -> None:
    """Vytvoří interaktivní HTML mapu reports/hotspot_map.html."""
    if df.empty:
        log.info("⚠  Mapa nevygenerována – žádné hotspoty.")
        return

    center_lat = df["Centroid Lat"].mean()
    center_lon = df["Centroid Lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="OpenStreetMap")

    for _, row in df.iterrows():
        lat = row["Centroid Lat"]
        lon = row["Centroid Lon"]
        avg_conf = row["Avg Confidence (%)"]
        color = _confidence_color(avg_conf)

        # Kompletní seznam datumů z Full Date List
        full_dates_raw = row.get("Full Date List", "")
        if isinstance(full_dates_raw, str) and full_dates_raw.strip():
            all_dates = [d.strip() for d in full_dates_raw.split(",") if d.strip()]
        else:
            all_dates = []

        location_name = row.get("Location", "")
        category = row.get("Spot Category", "")
        gmaps = row["Google Maps"]

        # Historie návštěv – formátovaný seznam datumů
        dates_html = ""
        if all_dates:
            dates_items = "".join(f"<li>{d}</li>" for d in all_dates)
            dates_html = (
                f'<hr style="margin:4px 0;">'
                f'<b>Historie návštěv ({len(all_dates)}):</b>'
                f'<div style="max-height:120px;overflow-y:auto;margin:4px 0;">'
                f'<ul style="margin:2px 0;padding-left:18px;font-size:11px;">{dates_items}</ul>'
                f'</div>'
            )

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 220px;">
            <h4 style="margin:0 0 6px;">#{int(row['Rank'])} {category}</h4>
            {'<b>' + location_name[:80] + '</b><br>' if location_name else ''}
            <hr style="margin:4px 0;">
            <b>Návštěv:</b> {row['Visit Count']}<br>
            <b>Avg Confidence:</b> {avg_conf:.1f} %<br>
            <b>Avg Délka:</b> {row['Avg Duration (min)']:.1f} min<br>
            {dates_html}
            <hr style="margin:4px 0;">
            <a href="{gmaps}" target="_blank">📍 Otevřít v Google Maps</a>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + min(row["Visit Count"], 20),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"#{int(row['Rank'])} – {row['Visit Count']}× | {avg_conf:.0f}%",
        ).add_to(m)

    # Legenda
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,.3);font-size:13px;">
        <b>Avg Confidence</b><br>
        <span style="color:green;">●</span> &lt; 25 %<br>
        <span style="color:orange;">●</span> 25–50 %<br>
        <span style="color:darkorange;">●</span> 50–75 %<br>
        <span style="color:red;">●</span> ≥ 75 %
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(OUTPUT_MAP), exist_ok=True)
    m.save(OUTPUT_MAP)
    log.info("🗺️  Interaktivní mapa uložena: %s", OUTPUT_MAP)


# ─────────────────────────────────────────────────────────────────────────────
# ASCII TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_top_table(df: pd.DataFrame, top_n: int = 10) -> None:
    """Vykreslí Top N hotspotů jako formátovanou ASCII tabulku."""
    if df.empty:
        print("\n  ⚠  Žádné hotspoty nebyly nalezeny.\n")
        return

    top = df.head(top_n).copy()
    display_cols = [
        "Rank", "Visit Count", "Avg Confidence (%)",
        "Max Confidence (%)", "Avg Duration (min)",
        "Spot Category", "Dates",
    ]
    # Přidej Location, pokud je neprázdný
    if "Location" in top.columns and top["Location"].str.strip().any():
        display_cols.insert(6, "Location")

    ddf = top[display_cols].copy()
    # Zkrať Location pro ASCII tabulku
    if "Location" in ddf.columns:
        ddf["Location"] = ddf["Location"].apply(lambda x: (x[:40] + "…") if len(str(x)) > 41 else x)

    # Šířky sloupců
    widths = {}
    for c in display_cols:
        mx = max(len(str(v)) for v in ddf[c]) if len(ddf) > 0 else 0
        widths[c] = max(len(c), mx) + 2

    total = sum(widths.values()) + len(display_cols) + 1

    print()
    print("🌿" + "═" * (total - 1))
    print("  BOTANICAL HOTSPOT MAPPER v2.0 – TOP SPOTS")
    print("═" * (total + 1))

    # Header
    hdr = "│"
    for c in display_cols:
        hdr += f" {c:^{widths[c]-2}} │"
    print(hdr)
    print("├" + "┼".join("─" * widths[c] for c in display_cols) + "┤")

    # Rows
    for _, row in ddf.iterrows():
        line = "│"
        for c in display_cols:
            val = str(row[c])
            line += f" {val:<{widths[c]-2}} │"
        print(line)

    print("└" + "┴".join("─" * widths[c] for c in display_cols) + "┘")

    # Google Maps links
    print()
    print("  📍 Google Maps Links (Top Spots):")
    for _, row in top.iterrows():
        loc_tag = f"  {row['Location'][:35]}" if row.get("Location") else ""
        print(f"     #{int(row['Rank']):>2d}  ({row['Visit Count']}×){loc_tag}  → {row['Google Maps']}")
    print()

    # Legend
    print("  📊 Interpretace:")
    print("     Visit Count       → Kolikrát jsem na tomto místě zastavil")
    print("     Avg Confidence    → Průměrná pravděpodobnost anomálie (%)")
    print("     Avg Duration      → Průměrná délka pauzy v minutách")
    print("     Spot Category     → High Probability Spot / Frequent Stop / Moderate Interest")
    print(f"     Confidence Filter → Min. {MIN_CONFIDENCE_THRESHOLD:.0f} % (max v clusteru)")
    print(f"     Cluster Radius    → {CLUSTER_RADIUS_M} metrů (body v tomto okruhu = 1 hotspot)")
    if EXCLUDED_LOCATIONS:
        print(f"     Excluded Zones    → {', '.join(EXCLUDED_LOCATIONS.keys())}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# FAST CSV LOADER (for botanical pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def load_weed_activities_fast(
    summary_csv_path: str,
    highres_csv_path: str,
    max_rides: int,
) -> pd.DataFrame:
    """
    Memory-efficient loader for the botanical-hotspots pipeline.

    1. Reads summary CSV → sorts by date DESC → filters cardio sports
       → extracts top *max_rides* activity IDs.
    2. Reads the master high-res training CSV in 250 000-row chunks,
       keeping only rows whose activity_id is in the set.
    3. Concatenates, converts timestamp to datetime, sorts, returns.
    """
    log.info("📂 Reading summary from %s…", summary_csv_path)
    summary = pd.read_csv(summary_csv_path, low_memory=False)
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
    summary = summary.dropna(subset=["date"]).sort_values("date", ascending=False)

    # Filter for cardio sports only
    if "sport" in summary.columns:
        summary = summary[summary["sport"].fillna("").apply(_is_cardio)]

    recent_ids: set[str] = set(summary.head(max_rides)["activity_id"].astype(str))
    if not recent_ids:
        log.warning("No recent cardio activity IDs found in summary CSV.")
        return pd.DataFrame()

    log.info("🔍 Chunked reading %s (filtering %d cardio activity IDs)…",
             highres_csv_path, len(recent_ids))

    needed_cols = [
        "activity_id", "timestamp", "speed", "heart_rate", "cadence",
        "altitude", "position_lat", "position_long", "is_active",
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
    log.info("✅ Loaded %d rows for %d activities (scanned %d total rows).",
             len(df), df["activity_id"].nunique(), rows_scanned)
    return df


def _records_from_dataframe(group_df: pd.DataFrame) -> list[Record]:
    """
    Convert a per-activity DataFrame slice (from the master CSV) into
    Record NamedTuples.  Uses position_lat / position_long directly.
    """
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


def process_activity_from_records(
    activity_id: str,
    sport: str,
    start_dt: datetime | None,
    records: list[Record],
) -> list[dict]:
    """
    Same logic as process_activity() but works on pre-parsed Record lists
    from the Master CSV (no FIT file needed).
    """
    if not records or not _is_cardio(sport):
        return []

    stops = detect_stops(records)
    results: list[dict] = []
    for stop in stops:
        lat, lon = _stop_gps(records, stop)
        if lat is None or lon is None:
            continue

        # Exclusion zone check – applied first, before any scoring
        if ENABLE_EXCLUSION_ZONES and _is_in_excluded_zone(lat, lon):
            continue

        confidence, _ = compute_confidence(records, stop)
        results.append({
            "date": start_dt.strftime("%Y-%m-%d") if start_dt else "?",
            "activity_id": activity_id,
            "stop_time": stop.start_time.strftime("%H:%M"),
            "duration_min": round(stop.duration_s / 60.0, 1),
            "lat": lat,
            "lon": lon,
            "confidence": round(confidence, 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    MAX_RIDES = 10000
    all_stops: list[dict] = []

    # ── Info o konfiguraci ───────────────────────────────────────────────────
    if EXCLUDED_LOCATIONS:
        log.info("🚫 Exclusion zones: %s", ", ".join(
            f"{n} ({lat:.4f},{lon:.4f} r={r}m)" for n, (lat, lon, r) in EXCLUDED_LOCATIONS.items()
        ))
    log.info("🎯 Min Confidence Threshold: %.0f %%\n", MIN_CONFIDENCE_THRESHOLD)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY PATH: Master CSV loading (pre-processed data)
    # ══════════════════════════════════════════════════════════════════════════
    if os.path.isfile(SUMMARY_CSV_PATH) and os.path.isfile(HIGHRES_CSV_PATH):
        log.info("⚡ Fast CSV loading mode from master dataset.")

        highres_df = load_weed_activities_fast(
            SUMMARY_CSV_PATH, HIGHRES_CSV_PATH, MAX_RIDES,
        )

        if highres_df.empty:
            log.info("🌿 No data loaded from master CSVs.")
            sys.exit(0)

        # Sport + date lookup from summary
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
            start_dt = date_map.get(str(aid))
            records = _records_from_dataframe(
                highres_df[highres_df["activity_id"] == str(aid)]
            )
            results = process_activity_from_records(
                str(aid), sport, start_dt, records,
            )
            if results:
                all_stops.extend(results)

        log.info("\n📊 CSV: %d cardio activities scanned, %d stops found.",
                 len(activity_ids), len(all_stops))

    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK: Direct FIT file parsing (when CSVs are not yet generated)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        log.info("📂 Master CSVs not found – falling back to direct FIT parsing.")
        strava_dir = str(STRAVA_FIT_DIR)
        if not os.path.isdir(strava_dir):
            log.error("Složka %s neexistuje.", strava_dir)
            sys.exit(1)

        fit_files = sorted(glob.glob(os.path.join(strava_dir, "**", "*.fit"), recursive=True))
        if not fit_files:
            log.warning("Žádné .fit soubory nalezeny v %s.", strava_dir)
            sys.exit(0)

        log.info("📂 Nalezeno celkem %d FIT souborů.", len(fit_files))

        cycling_rides: list[tuple[datetime, str]] = []
        for fp in tqdm(fit_files, desc="🔍 Scanning FIT files", unit="file", ncols=90):
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
            log.error("Žádné cardio aktivity nenalezeny.")
            sys.exit(1)

        cycling_rides.sort(key=lambda x: x[0], reverse=True)
        selected_rides = cycling_rides[:MAX_RIDES]
        selected_rides.sort(key=lambda x: x[0])
        print(f"\n✅ Analyzuji {len(selected_rides)} nejnovějších cardio jízd.\n")

        for dt_ride, fp in tqdm(selected_rides, desc="🔍 Processing FIT files", unit="file", ncols=90):
            try:
                results = process_activity(fp)
                if results:
                    all_stops.extend(results)
            except Exception as exc:
                log.debug("Chyba při zpracování %s: %s", os.path.basename(fp), exc)

        log.info("\n📊 FIT: %d souborů zpracováno, %d pauz nalezeno.",
                 len(selected_rides), len(all_stops))

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED OUTPUT (DBSCAN + ranking + map)
    # ══════════════════════════════════════════════════════════════════════════
    if not all_stops:
        log.info("🌿 Žádné 10+ min zastávky s GPS daty nalezeny.")
        sys.exit(0)

    # ── Exclusion Zones ──
    stops_df = pd.DataFrame(all_stops)
    n_before = len(stops_df)
    if EXCLUDED_LOCATIONS:
        mask = stops_df.apply(lambda r: not _is_in_excluded_zone(r["lat"], r["lon"]), axis=1)
        stops_df = stops_df[mask].reset_index(drop=True)
        n_excluded = n_before - len(stops_df)
        log.info("🚫 Exclusion zones: odstraněno %d / %d bodů.", n_excluded, n_before)
        if stops_df.empty:
            log.info("🌿 Po filtraci exclusion zones nezbyly žádné zastávky.")
            sys.exit(0)

    # ── DBSCAN clustering ──
    log.info("🗺️  Clustering %d zastávek (DBSCAN, radius=%d m)...", len(stops_df), CLUSTER_RADIUS_M)
    stops_df = cluster_stops(stops_df)
    n_clusters = stops_df["cluster"].nunique()
    log.info("   → Nalezeno %d unikátních hotspotů.", n_clusters)

    # ── Hotspot ranking ──
    hotspots = build_hotspot_ranking(stops_df)
    if hotspots.empty:
        log.info("🌿 Po aplikaci Confidence filtru (≥ %.0f %%) nezbyly žádné hotspoty.", MIN_CONFIDENCE_THRESHOLD)
        sys.exit(0)

    log.info("   → %d hotspotů prošlo confidence filtrem.", len(hotspots))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    hotspots.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info("📁 Report uložen: %s", OUTPUT_CSV)

    export_folium_map(hotspots)
    print_top_table(hotspots, top_n=10)

    if not hotspots.empty:
        top = hotspots.iloc[0]
        loc_info = f" ({top['Location'][:50]})" if top.get("Location") else ""
        log.info(
            "🏆 Nejčastější hotspot: %d návštěv, Avg Confidence %.1f%%, @ (%s, %s)%s",
            top["Visit Count"], top["Avg Confidence (%)"],
            top["Centroid Lat"], top["Centroid Lon"], loc_info,
        )


if __name__ == "__main__":
    main()
