"""
split_cycling_activities.py
----------------------------
Extracts per-second records for each cycling activity from the master high-res
training data and saves them as individual CSV files.

Output directory : data/cycling_splits/
Filename format  : YYYY_MM_DD_bike_XX.Xkm_XXXmin_XXXm_XXXXXXXXX.CSV
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
SUMMARY_CSV    = BASE_DIR / "data" / "summaries" / "master_high_res_summary.csv"
TRAINING_CSV   = BASE_DIR / "data" / "summaries" / "master_high_res_training_data.csv"
GEOCODE_CACHE  = BASE_DIR / "data" / "processed" / "geocode_cache.json"
OUTPUT_DIR     = BASE_DIR / "data" / "cycling_splits"

# Cycling chunk size for reading large training file
CHUNK_SIZE = 50_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_diacritics(text: str) -> str:
    """Remove diacritics and return ASCII-safe string."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def sanitize_location(raw: str) -> str:
    """Strip diacritics, replace spaces/hyphens with underscore, drop
    characters that are not alphanumeric or underscore."""
    cleaned = strip_diacritics(raw.strip())
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"[^\w]", "", cleaned)          # keep only word chars
    cleaned = re.sub(r"_+", "_", cleaned).strip("_") # collapse repeated _
    return cleaned or "Neznamo"


# Tokens to remove when parsing a location from activity_name
_SPORT_TOKENS = re.compile(
    r"\b(cycling|cyklistika|cyklo|ride|gravel|mtb|mountain\s*bike|bike|kolo|"
    r"jizda|jízda|road|generic|virtual|indoor|outdoor|spinning)\b",
    flags=re.IGNORECASE,
)


def location_from_activity_name(name: str) -> str | None:
    """Try to extract a location from a Garmin/Strava activity name.

    Patterns handled:
      'Brno - Cyklistika'  →  'Brno'
      'Šumava Gravel'      →  'Sumava'
      'Ride in Prague'     →  'Prague'
      'Cyklistika'         →  None  (only sport word, no location)
    """
    if not name or pd.isna(name):
        return None
    # Remove sport-related tokens
    candidate = _SPORT_TOKENS.sub(" ", str(name))
    # Remove standalone 'in', 'v', 've', 'na' prepositions
    candidate = re.sub(r"\b(in|v|ve|na)\b", " ", candidate, flags=re.IGNORECASE)
    # Remove separators and extra whitespace
    candidate = re.sub(r"[-–—|/\\]", " ", candidate)
    candidate = candidate.strip()
    candidate = re.sub(r"\s+", " ", candidate)
    if not candidate:
        return None
    return candidate


def location_from_geocache(
    geocache: dict,
    lat: float | None,
    lon: float | None,
) -> str | None:
    """Look up nearest cached geocode entry by rounding lat/lon to 4 dp."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None

    # Try exact 4-decimal match first
    key = f"{round(lat, 4)},{round(lon, 4)}"
    address = geocache.get(key)
    if address is None:
        return None

    # The Nominatim display_name is comma-separated; find the first
    # non-numeric, non-postal part (usually village/town/city).
    parts = [p.strip() for p in str(address).split(",")]
    for part in parts:
        # Skip pure numbers (house numbers, postal codes)
        if re.fullmatch(r"[\d\s]+", part):
            continue
        # Skip administrative noise words
        if re.search(
            r"\b(okres|kraj|jihovychod|jihozapad|severovychod|severozapad|"
            r"cesko|morava|ceskomoravska|region|district|municipality)\b",
            part,
            flags=re.IGNORECASE,
        ):
            continue
        return part
    return None


def derive_location(
    activity_name: str | None,
    lat: float | None,
    lon: float | None,
    geocache: dict,
) -> str:
    """Return a sanitized location string.  Falls back to 'Neznamo'."""
    # 1. Try activity name
    raw = location_from_activity_name(activity_name)
    # 2. Fall back to geocode cache
    if not raw:
        raw = location_from_geocache(geocache, lat, lon)
    if not raw:
        return "Neznamo"
    return sanitize_location(raw)


def safe_distance(value) -> str:
    """Return distance formatted to 1 decimal, or '0.0' on missing data."""
    try:
        f = float(value)
        if pd.isna(f):
            return "0.0"
        return f"{f:.1f}"
    except (TypeError, ValueError):
        return "0.0"


def safe_duration(value) -> str:
    """Return duration rounded to nearest integer minute, or '0' on missing data."""
    try:
        f = float(value)
        if pd.isna(f):
            return "0"
        return str(round(f))
    except (TypeError, ValueError):
        return "0"


def safe_ascent(value) -> str:
    """Return ascent rounded to nearest integer metre, or '0' on missing data."""
    try:
        f = float(value)
        if pd.isna(f):
            return "0"
        return str(round(f))
    except (TypeError, ValueError):
        return "0"


def build_filename(
    date_str: str,
    distance: str,
    duration: str,
    ascent: str,
    activity_id: int | str,
) -> str:
    """Assemble the output filename.

    Example: 2026_05_17_bike_45.2km_105min_650m_986876.CSV
    """
    date_fmt = date_str.replace("-", "_")
    return f"{date_fmt}_bike_{distance}km_{duration}min_{ascent}m_{activity_id}.CSV"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load summary and filter cycling ────────────────────────────────────
    print("Načítám master_high_res_summary.csv …")
    summary = pd.read_csv(SUMMARY_CSV, dtype={"activity_id": str})

    cycling_mask = summary["sport"].str.contains(
        r"cycl|biking|ride", case=False, na=False
    )
    cycling_df = summary[cycling_mask].copy()
    cycling_ids = set(cycling_df["activity_id"].astype(str))

    print(f"Nalezeno cyklistických aktivit: {len(cycling_df)}")

    if cycling_df.empty:
        print("Žádné cyklistické aktivity nenalezeny. Konec.")
        sys.exit(0)

    # ── 2. Load geocode cache ─────────────────────────────────────────────────
    geocache: dict = {}
    if GEOCODE_CACHE.exists():
        with open(GEOCODE_CACHE, encoding="utf-8") as fh:
            geocache = json.load(fh)
        print(f"Geocode cache načten: {len(geocache)} záznamů.")
    else:
        print("Geocode cache nenalezen – lokace budou 'Neznamo'.")

    # ── 3. Create output directory ────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Výstupní složka: {OUTPUT_DIR}")

    # ── 4. Read training data in chunks; accumulate per activity ──────────────
    print(f"\nNačítám {TRAINING_CSV.name} v kusech (chunk={CHUNK_SIZE:,}) …")
    accumulated: dict[str, list[pd.DataFrame]] = {}  # activity_id -> [chunks]

    total_rows = 0
    for chunk in pd.read_csv(
        TRAINING_CSV,
        chunksize=CHUNK_SIZE,
        dtype={"activity_id": str},
        low_memory=False,
    ):
        cycling_chunk = chunk[chunk["activity_id"].isin(cycling_ids)]
        if not cycling_chunk.empty:
            for aid, grp in cycling_chunk.groupby("activity_id", sort=False):
                accumulated.setdefault(aid, []).append(grp)
        total_rows += len(chunk)
        print(f"  … zpracováno {total_rows:,} řádků, "
              f"cyklo záznamy: {sum(len(v) for v in accumulated.values()):,}",
              end="\r", flush=True)

    print()  # newline after \r

    # ── 5. Merge accumulated chunks per activity ──────────────────────────────
    activity_data: dict[str, pd.DataFrame] = {
        aid: pd.concat(frames, ignore_index=True)
        for aid, frames in accumulated.items()
    }

    print(f"Aktivity s vteřinovými záznamy: {len(activity_data)} / {len(cycling_df)}")

    # ── 6. Export each activity ───────────────────────────────────────────────
    print("\nUkládám soubory …")
    saved = 0
    skipped = 0

    for i, (_, row) in enumerate(cycling_df.iterrows(), start=1):
        aid = str(row["activity_id"])
        date_str = str(row.get("date", "0000-00-00"))[:10]
        distance = safe_distance(row.get("distance_km"))
        duration = safe_duration(row.get("duration_minutes"))
        ascent = safe_ascent(row.get("ascent_m"))
        activity_name = row.get("activity_name")

        # Determine location (kept for potential future use)
        lat = lon = None
        if aid in activity_data:
            df_act = activity_data[aid]
            # Use first record that has valid GPS
            gps_rows = df_act.dropna(subset=["position_lat", "position_long"])
            if not gps_rows.empty:
                lat = gps_rows.iloc[0]["position_lat"]
                lon = gps_rows.iloc[0]["position_long"]

        filename = build_filename(date_str, distance, duration, ascent, aid)
        out_path = OUTPUT_DIR / filename

        # Progress indicator
        print(f"  [{i:>3}/{len(cycling_df)}] {filename}", flush=True)

        if aid not in activity_data:
            skipped += 1
            continue

        activity_data[aid].to_csv(out_path, index=False)
        saved += 1

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"\nHotovo! Uloženo: {saved}, přeskočeno (bez dat): {skipped}")
    print(f"Výstup: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
