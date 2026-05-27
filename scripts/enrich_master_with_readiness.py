"""
Enrich master activity summary with daily readiness metrics via LEFT JOIN on date.
Output: data/summaries/master_enriched_summary.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data/summaries/master_high_res_summary.csv"
READINESS_PATH = BASE_DIR / "data/summaries/athlete_readiness.csv"
OUTPUT_PATH = BASE_DIR / "data/summaries/master_enriched_summary.csv"


def main() -> None:
    # --- Load ---
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    readiness = pd.read_csv(READINESS_PATH, low_memory=False)

    print(f"master rows   : {len(master):,}")
    print(f"readiness rows: {len(readiness):,}")

    # --- Normalise join key ---
    # Convert to plain YYYY-MM-DD string (handles datetime strings like "2022-08-04 07:15:00")
    master["date"] = pd.to_datetime(master["date"]).dt.strftime("%Y-%m-%d")
    readiness["date"] = pd.to_datetime(readiness["date"]).dt.strftime("%Y-%m-%d")

    # Columns that readiness adds (everything except the shared key)
    readiness_cols = [c for c in readiness.columns if c != "date"]

    # Drop any columns that already exist in master to avoid _x/_y suffixes
    overlap = [c for c in readiness_cols if c in master.columns]
    if overlap:
        print(f"\nDropping {len(overlap)} overlapping readiness column(s) from right side: {overlap}")
        readiness = readiness.drop(columns=overlap)
        readiness_cols = [c for c in readiness_cols if c not in overlap]

    # --- LEFT JOIN ---
    enriched = pd.merge(master, readiness, on="date", how="left")

    # --- Validate row count ---
    assert len(enriched) == len(master), (
        f"Row count changed after merge: {len(master)} → {len(enriched)}"
    )
    print(f"\nEnriched rows : {len(enriched):,}  ✓ (unchanged)")

    # --- Report new columns ---
    matched_days = enriched["CTL"].notna().sum() if "CTL" in enriched.columns else 0
    print(f"Activities with readiness data matched: {matched_days:,} / {len(enriched):,}")
    print(f"\nNew columns added from readiness ({len(readiness_cols)}):")
    print("  " + ", ".join(readiness_cols))

    # Sample of new columns for the first matched row
    sample_cols = ["date"] + readiness_cols[:6]
    sample = enriched.loc[enriched["CTL"].notna(), sample_cols].head(3) if "CTL" in enriched.columns else enriched[sample_cols].head(3)
    print("\nSample (first 3 matched rows):")
    print(sample.to_string(index=False))

    # --- Save ---
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
