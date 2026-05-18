"""
Create a cycling-only subset of master_high_res_summary.csv.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "summaries" / "master_high_res_summary.csv"
OUTPUT = ROOT / "data" / "summaries" / "master_cycling_summary.csv"

CYCLING_KEYWORDS = r"cycl|biking|ride|bike"

# Candidate column names (checked in order)
SPORT_COLUMNS = ["sport", "type", "activity_type", "sport_type"]

df = pd.read_csv(INPUT, low_memory=False)

total = len(df)

# Find whichever sport column exists in this file
sport_col = next((c for c in SPORT_COLUMNS if c in df.columns), None)
if sport_col is None:
    raise ValueError(
        f"No sport column found. Available columns: {list(df.columns)}"
    )

cycling = df[
    df[sport_col].astype(str).str.contains(CYCLING_KEYWORDS, case=False, na=False)
]

cycling.to_csv(OUTPUT, index=False)

print(f"Sport column used : '{sport_col}'")
print(f"Total activities  : {total}")
print(f"Cycling activities: {len(cycling)}")
print(f"Saved to          : {OUTPUT}")
