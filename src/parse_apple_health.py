"""
parse_apple_health.py
─────────────────────
Načte všechny HealthExport_*.csv soubory (Apple Health export), vyčistí data
a vytvoří čtyři samostatné CSV soubory se strukturou zrcadlící Garmin exporty:

  data/summaries/apple_hrv.csv          date, last_night_avg, weekly_avg
  data/summaries/apple_sleep.csv        date, sleep_score, duration_minutes, …
  data/summaries/apple_daily_health.csv date, resting_heart_rate, body_battery…
  data/summaries/apple_vo2_max.csv      date, vo2_max

Tato data jsou SAMOSTATNÁ – nijak nemodifikují existující Garmin CSV soubory.

Mapování Apple → výstupní sloupce
  Heart Rate Variability (ms)  → last_night_avg  (průměr denního range)
                                 weekly_avg       (7denní rolling mean)
  Sleep ("7h 20m")             → duration_minutes (součet za den)
  Blood Oxygen (%)             → avg_spo2         (průměr denního range)
  Resting Heart Rate (bpm)     → resting_heart_rate
  Cardio Fitness (mL/min·kg)   → vo2_max

Poznámka k sleep fázím:
  Apple Health denní export neobsahuje fázové detaily (REM/Core/Deep).
  Sloupce sleep_score, *_percentage apod. jsou přítomny (zachování schématu),
  ale prázdné (NaN).
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Cesty ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
SUMMARIES_DIR = DATA_DIR / "summaries"

# Výstupní soubory (apple_ prefix, nikdy nepřepisují Garmin data)
OUT_HRV    = SUMMARIES_DIR / "apple_hrv.csv"
OUT_SLEEP  = SUMMARIES_DIR / "apple_sleep.csv"
OUT_HEALTH = SUMMARIES_DIR / "apple_daily_health.csv"
OUT_VO2    = SUMMARIES_DIR / "apple_vo2_max.csv"

# ── Schémata sloupců — přesné zrcadlo Garmin exportů ─────────────────────────
HRV_COLS    = ["date", "last_night_avg", "weekly_avg"]
SLEEP_COLS  = ["date", "sleep_score", "duration_minutes", "sleep_start_time",
               "sleep_end_time", "rem_sleep_percentage", "light_sleep_percentage",
               "deep_sleep_percentage", "awake_count", "avg_spo2", "avg_respiration"]
HEALTH_COLS = ["date", "resting_heart_rate", "body_battery_highest",
               "body_battery_lowest", "stress_average", "stress_max",
               "active_calories", "resting_calories"]
VO2_COLS    = ["date", "vo2_max"]


# ── Pomocné funkce ────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, keyword: str) -> "str | None":
    """Vrátí první sloupec obsahující keyword (case-insensitive), jinak None."""
    kw = keyword.lower()
    for c in df.columns:
        if kw in c.lower():
            return c
    return None


def _clean_numeric(value: "str | float") -> float:
    """Normalizuje evropský numerický formát → float.

    • Odstraní \\xa0 (non-breaking space), mezery a thousands separátory.
    • Nahradí desetinnou čárku tečkou.
    • Range „22,44-66,01" nebo „97-98" → průměr (min+max)/2.
    • „-" nebo prázdný string → NaN.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().strip('"').strip("'")
    if s in ("-", ""):
        return np.nan
    # Agresivně odstraň všechny druhy mezer před dalším zpracováním
    # (Apple používá \xa0 i \u202f jako thousands sep, např. "1\xa0980", "10\u202f810")
    s = s.replace("\xa0", "").replace("\u202f", "").replace("\u00a0", "").replace(" ", "")
    # Desetinná čárka → tečka
    s = s.replace(",", ".")
    # Range: dvě čísla oddělená pomlčkou (obě musejí být kladná čísla)
    if "-" in s:
        parts = s.split("-", 1)
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (lo + hi) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# "7h 20m" | "6h" | "45m" | "1h 2m"
_SLEEP_RE = re.compile(r"(?:(?P<hours>\d+)\s*h)?\s*(?:(?P<minutes>\d+)\s*m)?", re.I)


def _parse_sleep_minutes(value: "str | float") -> float:
    """Převede Apple sleep string ('Xh Ym') na celkové minuty. '-' / prázdno → NaN."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s in ("-", ""):
        return np.nan
    m = _SLEEP_RE.match(s)
    if not m or (m.group("hours") is None and m.group("minutes") is None):
        return np.nan
    hours   = int(m.group("hours"))   if m.group("hours")   else 0
    minutes = int(m.group("minutes")) if m.group("minutes") else 0
    total   = hours * 60 + minutes
    return float(total) if total > 0 else np.nan


def _parse_date(value: "str | float") -> "str | float":
    """Převede DD.MM.YYYY (nebo DD.MM.YYYY HH:MM) → YYYY-MM-DD ISO. Neplatné → NaN.

    Používá format='mixed' s dayfirst=True pro robustní parsování bez zbytečných
    dropů řádků kvůli variacím formátu (uvozovky, mezery, různé separátory).
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().strip('"').strip("'")
    if not s:
        return np.nan
    try:
        return pd.to_datetime(s, format="mixed", dayfirst=True).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return np.nan


# ── Načítání souborů ──────────────────────────────────────────────────────────

def find_health_files() -> list[str]:
    """Rekurzivně najde všechny HealthExport_*.csv soubory pod DATA_DIR."""
    pattern = str(DATA_DIR / "**" / "HealthExport_*.csv")
    return sorted(glob.glob(pattern, recursive=True))


def load_and_concat(files: list[str]) -> pd.DataFrame:
    """Načte a spojí všechny soubory do jednoho surového DataFrame."""
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, skipinitialspace=True)
            # Odstraň whitespace i případné uvozovky z názvů sloupců
            df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
            frames.append(df)
            print(f"  načteno: {os.path.relpath(f, PROJECT_ROOT)}  ({len(df)} řádků)")
        except Exception as exc:
            print(f"  CHYBA: {f}: {exc}")
    if not frames:
        raise SystemExit("Nebyly nalezeny žádné platné HealthExport_*.csv soubory.")
    combined = pd.concat(frames, ignore_index=True)
    # Odstraň identické řádky způsobené překrývajícími se nebo duplicitními exporty
    before = len(combined)
    combined = combined.drop_duplicates()
    dropped = before - len(combined)
    if dropped:
        print(f"  → odstraněno {dropped} zcela duplicitních řádků (překrývající se exporty)")
    return combined


# ── Čištění a extrakce ────────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Parsuje a normalizuje všechny relevantní sloupce ze surového DataFrame."""
    df = df.copy()

    # ── Datum ─────────────────────────────────────────────────────────────────
    df["date"] = df["Date"].apply(_parse_date)
    df = df.dropna(subset=["date"])

    # ── Pouze řádky denního souhrnu (ne workout řádky) ────────────────────────
    workout_col = df.get("Workout Type", pd.Series("", index=df.index))
    daily_mask  = workout_col.fillna("").str.strip() == ""

    # ── Cardio Fitness → vo2_max ───────────────────────────────────────────────
    cf = _find_col(df, "cardio fitness")
    df["vo2_max"] = df[cf].apply(_clean_numeric) if cf else np.nan

    # ── Resting Heart Rate ────────────────────────────────────────────────────
    rhr = _find_col(df, "resting heart rate")
    df["resting_heart_rate"] = df[rhr].apply(_clean_numeric) if rhr else np.nan

    # ── HRV: denní range → průměr (min+max)/2 = odhad last_night_avg ──────────
    hrv_col = _find_col(df, "heart rate variability")
    df["hrv_avg"] = df[hrv_col].apply(_clean_numeric) if hrv_col else np.nan

    # ── Blood Oxygen → avg_spo2 (daily rows only) ─────────────────────────────
    spo2_col = _find_col(df, "blood oxygen")
    df["avg_spo2"] = np.nan
    if spo2_col:
        df.loc[daily_mask, "avg_spo2"] = df.loc[daily_mask, spo2_col].apply(_clean_numeric)

    # ── Sleep duration (daily rows only) ─────────────────────────────────────
    sleep_col = _find_col(df, "sleep")
    df["duration_minutes"] = np.nan
    if sleep_col:
        df.loc[daily_mask, "duration_minutes"] = (
            df.loc[daily_mask, sleep_col].apply(_parse_sleep_minutes)
        )

    # ── Active Calories (daily rows only) ─────────────────────────────────────
    ac_col = _find_col(df, "active calories")
    df["active_calories"] = np.nan
    if ac_col:
        df.loc[daily_mask, "active_calories"] = (
            df.loc[daily_mask, ac_col].apply(_clean_numeric)
        )

    # ── Resting Calories (daily rows only) ────────────────────────────────────
    rc_col = _find_col(df, "resting calories")
    df["resting_calories"] = np.nan
    if rc_col:
        df.loc[daily_mask, "resting_calories"] = (
            df.loc[daily_mask, rc_col].apply(_clean_numeric)
        )

    return df


# ── Denní agregace ────────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agreguje více vzorků za den.

    Biometrické hodnoty (HRV, RHR, VO2, SpO2) → mean()
    Spánek (duration_minutes) → sum()  [více záznamů = součet, ne průměr]
    """
    agg: dict = {
        # Biometriky → průměr
        "vo2_max":            "mean",
        "resting_heart_rate": "mean",
        "hrv_avg":            "mean",
        "avg_spo2":           "mean",
        # Součtové veličiny → sum
        "duration_minutes":   "sum",
        "active_calories":    "sum",
        "resting_calories":   "sum",
    }
    # Ponech jen sloupce, které existují
    agg = {k: v for k, v in agg.items() if k in df.columns}
    daily = df.groupby("date", as_index=False).agg(agg)
    # sum() produkuje 0 tam, kde nebyla žádná data → nahraď NaN
    for sum_col in ("duration_minutes", "active_calories", "resting_calories"):
        if sum_col in daily.columns:
            daily[sum_col] = daily[sum_col].replace(0, np.nan)
    return daily.sort_values("date").reset_index(drop=True)


# ── Sestavení výstupních DataFrame ────────────────────────────────────────────

def _enforce_schema(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Přidá chybějící sloupce jako NaN a vrátí df s přesně danými sloupci v daném pořadí."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].reset_index(drop=True)


def build_hrv(daily: pd.DataFrame) -> pd.DataFrame:
    """apple_hrv.csv — date, last_night_avg, weekly_avg.

    last_night_avg = průměr Apple HRV range za den
    weekly_avg     = 7denní rolling mean z last_night_avg (min 3 vzorky)
    """
    out = daily[["date", "hrv_avg"]].dropna(subset=["hrv_avg"]).copy()
    out = out.rename(columns={"hrv_avg": "last_night_avg"})
    out["last_night_avg"] = out["last_night_avg"].round(1)
    out = out.sort_values("date").reset_index(drop=True)
    out["weekly_avg"] = (
        out["last_night_avg"]
        .rolling(window=7, min_periods=3)
        .mean()
        .round(1)
    )
    return _enforce_schema(out, HRV_COLS)


def build_sleep(daily: pd.DataFrame) -> pd.DataFrame:
    """apple_sleep.csv — Garmin sleep.csv kompatibilní schéma.

    Pouze duration_minutes a avg_spo2 jsou dostupná z Apple denního exportu.
    Ostatní sloupce (sleep_score, fáze, timestamps) zůstávají NaN.
    """
    cols_available = ["date", "duration_minutes", "avg_spo2"]
    out = daily[cols_available].dropna(subset=["duration_minutes"]).copy()
    out["duration_minutes"] = out["duration_minutes"].round().astype(int)
    out["avg_spo2"] = out["avg_spo2"].round(1)
    return _enforce_schema(out, SLEEP_COLS)


def build_daily_health(daily: pd.DataFrame) -> pd.DataFrame:
    """apple_daily_health.csv — Garmin daily_health.csv kompatibilní schéma.

    Z Apple exportu dostupné: resting_heart_rate, active_calories, resting_calories.
    Sloupce body_battery_*, stress_* zůstávají NaN (Apple je neposkytuje).
    """
    key_sources = ["resting_heart_rate", "active_calories", "resting_calories"]
    available = ["date"] + [c for c in key_sources if c in daily.columns]
    out = daily[available].copy()
    out = out.dropna(subset=[c for c in key_sources if c in out.columns], how="all")
    if "resting_heart_rate" in out.columns:
        out["resting_heart_rate"] = out["resting_heart_rate"].round(1)
    if "active_calories" in out.columns:
        out["active_calories"] = out["active_calories"].round(0)
    if "resting_calories" in out.columns:
        out["resting_calories"] = out["resting_calories"].round(0)
    return _enforce_schema(out, HEALTH_COLS)


def build_vo2(daily: pd.DataFrame) -> pd.DataFrame:
    """apple_vo2_max.csv — date, vo2_max."""
    out = daily[["date", "vo2_max"]].dropna(subset=["vo2_max"]).copy()
    out["vo2_max"] = out["vo2_max"].round(2)
    return _enforce_schema(out, VO2_COLS)


# ── Uložení ───────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: Path, label: str) -> None:
    """Uloží DataFrame do CSV a vypíše počet řádků."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  {label:<30} → {len(df):>5} záznamů  ({path.name})")


# ── Hlavní funkce ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("Apple Health → standalone apple_*.csv export")
    print("=" * 64)

    # 1) Najdi soubory
    files = find_health_files()
    print(f"\nNalezeno {len(files)} HealthExport_*.csv souborů:")
    for f in files:
        print(f"  • {os.path.relpath(f, PROJECT_ROOT)}")

    # 2) Načti & spoj
    print("\nNačítám soubory…")
    raw = load_and_concat(files)
    print(f"Celkem řádků po spojení: {len(raw)}")

    # 3) Vyčisti a extrahuj relevantní sloupce
    print("\nČistím data…")
    cleaned = clean_raw(raw)
    print(f"Řádků s platným datem: {len(cleaned)}")

    # 4) Denní agregace
    daily = aggregate_daily(cleaned)
    print(f"Unikátních dnů po agregaci: {len(daily)}")

    # 5) Sestav výstupní DataFrame
    hrv    = build_hrv(daily)
    sleep  = build_sleep(daily)
    health = build_daily_health(daily)
    vo2    = build_vo2(daily)

    # 6) Ulož
    print("\nUkládám soubory:")
    save(hrv,    OUT_HRV,    "apple_hrv.csv")
    save(sleep,  OUT_SLEEP,  "apple_sleep.csv")
    save(health, OUT_HEALTH, "apple_daily_health.csv")
    save(vo2,    OUT_VO2,    "apple_vo2_max.csv")

    print("\n" + "─" * 64)
    print("Hotovo. Garmin/Strava CSV soubory nebyly změněny. ✓")
    print("─" * 64)


if __name__ == "__main__":
    main()