"""
master_rebuild.py  –  Master Training Database Rebuild
=======================================================
Sloučí Garmin (data/fit/) a Strava (data/fit/strava_originals/) FIT soubory
do jedné čisté tréninkové databáze *bez duplicit*.

Deduplikace (±30 min) – priorita snímač tepové frekvence → hustota dat:
  1. Snímač: Hrudní pás (ANT+ device_type 120) > Optický snímač (zápěstí).
     • Případ A: Garmin má hrudní pás  → Garmin vždy vítězí.
     • Případ B: Garmin má optiku, Strava má hrudní pás → vítězí Strava.
     • Případ C: Ani jeden nemá hrudní pás → Garmin (výchozí zdroj).
  2. Pojistka – stejný typ snímače: rozhoduje hustota HR dat (≥ 90 % vzorků).
  3. Při shodě: větší soubor.

Výstup (režim 'w' – vždy od nuly):
  • data/summaries/master_high_res_training_data.csv
  • data/summaries/master_high_res_summary.csv
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
from datetime import datetime, timedelta
from statistics import mean
from typing import Optional

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fitparse import FitFile

from config.settings import (
    DEDUP_WINDOW_MIN, HR_DENSITY_THRESHOLD,
    ANT_DEVICE_TYPE_HR, INTEGRITY_DIFF_PCT,
)

# Import sdílené logiky z fit_parser (formerly fit_to_highres_csv.py)
from src.ingestion.fit_parser import (
    BASELINE_RHR,
    HIGHRES_COLS,
    MAX_HR,
    SUMMARY_COLS,
    SUMMARY_FOLDER,
    extract_activity_id,
    parse_fit_file,
    parse_fit_to_memory,
    _to_naive_utc,
)

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACE
# ─────────────────────────────────────────────────────────────────────────────

GARMIN_FIT_FOLDER  = "data/fit"
STRAVA_FIT_FOLDER  = "data/fit/strava_originals"

METADATA_CACHE_FILE = os.path.join(SUMMARY_FOLDER, "metadata_cache.json")

MASTER_HIGHRES_CSV = os.path.join(SUMMARY_FOLDER, "master_high_res_training_data.csv")
MASTER_SUMMARY_CSV = os.path.join(SUMMARY_FOLDER, "master_high_res_summary.csv")

DEDUP_WINDOW          = timedelta(minutes=DEDUP_WINDOW_MIN)

# Minimální počet R-R intervalů (HRV zpráv) pro spolehlivou detekci hrudního pásu.
# Strava exporty mažou device_info, ale HRV zprávy přežijí → proxy pro SENSOR_STRAP.
HRV_MIN_INTERVALS = 30

# Verze schématu metadata cache. Při přidání nového pole zvyšte o 1 – vynutí
# přeparsování všech FIT souborů a zabrání čtení neúplných záznamů.
CACHE_VERSION = 2

SOURCE_STRAVA = "strava"
SOURCE_GARMIN = "garmin"

SENSOR_STRAP   = "strap"    # hrudní pás (ANT+/BLE external HR sensor)
SENSOR_OPTICAL = "optical"  # optický snímač na zápěstí
SENSOR_UNKNOWN = "unknown"  # nelze určit

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
log = logging.getLogger("master_rebuild")
log.setLevel(logging.INFO)
log.propagate = False          # ← nepropagovat do root loggeru z fit_to_highres_csv
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    _fh = logging.FileHandler("logs/master_rebuild.log", mode="w", encoding="utf-8")
    _fh.setFormatter(_fmt)
    log.handlers = [_sh, _fh]


# ─────────────────────────────────────────────────────────────────────────────
# TIMEZONE HELPER – imported from fit_parser (_to_naive_utc)
# Single source of truth for the entire pipeline.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# KOMBINOVANÁ EXTRAKCE METADAT FIT SOUBORU  (jediné otevření souboru)
# ─────────────────────────────────────────────────────────────────────────────

def get_fit_metadata(fit_path: str) -> dict:
    """
    Otevře FIT soubor jednou a extrahuje všechna metadata najednou – snižuje
    počet diskových operací o 66 % oproti volání tří samostatných funkcí.

    Vrací dict s klíči:
      start_time   – datetime nebo None
      hr_samples   – počet record zpráv s platnou heart_rate > 0
      total_records– celkový počet record zpráv
      hr_density   – hr_samples / total_records  (0.0–1.0)
      sensor_type  – SENSOR_STRAP | SENSOR_OPTICAL | SENSOR_UNKNOWN
    """
    result: dict = {
        "start_time":    None,
        "hr_samples":    0,
        "total_records": 0,
        "hr_density":    0.0,
        "sensor_type":   SENSOR_UNKNOWN,
        "hrv_count":     0,   # počet R-R intervalů; > 0 → hrudní pás (i bez device_info)
    }

    try:
        fitfile = FitFile(fit_path)
    except Exception as exc:
        log.warning("Nelze otevřít %s: %s", fit_path, exc)
        return result

    # ── start_time (přednostně ze session, záloha z prvního recordu) ─────────
    # Both Garmin (UTC naive) and Strava (tz-aware local) exports are normalised
    # to naive UTC so the ±30 min dedup window compares identical time scales.
    _first_record_ts: Optional[datetime] = None
    for msg in fitfile.get_messages("session"):
        ts = msg.get_values().get("start_time")
        if isinstance(ts, datetime):
            result["start_time"] = _to_naive_utc(ts)
            break

    # ── HR density (record zprávy) ────────────────────────────────────────────
    hr_samples = 0
    total_records = 0
    for msg in fitfile.get_messages("record"):
        total_records += 1
        vals = msg.get_values()
        if result["start_time"] is None and _first_record_ts is None:
            ts = vals.get("timestamp")
            if isinstance(ts, datetime):
                _first_record_ts = _to_naive_utc(ts)
        hr = vals.get("heart_rate")
        if hr is not None:
            try:
                if int(hr) > 0:
                    hr_samples += 1
            except (TypeError, ValueError):
                pass

    if result["start_time"] is None:
        result["start_time"] = _to_naive_utc(_first_record_ts)

    result["hr_samples"]    = hr_samples
    result["total_records"] = total_records
    result["hr_density"]    = (hr_samples / total_records) if total_records else 0.0

    # ── Sensor type (device_info zprávy) ─────────────────────────────────────
    has_optical_hr = False
    for msg in fitfile.get_messages("device_info"):
        vals = msg.get_values()
        dev_type = vals.get("device_type")
        is_hr_device = (
            dev_type == ANT_DEVICE_TYPE_HR
            or str(dev_type).lower() in ("heart_rate", "heartrate", "120")
        )
        if not is_hr_device:
            continue
        source_type = str(vals.get("source_type", "")).lower()
        ant_number  = vals.get("ant_device_number")
        device_idx  = vals.get("device_index")
        is_external = (
            source_type in ("antplus", "ant", "bluetooth", "bluetooth_low_energy")
            or (ant_number is not None and int(ant_number) > 0)
            or (device_idx is not None and str(device_idx) not in ("0", "creator"))
        )
        if is_external:
            result["sensor_type"] = SENSOR_STRAP
            break
        else:
            has_optical_hr = True
    if result["sensor_type"] == SENSOR_UNKNOWN and has_optical_hr:
        result["sensor_type"] = SENSOR_OPTICAL

    # ── HRV zprávy (R-R intervaly) ─────────────────────────────────────────────
    # Strava exportuje FIT soubory bez device_info, takže sensor_type zůstane
    # SENSOR_UNKNOWN.  Přítomnost HRV zpráv (R-R intervaly) je spolehlivý důkaz
    # hrudního pásu – optický snímač na zápěstí HRV zprávy negeneruje.
    hrv_count = 0
    for msg in fitfile.get_messages("hrv"):
        vals = msg.get_values()
        intervals = vals.get("time")
        if isinstance(intervals, (list, tuple)):
            hrv_count += sum(1 for v in intervals if v is not None)
        elif intervals is not None:
            hrv_count += 1
    result["hrv_count"] = hrv_count

    return result


def get_start_time(fit_path: str) -> Optional[datetime]:
    """Thin wrapper – zachovává zpětnou kompatibilitu. Používej get_fit_metadata()."""
    return get_fit_metadata(fit_path)["start_time"]


def check_hr_density(fit_path: str) -> tuple[int, int, float]:
    """Thin wrapper – zachovává zpětnou kompatibilitu. Používej get_fit_metadata()."""
    m = get_fit_metadata(fit_path)
    return m["hr_samples"], m["total_records"], m["hr_density"]


def detect_hr_sensor_type(fit_path: str) -> str:
    """Thin wrapper – zachovává zpětnou kompatibilitu. Používej get_fit_metadata()."""
    return get_fit_metadata(fit_path)["sensor_type"]


# ─────────────────────────────────────────────────────────────────────────────
# SBĚR SOUBORŮ
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# METADATA CACHE  (metadata_cache.json)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_load() -> dict:
    """Načte metadata_cache.json → dict keyed by absolute path."""
    if os.path.exists(METADATA_CACHE_FILE):
        try:
            with open(METADATA_CACHE_FILE, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("__version__") != CACHE_VERSION:
                log.info(
                    "Metadata cache: verze se liší (v%s → v%d) – přegenerovávám všechna metadata.",
                    data.get("__version__", "?"), CACHE_VERSION,
                )
                return {}
            return {k: v for k, v in data.items() if k != "__version__"}
        except Exception as exc:
            log.warning("Nelze načíst metadata cache (%s) – začínám od nuly.", exc)
    return {}


def _cache_save(cache: dict) -> None:
    """Uloží metadata cache na disk (kompaktní JSON)."""
    try:
        os.makedirs(os.path.dirname(METADATA_CACHE_FILE), exist_ok=True)
        data_to_save = {"__version__": CACHE_VERSION, **cache}
        with open(METADATA_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(data_to_save, fh, separators=(",", ":"))
    except Exception as exc:
        log.warning("Nelze uložit metadata cache: %s", exc)


def _cache_meta_to_json(meta: dict) -> dict:
    """Serializuje meta dict pro JSON (datetime → ISO string)."""
    m = meta.copy()
    if isinstance(m.get("start_time"), datetime):
        m["start_time"] = m["start_time"].isoformat()
    return m


def _cache_meta_from_json(meta: dict) -> dict:
    """Deserializuje meta dict z JSON (ISO string → naive UTC datetime)."""
    m = meta.copy()
    st = m.get("start_time")
    if isinstance(st, str):
        try:
            dt = datetime.fromisoformat(st)
            # Ensure naive UTC regardless of whether the stored string had a
            # UTC offset (Strava exports may have stored tz-aware ISO strings).
            m["start_time"] = _to_naive_utc(dt)
        except ValueError:
            m["start_time"] = None
    return m


def _load_metadata_worker(args: tuple) -> dict:
    """
    Top-level picklable worker – musí být na úrovni modulu pro multiprocessing.
    Otevře FIT soubor jednou přes get_fit_metadata() a vrátí celý entry dict
    se zakešovanými metadaty.  Pokud soubor nelze zpracovat, vrátí entry
    s start_time=None (filtruje se v collect_fit_files).
    """
    path, source, file_size = args
    meta = get_fit_metadata(path)
    return {
        "path":       path,
        "source":     source,
        "file_size":  file_size,
        "start_time": meta["start_time"],
        "_meta":      meta,          # kompletní metadata – používá _pick_winner
    }


def collect_fit_files() -> tuple[list[dict], int, int]:
    """
    Vrací (entries, cache_hits, fresh_count) kde entries je seznam diktů
      {path, source, start_time, file_size, _meta}
    seřazený dle start_time.

    Metadata jsou načtena z metadata_cache.json pokud mtime souboru nezměněn;
    jinak jsou načtena **paralelně** přes multiprocessing.Pool (každý FIT
    soubor otevřen právě jednou). Cache se po každém běhu aktualizuje.
    """
    cache = _cache_load()
    cache_hits = 0

    all_files: list[tuple] = []  # (path, source, file_size, mtime)

    # Garmin: rekurzivně všechny .fit soubory v data/fit/ včetně podsložek,
    # ale bez Strava podsložky (STRAVA_FIT_FOLDER je vnořená uvnitř GARMIN_FIT_FOLDER)
    garmin_pattern = os.path.join(GARMIN_FIT_FOLDER, "**", "*.fit")
    garmin_files = sorted(
        f for f in glob.glob(garmin_pattern, recursive=True)
        if STRAVA_FIT_FOLDER not in f
    )
    log.info("Garmin složka: %d FIT souborů", len(garmin_files))
    for fp in garmin_files:
        all_files.append((fp, SOURCE_GARMIN, os.path.getsize(fp), os.path.getmtime(fp)))

    # Strava originals (rekurzivně)
    strava_pattern = os.path.join(STRAVA_FIT_FOLDER, "**", "*.fit")
    strava_files = sorted(glob.glob(strava_pattern, recursive=True))
    log.info("Strava složka: %d FIT souborů", len(strava_files))
    for fp in strava_files:
        all_files.append((fp, SOURCE_STRAVA, os.path.getsize(fp), os.path.getmtime(fp)))

    total = len(all_files)

    # Split files into cache-hits and files that need fresh parsing
    fresh_args: list[tuple] = []            # (path, source, file_size) for Pool
    fresh_indices: list[int] = []           # positions in all_files needing fresh parse
    cached_entries: dict[int, dict] = {}    # index → pre-built entry (no disk I/O)

    for idx, (fp, source, fsize, mtime) in enumerate(all_files):
        abs_fp = os.path.abspath(fp)
        cached = cache.get(abs_fp)
        if cached is not None and abs(cached["mtime"] - mtime) < 1.0:
            meta = _cache_meta_from_json(cached["meta"])
            cached_entries[idx] = {
                "path":       fp,
                "source":     source,
                "file_size":  fsize,
                "start_time": meta["start_time"],
                "_meta":      meta,
            }
            cache_hits += 1
        else:
            fresh_args.append((fp, source, fsize))
            fresh_indices.append(idx)

    fresh_count = len(fresh_args)
    cpu_count = max(1, mp.cpu_count() - 1)
    log.info(
        "Celkem %d FIT souborů: %d z cache, %d ke zpracování (%d CPU).",
        total, cache_hits, fresh_count, cpu_count,
    )

    # Parallel metadata load for files not in cache
    fresh_index_map: dict[int, int] = {orig: pos for pos, orig in enumerate(fresh_indices)}
    fresh_results: list[dict] = []
    if fresh_args:
        with mp.Pool(processes=cpu_count) as pool:
            fresh_results = pool.map(_load_metadata_worker, fresh_args, chunksize=8)

    # Merge results in original order + update cache for fresh entries
    all_entries: list[dict] = []
    for idx, (fp, source, fsize, mtime) in enumerate(all_files):
        if idx in cached_entries:
            all_entries.append(cached_entries[idx])
        else:
            fi = fresh_index_map[idx]
            entry = fresh_results[fi]
            all_entries.append(entry)
            # Persist to cache
            abs_fp = os.path.abspath(fp)
            cache[abs_fp] = {"mtime": mtime, "meta": _cache_meta_to_json(entry["_meta"])}

    _cache_save(cache)

    valid: list[dict] = []
    for i, e in enumerate(all_entries, 1):
        if e["start_time"] is None:
            log.warning(
                "  [%d/%d] %s – start_time nenalezen, přeskakuji.",
                i, total, os.path.basename(e["path"]),
            )
            continue
        valid.append(e)

    valid.sort(key=lambda x: x["start_time"])
    log.info(
        "Úspěšně načteno %d start-times z %d souborů (cache: %d, čerstvě: %d).",
        len(valid), total, cache_hits, fresh_count,
    )
    return valid, cache_hits, fresh_count


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLIKACE
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(entries: list[dict]) -> list[dict]:
    """
    Projde seřazený seznam aktivit a pokud se dvě překrývají v ±30 min,
    volá _pick_winner() s prioritou: hrudní pás > optika > HR density > velikost.
    """
    if not entries:
        return []

    kept: list[dict] = [entries[0]]
    duplicates_found = 0
    garmin_wins = 0
    strava_wins = 0

    for candidate in entries[1:]:
        last = kept[-1]
        diff = abs((candidate["start_time"] - last["start_time"]).total_seconds())

        if diff <= DEDUP_WINDOW.total_seconds():
            duplicates_found += 1
            winner, reason = _pick_winner(last, candidate)
            loser = candidate if winner is last else last

            date_str = candidate["start_time"].strftime("%Y-%m-%d %H:%M")
            log.info(
                "Duplicita [%s]: %s  "
                "(vítěz: %s, poražený: %s, rozdíl: %.0f min)",
                date_str,
                reason,
                os.path.basename(winner["path"]),
                os.path.basename(loser["path"]),
                diff / 60.0,
            )

            if winner["source"] == SOURCE_GARMIN:
                garmin_wins += 1
            else:
                strava_wins += 1

            if winner is candidate:
                kept[-1] = candidate
        else:
            kept.append(candidate)

    log.info(
        "Deduplikace: %d duplicit | Garmin zvítězil %dx, Strava %dx | "
        "%d unikátních aktivit.",
        duplicates_found, garmin_wins, strava_wins, len(kept),
    )
    return kept


def _sensor_label(sensor: str) -> str:
    return {SENSOR_STRAP: "Hrudní pás", SENSOR_OPTICAL: "Optika", SENSOR_UNKNOWN: "Neznámý"}[sensor]


def _pick_winner(a: dict, b: dict) -> tuple[dict, str]:
    """
    Rozhodne, který záznam ponechat. Vrací (winner, reason_string).

    Prioritní pořadí:
      0) Pojistka Integrity – pokud jeden soubor má >25 % více záznamů,
         vyhrává automaticky (druhý je zřejmě oříznutý / poškozený).
      1) Snímač – hrudní pás > optický snímač
         Výjimka: pokud je jeden snímač SENSOR_UNKNOWN (časté v Strava exportech),
         přeskočí sensor-rank a rovnou rozhodne HR density.
      2) HR density – vyšší density (≥ 90 %)
      3) Velikost souboru

    Každý soubor je otevřen jen jednou (get_fit_metadata) – 66 % méně I/O.
    """
    # ── Použij zakešovaná metadata z collect_fit_files (nulové I/O) ────────────
    meta_a = a["_meta"]
    meta_b = b["_meta"]

    a_total   = meta_a["total_records"]
    b_total   = meta_b["total_records"]
    a_sensor  = meta_a["sensor_type"]
    b_sensor  = meta_b["sensor_type"]
    a_hr      = meta_a["hr_samples"]
    b_hr      = meta_b["hr_samples"]
    a_density = meta_a["hr_density"]
    b_density = meta_b["hr_density"]

    # ── Krok 0: Pojistka Integrity (records_count) ──────────────────────────
    max_total = max(a_total, b_total)
    min_total = min(a_total, b_total)

    if max_total > 0 and min_total < max_total * (1 - INTEGRITY_DIFF_PCT):
        winner = a if a_total >= b_total else b
        loser  = b if a_total >= b_total else a

        def _fmt(n: int) -> str:
            return f"{n/1000:.1f}k" if n >= 1000 else str(n)

        # Distinguish Smart Recording (few points, good HR coverage) from
        # genuine data truncation/corruption (few points AND poor HR coverage).
        loser_density = (meta_a["hr_density"] if a_total < b_total
                         else meta_b["hr_density"])
        ratio = min_total / max_total
        if ratio < 0.5 and loser_density >= 0.5:
            integrity_label = "Rozdíl hustoty záznamu (Smart Recording vs 1-sec)"
        else:
            integrity_label = "Pojistka Integrity (možné zkrácení/poškození dat)"

        reason = (
            f"{winner['source'].upper()} má výrazně více dat "
            f"({_fmt(max_total)} vs {_fmt(min_total)} bodů) "
            f"-> Vybírám {winner['source'].upper()} [{integrity_label}]"
        )
        return winner, reason

    # ── Krok 1+: Různé zdroje ───────────────────────────────────────────
    if a["source"] != b["source"]:
        garmin = a if a["source"] == SOURCE_GARMIN else b
        strava = b if a["source"] == SOURCE_GARMIN else a

        g_sensor  = meta_a["sensor_type"] if a["source"] == SOURCE_GARMIN else meta_b["sensor_type"]
        s_sensor  = meta_b["sensor_type"] if a["source"] == SOURCE_GARMIN else meta_a["sensor_type"]
        g_hr      = meta_a["hr_samples"]  if a["source"] == SOURCE_GARMIN else meta_b["hr_samples"]
        s_hr      = meta_b["hr_samples"]  if a["source"] == SOURCE_GARMIN else meta_a["hr_samples"]
        g_total   = meta_a["total_records"] if a["source"] == SOURCE_GARMIN else meta_b["total_records"]
        s_total   = meta_b["total_records"] if a["source"] == SOURCE_GARMIN else meta_a["total_records"]
        g_density = meta_a["hr_density"]  if a["source"] == SOURCE_GARMIN else meta_b["hr_density"]
        s_density = meta_b["hr_density"]  if a["source"] == SOURCE_GARMIN else meta_a["hr_density"]
        g_label   = _sensor_label(g_sensor)
        s_label   = _sensor_label(s_sensor)

        # Případ A: Garmin má hrudní pás → vyjdi zaždy s Garminem
        if g_sensor == SENSOR_STRAP:
            reason = (
                f"Garmin ({g_label}) vs Strava ({s_label}) "
                f"-> Ponechávám GARMIN (hrudní pás + Running Dynamics)"
            )
            return garmin, reason

        # Případ B: Garmin nemá pás, Strava má pás → Strava
        if s_sensor == SENSOR_STRAP:
            reason = (
                f"Garmin ({g_label}) vs Strava ({s_label}) "
                f"-> Vybírám STRAVU kvůli přesnějším datům z hrudního pásu"
            )
            return strava, reason

        # Případ B2: device_info chybí (typicky Strava export), ale HRV zprávy
        # (R-R intervaly) prozradí hrudní pás.  Optický snímač na zápěstí HRV
        # zprávy negeneruje, takže přítomnost ≥ HRV_MIN_INTERVALS intervalů je
        # spolehlivý proxy pro SENSOR_STRAP i bez device_info sekce.
        g_hrv = (meta_a if a["source"] == SOURCE_GARMIN else meta_b).get("hrv_count", 0)
        s_hrv = (meta_b if a["source"] == SOURCE_GARMIN else meta_a).get("hrv_count", 0)
        if g_hrv >= HRV_MIN_INTERVALS and s_hrv < HRV_MIN_INTERVALS:
            reason = (
                f"Garmin ({g_label}) vs Strava ({s_label}), "
                f"hrudní pás detekován přes HRV zprávy (R-R intervaly): "
                f"Garmin {g_hrv} vs Strava {s_hrv} "
                f"-> Ponechávám GARMIN (HRV = hrudní pás)"
            )
            return garmin, reason
        if s_hrv >= HRV_MIN_INTERVALS and g_hrv < HRV_MIN_INTERVALS:
            reason = (
                f"Garmin ({g_label}) vs Strava ({s_label}), "
                f"hrudní pás detekován přes HRV zprávy (R-R intervaly): "
                f"Strava {s_hrv} vs Garmin {g_hrv} "
                f"-> Vybírám STRAVA (HRV = hrudní pás)"
            )
            return strava, reason

        # Případ C: Ani jeden nemá pás → Garmin jako primární zdroj;
        #           pojistka: HR density
        if g_density >= HR_DENSITY_THRESHOLD:
            reason = (
                f"Garmin ({g_label}) vs Strava ({s_label}), "
                f"oba bez pásu; Garmin HR density "
                f"{g_hr}/{g_total} = {g_density:.0%} "
                f"-> Ponechávám GARMIN"
            )
            return garmin, reason
        else:
            reason = (
                f"Garmin ({g_label}) vs Strava ({s_label}), "
                f"oba bez pásu; Garmin HR nedostatečný "
                f"({g_hr}/{g_total} = {g_density:.0%} < {HR_DENSITY_THRESHOLD:.0%}), "
                f"Strava {s_density:.0%} "
                f"-> Přepínám na STRAVA"
            )
            return strava, reason

    # ── Stejný zdroj ──────────────────────────────────────────────────────────
    _sensor_rank = {SENSOR_STRAP: 2, SENSOR_OPTICAL: 1, SENSOR_UNKNOWN: 0}

    if a_sensor != b_sensor:
        # Pokud je jeden snímač UNKNOWN (typicky Strava export bez device_info),
        # přeskočíme sensor-rank a rozhodujeme primárně podle HR density.
        # Automatické favorizování "known" snímače by mohlo zvýhodňovat soubor
        # s horší kvalitou dat jen proto, že dokázal určit typ optiky.
        if a_sensor == SENSOR_UNKNOWN or b_sensor == SENSOR_UNKNOWN:
            # If the known sensor is a strap it wins outright regardless of density
            # (UNKNOWN vs STRAP → strap; UNKNOWN vs OPTICAL → fall through to density).
            known_sensor = b_sensor if a_sensor == SENSOR_UNKNOWN else a_sensor
            if known_sensor == SENSOR_STRAP:
                winner = b if a_sensor == SENSOR_UNKNOWN else a
                reason = (
                    f"Stejný zdroj ({a['source'].upper()}), "
                    f"jeden snímač neznámý; druhý je hrudní pás "
                    f"-> Vybírám hrudní pás"
                )
                return winner, reason
            # UNKNOWN vs OPTICAL: fall through to HR density comparison
        else:
            winner = a if _sensor_rank[a_sensor] >= _sensor_rank[b_sensor] else b
            reason = (
                f"Stejný zdroj ({a['source'].upper()}), lepší snímač: "
                f"{_sensor_label(a_sensor)} vs {_sensor_label(b_sensor)}"
            )
            return winner, reason

    # HRV fallback pro stejný zdroj – oba snímače UNKNOWN (např. dvě Strava verze
    # téže aktivity, kde obě postrádají device_info).
    if a_sensor == SENSOR_UNKNOWN and b_sensor == SENSOR_UNKNOWN:
        a_hrv = meta_a.get("hrv_count", 0)
        b_hrv = meta_b.get("hrv_count", 0)
        if a_hrv >= HRV_MIN_INTERVALS and b_hrv < HRV_MIN_INTERVALS:
            reason = (
                f"Stejný zdroj ({a['source'].upper()}), oba snímače neznámé; "
                f"hrudní pás detekován přes HRV: {a_hrv} vs {b_hrv} R-R intervalů "
                f"-> {os.path.basename(a['path'])}"
            )
            return a, reason
        if b_hrv >= HRV_MIN_INTERVALS and a_hrv < HRV_MIN_INTERVALS:
            reason = (
                f"Stejný zdroj ({a['source'].upper()}), oba snímače neznámé; "
                f"hrudní pás detekován přes HRV: {b_hrv} vs {a_hrv} R-R intervalů "
                f"-> {os.path.basename(b['path'])}"
            )
            return b, reason

    # Stejný (nebo nerozlišitelný) snímač → HR density
    if a_density != b_density:
        winner = a if a_density >= b_density else b
        reason = (
            f"Stejný zdroj ({a['source'].upper()}), "
            f"snímač shodný ({_sensor_label(a_sensor)}), vyšší HR density: "
            f"{max(a_density, b_density):.0%} vs {min(a_density, b_density):.0%}"
        )
        return winner, reason

    winner = a if a["file_size"] >= b["file_size"] else b
    reason = (
        f"Stejný zdroj ({a['source'].upper()}), "
        f"snímač shodný ({_sensor_label(a_sensor)}), shodná HR density ({a_density:.0%}), "
        f"větší soubor: {os.path.basename(winner['path'])}"
    )
    return winner, reason


# ─────────────────────────────────────────────────────────────────────────────
# HLAVNÍ FUNKCE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 72)
    log.info("MASTER REBUILD – Čistá tréninková databáze (Garmin + Strava)")
    log.info("  MAX_HR=%d bpm, baseline_RHR=%d bpm", MAX_HR, BASELINE_RHR)
    log.info("  Deduplikační okno: ±%d min | HR density: %.0f%% | Integrity: ±%.0f%% | ANT+ HR: %d",
             int(DEDUP_WINDOW.total_seconds() / 60), HR_DENSITY_THRESHOLD * 100,
             INTEGRITY_DIFF_PCT * 100, ANT_DEVICE_TYPE_HR)
    log.info("=" * 72)

    os.makedirs(SUMMARY_FOLDER, exist_ok=True)

    # 1. Sběr a deduplikace
    all_entries, _cache_hits, _fresh_count = collect_fit_files()

    unique_entries = deduplicate(all_entries)

    if not unique_entries:
        log.error("Žádné aktivity k zpracování. Ukončuji.")
        return

    # ── Inkrementální režim: přeskoč již zpracované aktivity ────────────────
    existing_ids: set[str] = set()
    if os.path.isfile(MASTER_SUMMARY_CSV):
        try:
            with open(MASTER_SUMMARY_CSV, "r", encoding="utf-8") as ef:
                reader = csv.DictReader(ef)
                for row in reader:
                    aid = row.get("activity_id", "").strip()
                    if aid:
                        existing_ids.add(aid)
            log.info("Nalezeno %d již zpracovaných aktivit v master summary.", len(existing_ids))
        except Exception as exc:
            log.warning("Nelze načíst existující IDs z %s: %s – zpracuji vše.", MASTER_SUMMARY_CSV, exc)
            existing_ids = set()

    new_entries = [
        e for e in unique_entries
        if extract_activity_id(e["path"]) not in existing_ids
    ]
    skipped = len(unique_entries) - len(new_entries)

    if not new_entries:
        log.info("Všech %d aktivit již zpracováno – nic nového k updatu.", len(unique_entries))
        return

    # 2. Zpracování – streaming do master CSV (append)
    summary_rows: list[dict] = []
    processed = failed = 0

    log.info("─" * 72)
    log.info("Zahajuji zpracování %d nových aktivit (%d přeskočeno)...",
             len(new_entries), skipped)
    log.info("─" * 72)

    paths = [e["path"] for e in new_entries]
    cpu_count = max(1, mp.cpu_count() - 1)
    log.info("Paralelní parsování FIT souborů (%d CPU)...", cpu_count)

    master_summary_cols = SUMMARY_COLS + ["source"]

    def _csv_needs_header(path: str) -> bool:
        return not os.path.isfile(path) or os.path.getsize(path) == 0

    write_highres_header = _csv_needs_header(MASTER_HIGHRES_CSV)
    write_summary_header = _csv_needs_header(MASTER_SUMMARY_CSV)

    with (
        open(MASTER_HIGHRES_CSV, "a", newline="", encoding="utf-8") as fh,
        open(MASTER_SUMMARY_CSV, "a", newline="", encoding="utf-8") as sf,
    ):
        hr_writer = csv.DictWriter(fh, fieldnames=HIGHRES_COLS)
        sm_writer = csv.DictWriter(sf, fieldnames=master_summary_cols)

        if write_highres_header:
            hr_writer.writeheader()
        if write_summary_header:
            sm_writer.writeheader()

        # pool.imap preserves input order; each worker returns
        # (summary_dict, highres_rows) or None on failure.
        with mp.Pool(processes=cpu_count) as pool:
            for i, (entry, result) in enumerate(
                zip(new_entries, pool.imap(parse_fit_to_memory, paths, chunksize=4)),
                1,
            ):
                src_tag = f"[{entry['source'].upper():6s}]"
                log.info("─── [%d/%d] %s %s",
                         i, len(new_entries), src_tag,
                         os.path.basename(entry["path"]))

                if result is not None:
                    summary, highres_rows = result
                    for row in highres_rows:
                        hr_writer.writerow(row)
                    summary["source"] = entry["source"]
                    sm_writer.writerow(summary)
                    summary_rows.append(summary)
                    processed += 1
                else:
                    failed += 1

                if i % 10 == 0:
                    fh.flush()
                    sf.flush()

    # 4. Statistiky
    # Počet nových metrik (sloupce přidané nad rámec původních 16)
    NEW_METRIC_NAMES = [
        "distance_km", "ascent_m", "descent_m", "avg_speed_kmh", "max_speed_kmh",
        "calories", "avg_cadence", "max_cadence", "avg_temp", "max_temp",
        "training_effect_aerobic", "training_effect_anaerobic", "vo2_max",
    ]
    new_metrics_count = len(NEW_METRIC_NAMES)

    log.info("=" * 72)
    log.info("DOKONČENO")
    log.info("  Zpracováno: %d aktivit  |  Chyb: %d  |  Přeskočeno: %d",
             processed, failed, skipped)

    garmin_count = sum(1 for r in summary_rows if r.get("source") == SOURCE_GARMIN)
    strava_count = sum(1 for r in summary_rows if r.get("source") == SOURCE_STRAVA)
    log.info("  Z toho Garmin: %d  |  Strava: %d", garmin_count, strava_count)

    if summary_rows:
        trimps = [r["total_trimp"] for r in summary_rows if r["total_trimp"]]
        if trimps:
            log.info("  Celkový TRIMP:           %.1f", sum(trimps))
            log.info("  Průměrný TRIMP/aktivita: %.1f", mean(trimps))

    log.info("  High-res CSV: %s", MASTER_HIGHRES_CSV)
    log.info("  Summary CSV:  %s", MASTER_SUMMARY_CSV)
    log.info("")
    log.info("Upgrade dokončen: Extrahováno %d nových metrik u %d aktivit.",
             new_metrics_count, processed)
    log.info("  Metadata cache: %d soubor(ů) z cache, %d zpracováno znovu.",
             _cache_hits, _fresh_count)
    log.info("=" * 72)

    # 5. Merge VO2 Max from external CSV (Garmin FIT files rarely contain it)
    _merge_vo2_max(MASTER_SUMMARY_CSV)


def _merge_vo2_max(summary_csv: str) -> None:
    """
    Napojí sloupec vo2_max ze souboru data/summaries/vo2_max.csv do master
    summary CSV pomocí LEFT JOIN přes sloupec 'date'.

    Formát vo2_max.csv (minimální):
        date,vo2_max
        2024-03-15,58.2
        2024-05-01,57.8
        ...

    Pravidla:
    • Hodnoty v CSV mají přednost před hodnotami z FIT souborů (vždy "").
    • Pokud pro daný den existuje více aktivit, stejná hodnota se aplikuje na všechny.
    • Pokud vo2_max.csv neexistuje, funkce tiše skončí bez chyby.
    • Pokud je vo2_max.csv prázdný nebo nemá správné sloupce, zapíše varování.
    """
    import pandas as pd  # local import – master_rebuild otherwise uses only stdlib

    VO2_CSV = os.path.join(SUMMARY_FOLDER, "vo2_max.csv")
    if not os.path.isfile(VO2_CSV):
        log.info("[VO2 Max] %s nenalezen – přeskočeno.", VO2_CSV)
        return
    if not os.path.isfile(summary_csv):
        log.warning("[VO2 Max] Summary CSV %s neexistuje – přeskočeno.", summary_csv)
        return

    try:
        vo2_df = pd.read_csv(VO2_CSV, dtype=str)
        if "date" not in vo2_df.columns or "vo2_max" not in vo2_df.columns:
            log.warning("[VO2 Max] %s musí obsahovat sloupce 'date' a 'vo2_max'.", VO2_CSV)
            return

        vo2_df = vo2_df[["date", "vo2_max"]].copy()
        vo2_df["date"] = pd.to_datetime(vo2_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        vo2_df = vo2_df.dropna(subset=["date"]).drop_duplicates("date")

        summary_df = pd.read_csv(summary_csv, dtype=str)
        if "date" not in summary_df.columns:
            log.warning("[VO2 Max] Summary CSV nemá sloupec 'date' – přeskočeno.")
            return

        summary_df["date"] = pd.to_datetime(summary_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Drop the empty vo2_max column that fit_parser writes, then merge
        if "vo2_max" in summary_df.columns:
            summary_df = summary_df.drop(columns=["vo2_max"])

        merged = summary_df.merge(vo2_df, on="date", how="left")

        # Preserve original column order: insert vo2_max after training_effect_anaerobic
        cols = list(merged.columns)
        if "vo2_max" in cols and "training_effect_anaerobic" in cols:
            cols.remove("vo2_max")
            idx = cols.index("training_effect_anaerobic") + 1
            cols.insert(idx, "vo2_max")
            merged = merged[cols]

        merged.to_csv(summary_csv, index=False)
        filled = merged["vo2_max"].notna().sum()
        log.info("[VO2 Max] Doplněno %d hodnot VO2 Max do %s.", filled, summary_csv)

    except Exception as exc:
        log.error("[VO2 Max] Chyba při merge: %s", exc)


if __name__ == "__main__":
    main()
