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
from datetime import datetime, timedelta, timezone as _tz
from statistics import mean
from typing import Optional

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fitparse import FitFile

from config.settings import (
    DEDUP_WINDOW_MIN, HR_DENSITY_THRESHOLD,
    ANT_DEVICE_TYPE_HR, INTEGRITY_DIFF_PCT,
)

# Import sdílené logiky z fit_to_highres_csv.py
from fit_to_highres_csv import (
    BASELINE_RHR,
    HIGHRES_COLS,
    MAX_HR,
    SUMMARY_COLS,
    SUMMARY_FOLDER,
    extract_activity_id,
    parse_fit_file,
    parse_fit_to_memory,
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
# TIMEZONE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalizuje datetime na naivní UTC čas (bez tzinfo).

    Garmin FIT soubory ukládají časy v UTC jako naivní datetime.
    Strava exporty někdy obsahují tz-aware datetime v lokálním čase.
    Tato funkce zajistí, že před deduplikací jsou oba zdroje vždy v naivním UTC,
    čímž se předejde chybám při porovnávání (±30 min okno) způsobeným
    timezone offsetem.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(_tz.utc).replace(tzinfo=None)
    return dt


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
                return json.load(fh)
        except Exception as exc:
            log.warning("Nelze načíst metadata cache (%s) – začínám od nuly.", exc)
    return {}


def _cache_save(cache: dict) -> None:
    """Uloží metadata cache na disk (kompaktní JSON)."""
    try:
        os.makedirs(os.path.dirname(METADATA_CACHE_FILE), exist_ok=True)
        with open(METADATA_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, separators=(",", ":"))
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

    # Garmin: jen soubory přímo v data/fit/ (NE v podsložkách)
    garmin_pattern = os.path.join(GARMIN_FIT_FOLDER, "*.fit")
    garmin_files = sorted(glob.glob(garmin_pattern))
    log.info("Garmin složka: %d FIT souborů", len(garmin_files))
    for fp in garmin_files:
        all_files.append((fp, SOURCE_GARMIN, os.path.getsize(fp), os.path.getmtime(fp)))

    # Strava originals
    strava_pattern = os.path.join(STRAVA_FIT_FOLDER, "*.fit")
    strava_files = sorted(glob.glob(strava_pattern))
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

    # 2. Zpracování – streaming do master CSV
    summary_rows: list[dict] = []
    processed = failed = 0

    log.info("─" * 72)
    log.info("Zahajuji zpracování %d unikátních aktivit...", len(unique_entries))
    log.info("─" * 72)

    paths = [e["path"] for e in unique_entries]
    cpu_count = max(1, mp.cpu_count() - 1)
    log.info("Paralelní parsování FIT souborů (%d CPU)...", cpu_count)

    with open(MASTER_HIGHRES_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HIGHRES_COLS)
        writer.writeheader()

        # pool.imap preserves input order; each worker returns
        # (summary_dict, highres_rows) or None on failure.
        with mp.Pool(processes=cpu_count) as pool:
            for i, (entry, result) in enumerate(
                zip(unique_entries, pool.imap(parse_fit_to_memory, paths, chunksize=4)),
                1,
            ):
                src_tag = f"[{entry['source'].upper():6s}]"
                log.info("─── [%d/%d] %s %s",
                         i, len(unique_entries), src_tag,
                         os.path.basename(entry["path"]))

                if result is not None:
                    summary, highres_rows = result
                    for row in highres_rows:
                        writer.writerow(row)
                    summary["source"] = entry["source"]
                    summary_rows.append(summary)
                    processed += 1
                else:
                    failed += 1

                if i % 10 == 0:
                    fh.flush()

    # 3. Summary CSV (přidáme sloupec 'source')
    master_summary_cols = SUMMARY_COLS + ["source"]

    with open(MASTER_SUMMARY_CSV, "w", newline="", encoding="utf-8") as sf:
        ws = csv.DictWriter(sf, fieldnames=master_summary_cols)
        ws.writeheader()
        ws.writerows(summary_rows)

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
    log.info("  Zpracováno: %d aktivit  |  Chyb: %d", processed, failed)

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


if __name__ == "__main__":
    main()
