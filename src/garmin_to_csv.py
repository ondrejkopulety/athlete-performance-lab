#!/usr/bin/env python3
"""
Robustní skript pro stahování dat z Garmin Connect API.

Funkce:
    - Automatické čištění adresářů data/summaries/ a data/raw/ na začátku
    - Stahování aktivit, HRV, VO2 Max, Daily Health a Training Readiness
    - Speciální zpracování spánku přes endpoint /sleep-service/sleep/dailySleepData
    - Náhodné pauzy mezi požadavky (2-4 sekundy)
    - Real-time logování průběhu synchronizace
    - Automatická deduplikace CSV dat na základě sloupce 'date'
"""

import os
import sys
import json
import csv
import shutil
import time
import random
import logging
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from garminconnect import Garmin
from dotenv import load_dotenv

from config.settings import INITIAL_BACKFILL_DAYS

# Cesty k datovým složkám
DATA_DIR = Path("data")
SUMMARIES_DIR = DATA_DIR / "summaries"
RAW_DIR = DATA_DIR / "raw"
FIT_DIR = DATA_DIR / "fit"

# CSV soubory
CSV_FILES = {
    "activities": SUMMARIES_DIR / "activities.csv",
    "hrv": SUMMARIES_DIR / "hrv.csv",
    "vo2_max": SUMMARIES_DIR / "vo2_max.csv",
    "daily_health": SUMMARIES_DIR / "daily_health.csv",
    "training_readiness": SUMMARIES_DIR / "training_readiness.csv",
    "training_status": SUMMARIES_DIR / "training_status.csv",
    "load_focus": SUMMARIES_DIR / "load_focus.csv",
    "lactate_threshold": SUMMARIES_DIR / "lactate_threshold.csv",
    "heart_rate_summary": SUMMARIES_DIR / "heart_rate_summary.csv",
    "heart_rate_details": SUMMARIES_DIR / "heart_rate_details.csv",
    "movement": SUMMARIES_DIR / "movement.csv",
    "intensity": SUMMARIES_DIR / "intensity.csv",
    "sleep": SUMMARIES_DIR / "sleep.csv",
}

# Konfigurace loggingu
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# POMOCNÉ FUNKCE
# ============================================================================

def wipe_data_directories():
    """
    Smaže všechny soubory ve složkách data/summaries/ a data/raw/.
    Jedná se o čistý start pro novou synchronizaci.
    """
    logger.info("[INFO] Čišťuji datové adresáře...")
    
    for directory in [SUMMARIES_DIR, RAW_DIR]:
        if directory.exists():
            try:
                shutil.rmtree(directory)
                logger.info(f"[INFO] Smazán adresář: {directory}")
            except Exception as e:
                logger.warning(f"[WARN] Chyba při mazání {directory}: {e}")
    
    # Vytvoř adresáře znovu, aby byla připravena struktura
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("[INFO] Datové adresáře jsou připraveny")


def random_sleep(min_seconds: int = 2, max_seconds: int = 4) -> None:
    """Náhodná pauza mezi požadavky, aby nás Garmin nezablokoval."""
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)


def append_or_update_csv(csv_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Přidá nebo aktualizuje CSV soubor.
    
    Pokud soubor existuje, sloučí nová data se starými a odstraní duplicity
    podle sloupce 'date'. Novější data přepíší starší.
    
    Args:
        csv_path: Cesta k CSV souboru
        data: Seznam slovníků reprezentujících řádky
    """
    if not data:
        return
    
    csv_path = Path(csv_path)
    
    # Pokud soubor existuje, načti jeho obsah
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            df_new = pd.DataFrame(data)
            
            # Sloučení - nová data mají přednost
            # Používáme 'date' jako klíč pro deduplikaci
            if 'date' in df_existing.columns and 'date' in df_new.columns:
                # Znormalizuj datumové sloupce
                df_existing['date'] = pd.to_datetime(df_existing['date'])
                df_new['date'] = pd.to_datetime(df_new['date'])
                
                # Odstraň staré řádky, které jsou v nových datech
                df_merged = pd.concat([df_existing, df_new], ignore_index=True)
                df_merged = df_merged.drop_duplicates(subset=['date'], keep='last')
                df_merged = df_merged.sort_values('date')
                
                # Konvertuj zpět na string, pokud to bylo původně
                df_merged['date'] = df_merged['date'].astype(str)
            else:
                # Pokud není 'date' sloupec, prostě přidej nová data
                df_merged = pd.concat([df_existing, pd.DataFrame(data)], ignore_index=True)
            
            df_merged.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"[WARN] Chyba při slučování CSV {csv_path}: {e}")
            # Fallback: přepsat soubor
            pd.DataFrame(data).to_csv(csv_path, index=False)
    else:
        # Vytvoř nový soubor
        pd.DataFrame(data).to_csv(csv_path, index=False)


def load_credentials():
    """Načte přihlašovací údaje z .env souboru."""
    load_dotenv()

    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    if not all([email, password]):
        logger.error("[ERROR] Chybí přihlašovací údaje v .env souboru (GARMIN_EMAIL, GARMIN_PASSWORD)")
        sys.exit(1)

    # display_name se nadále nepoužívá – garminconnect ho získá automaticky po přihlášení
    display_name = os.getenv("GARMIN_DISPLAY_NAME", "")
    return email, password, display_name


def save_raw_response(category: str, endpoint: str, response: Any) -> None:
    """
    Uloží kompletní nezpracovanou odpověď z API do data/raw/ pro debugging.
    
    Args:
        category: Kategorie (vo2_max, training_status, training_readiness)
        endpoint: Endpoint URL
        response: Kompletní odpověď
    """
    try:
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = raw_dir / f"{category}_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "endpoint": endpoint,
                "response": response,
                "response_type": type(response).__name__,
            }, f, indent=2, default=str)
        
        logger.debug(f"[DEBUG] Raw response saved: {filename}")
    except Exception as e:
        logger.debug(f"[DEBUG] Failed to save raw response: {str(e)[:50]}")


def find_value_recursive(obj: Any, key_pattern: str, depth: int = 0, max_depth: int = 10) -> List[Any]:
    """
    Hledá hodnoty v JSON objektu rekurzivně podle klíčového vzoru.
    
    Args:
        obj: Objekt k prohledávání
        key_pattern: Vzor klíče (case-insensitive)
        depth: Aktuální hloubka rekurze
        max_depth: Maximální hloubka rekurze
    
    Returns:
        Seznam nalezených hodnot
    """
    results = []
    key_pattern_lower = key_pattern.lower()
    
    if depth > max_depth:
        return results
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() == key_pattern_lower or key_pattern_lower in key.lower():
                results.append(value)
            
            if isinstance(value, (dict, list)):
                results.extend(find_value_recursive(value, key_pattern, depth + 1, max_depth))
    
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_value_recursive(item, key_pattern, depth + 1, max_depth))
    
    return results


# ============================================================================
# GARMIN API FUNKCE
# ============================================================================

TOKEN_STORE = ".garminconnect"


def authenticate(email: str, password: str) -> Garmin:
    """
    Ověření se Garmin Connect API s persistencí tokenů (kompatibilní s garminconnect v0.2.x).

    Nejdříve se pokusí načíst uloženou session z TOKEN_STORE adresáře.
    Pokud selže (tokeny chybí nebo vypršely), provede plné přihlášení a uloží
    nové tokeny pomocí garth.save() pro příští spuštění.
    """
    logger.info("[INFO] Ověřuji se do Garmin Connect...")

    token_dir = Path(TOKEN_STORE)

    # Pokus 1: Obnova session z uložených tokenů (bez nového přihlašování)
    if token_dir.exists():
        try:
            logger.info(f"[INFO] Zkouším načíst session z {TOKEN_STORE}...")
            api = Garmin(email, password)
            api.garth.load(TOKEN_STORE)
            # Ověř platnost session dotazem na display_name
            _ = api.display_name
            logger.info("[INFO] Session úspěšně obnovena z uložených tokenů")
            return api
        except Exception as token_err:
            logger.info(f"[INFO] Uložená session je neplatná ({type(token_err).__name__}), provádím nové přihlášení...")

    # Pokus 2: Plné přihlášení s uložením tokenů pro příště
    try:
        api = Garmin(email, password)
        api.login()
        try:
            token_dir.mkdir(parents=True, exist_ok=True)
            api.garth.save(TOKEN_STORE)
            logger.info(f"[INFO] Tokeny uloženy do {TOKEN_STORE} pro příští spuštění")
        except Exception as save_err:
            logger.warning(f"[WARN] Nepodařilo se uložit tokeny: {save_err}")
        logger.info(f"[INFO] Přihlášení úspěšné (display_name={api.display_name})")
        return api
    except Exception as e:
        logger.error(f"[ERROR] Přihlášení selhalo: {e}", exc_info=True)
        sys.exit(1)


def sync_activities(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje aktivitní data pomocí garmin_obj.get_activities_by_date().

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji AKTIVITY pro datumový rozsah {start_date} až {end_date}...")

    try:
        activities = garmin_obj.get_activities_by_date(start_date, end_date)

        activities_data = []
        fit_count = 0

        if activities and isinstance(activities, list):
            for activity in activities:
                # get_activities_by_date vrací startTimeLocal jako "YYYY-MM-DD HH:MM:SS"
                start_time_local = activity.get("startTimeLocal", "")
                activity_date_str = start_time_local[:10] if start_time_local else ""

                if not activity_date_str:
                    continue

                activity_id = activity.get("activityId")

                activities_data.append({
                    "date": activity_date_str,
                    "activity_id": activity_id,
                    "activity_name": activity.get("activityName", ""),
                    "activity_type_id": activity.get("activityType", {}).get("typeId", ""),
                    "duration_seconds": activity.get("duration", 0),
                    "distance_meters": activity.get("distance", 0),
                    "calories": activity.get("calories", 0),
                    "avg_heart_rate": activity.get("averageHR", 0),
                    "max_heart_rate": activity.get("maxHR", 0),
                    "start_time": start_time_local,
                })

                # Pokus se stáhnout a rozbalit FIT soubor
                try:
                    fit_data = garmin_obj.download_activity(
                        activity_id, dl_fmt=garmin_obj.ActivityDownloadFormat.ORIGINAL
                    )
                    if fit_data:
                        if zipfile.is_zipfile(io.BytesIO(fit_data)):
                            # Rozbal ZIP v paměti, ulož pouze .fit soubory
                            with zipfile.ZipFile(io.BytesIO(fit_data)) as zf:
                                for name in zf.namelist():
                                    if name.lower().endswith(".fit"):
                                        fit_path = FIT_DIR / f"activity_{activity_id}.fit"
                                        fit_path.write_bytes(zf.read(name))
                                        fit_count += 1
                        else:
                            # Stažená data jsou přímo FIT soubor
                            fit_path = FIT_DIR / f"activity_{activity_id}.fit"
                            fit_path.write_bytes(fit_data)
                            fit_count += 1
                except Exception as e:
                    logger.warning(f"[WARN] FIT soubor pro aktivitu {activity_id}: {str(e)[:50]}")

                logger.info(f"[INFO] Synchronizuji AKTIVITU pro datum {activity_date_str}")
                random_sleep()

        if activities_data:
            append_or_update_csv(CSV_FILES["activities"], activities_data)
            logger.info(f"[INFO] Synchronizovány AKTIVITY: {len(activities_data)} záznamů, {fit_count} FIT souborů")
        else:
            logger.info("[INFO] Žádné aktivity k synchronizaci")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci aktivit: {e}")


def sync_hrv(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje HRV data pomocí garmin_obj.get_hrv_data(date_str).

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji HRV pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        hrv_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                hrv_response = garmin_obj.get_hrv_data(date_str)

                if hrv_response and isinstance(hrv_response, dict):
                    hrv_summary = hrv_response.get("hrvSummary", {})
                    hrv_readings = hrv_response.get("hrvReadings", [])

                    # HRV má smysluplná data, pokud je seznam HRV odečtů nebo máme summary
                    if hrv_readings or (isinstance(hrv_summary, dict) and hrv_summary.get("lastNightAvg")):
                        hrv_data.append({
                            "date": date_str,
                            "weekly_avg": hrv_summary.get("weeklyAvg", 0),
                            "last_night_avg": hrv_summary.get("lastNightAvg", 0),
                            "last_night_5min_high": hrv_summary.get("lastNight5MinHigh", 0),
                            "status": hrv_summary.get("status", ""),
                            "feedback_text": hrv_summary.get("feedbackPhrase", ""),
                            "sample_count": len(hrv_readings),
                        })
                        logger.info(f"[INFO] Synchronizuji HRV pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.warning(f"[WARN] Chyba při stahování HRV pro {date_str}: {str(e)[:50]}")

            current_date += timedelta(days=1)

        if hrv_data:
            append_or_update_csv(CSV_FILES["hrv"], hrv_data)
            logger.info(f"[INFO] Synchronizovány HRV data: {len(hrv_data)} záznamů")
        else:
            logger.info("[INFO] Žádná HRV data k synchronizaci")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci HRV: {e}")


def sync_vo2_max(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje VO2 Max data pomocí garmin_obj.get_max_metrics(date_str).

    Hodnota VO2 Max se nachází v: response['generic']['vo2MaxPreciseValue']

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji VO2 MAX pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        vo2_data = []

        def _fetch_and_parse_vo2(date_str: str) -> Optional[float]:
            """Stáhne a naparsuje VO2 Max pro jeden den; vrátí hodnotu nebo None."""
            try:
                max_metrics = garmin_obj.get_max_metrics(date_str)
                if max_metrics:
                    save_raw_response(f"vo2_max_{date_str}", "get_max_metrics", max_metrics)
                if isinstance(max_metrics, dict):
                    items = [max_metrics]
                elif isinstance(max_metrics, list):
                    items = max_metrics
                else:
                    return None
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    generic = item.get("generic", {})
                    vo2_value = generic.get("vo2MaxPreciseValue") if isinstance(generic, dict) else None
                    if not vo2_value:
                        found = find_value_recursive(item, "vo2MaxPreciseValue")
                        vo2_value = found[0] if found else None
                    if not vo2_value:
                        found = find_value_recursive(item, "vo2Max")
                        vo2_value = found[0] if found else None
                    if vo2_value:
                        return vo2_value
            except Exception as e:
                logger.warning(f"[WARN] Chyba při stahování VO2 Max pro {date_str}: {str(e)[:100]}")
            return None

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            vo2_value = _fetch_and_parse_vo2(date_str)
            if vo2_value:
                vo2_data.append({"date": date_str, "vo2_max": vo2_value})
                logger.info(f"[INFO] Synchronizuji VO2 MAX pro datum {date_str}: {vo2_value}")
            random_sleep()
            current_date += timedelta(days=1)

        # Fallback: pokud za zadaný rozsah nic není, zkus posledních 30 dní
        if not vo2_data:
            logger.info("[INFO] VO2 Max: v zadaném rozsahu nic, zkouším posledních 30 dní...")
            extended_start = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)
            extended_start_str = extended_start.strftime("%Y-%m-%d")
            scan_date = extended_start
            scan_end = datetime.strptime(start_date, "%Y-%m-%d")  # avoid re-scanning already covered range
            while scan_date < scan_end:
                date_str = scan_date.strftime("%Y-%m-%d")
                vo2_value = _fetch_and_parse_vo2(date_str)
                if vo2_value:
                    vo2_data.append({"date": date_str, "vo2_max": vo2_value})
                    logger.info(f"[INFO] VO2 MAX (fallback 30d) nalezen pro {date_str}: {vo2_value}")
                random_sleep()
                scan_date += timedelta(days=1)

        if vo2_data:
            append_or_update_csv(CSV_FILES["vo2_max"], vo2_data)
            logger.info(f"[INFO] Synchronizovány VO2 Max data: {len(vo2_data)} záznamů")
        else:
            logger.info("[INFO] VO2 Max: bez dostupných dat v API (ani za posledních 30 dní)")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci VO2 Max: {e}")


def sync_sleep(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje spánková data pomocí garmin_obj.get_sleep_data(date_str).

    Mapuje quality_score na sleepScores['overall']['value'].

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji SPÁNEK pro datumový rozsah {start_date} až {end_date}...")
    
    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        sleep_data = []
        
        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            
            try:
                sleep_response = garmin_obj.get_sleep_data(date_str)
                
                if sleep_response and "dailySleepDTO" in sleep_response:
                    daily_sleep = sleep_response["dailySleepDTO"]
                    
                    # Mapuj sleep_score z sleepScores['overall']['value']
                    sleep_scores = daily_sleep.get("sleepScores", {})
                    sleep_score = 0
                    if isinstance(sleep_scores, dict) and "overall" in sleep_scores:
                        sleep_score = sleep_scores["overall"].get("value", 0)
                    
                    # Zkontroluj, zda máme nějaká data
                    sleep_time_seconds = daily_sleep.get("sleepTimeSeconds", 0)
                    duration_minutes = sleep_time_seconds // 60 if sleep_time_seconds else 0
                    
                    if sleep_score > 0 or duration_minutes > 0:
                        sleep_data.append({
                            "date": date_str,
                            "sleep_score": sleep_score,
                            "duration_minutes": duration_minutes,
                            "sleep_start_time": daily_sleep.get("sleepStartTimestampGMT", ""),
                            "sleep_end_time": daily_sleep.get("sleepEndTimestampGMT", ""),
                            "rem_sleep_percentage": sleep_scores.get("remPercentage", {}).get("value", 0),
                            "light_sleep_percentage": sleep_scores.get("lightPercentage", {}).get("value", 0),
                            "deep_sleep_percentage": sleep_scores.get("deepPercentage", {}).get("value", 0),
                            "awake_count": daily_sleep.get("awakeCount", 0),
                            "avg_spo2": daily_sleep.get("averageSpO2Value", 0),
                            "avg_respiration": daily_sleep.get("averageRespirationValue", 0),
                        })
                        logger.info(f"[INFO] Synchronizuji SPÁNEK pro datum {date_str}")
                
                random_sleep()
            except Exception as e:
                logger.warning(f"[WARN] Chyba při stahování spánkových dat pro {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        if sleep_data:
            append_or_update_csv(CSV_FILES["sleep"], sleep_data)
            logger.info(f"[INFO] Synchronizovány SPÁNEK data: {len(sleep_data)} záznamů")
        else:
            logger.info("[INFO] Žádná spánková data k synchronizaci")
    
    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci spánku: {e}")


def sync_daily_health(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje Daily Health data pomocí garmin_obj.get_user_summary(date_str).

    Extrahuje:
    - Body Battery: bodyBatteryHighestValue, bodyBatteryLowestValue
    - Stress: averageStressLevel, maxStressLevel
    - Resting HR: restingHeartRate

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji DAILY HEALTH pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        health_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                health_response = garmin_obj.get_user_summary(date_str)

                if health_response and isinstance(health_response, dict):
                    bb_highest = health_response.get("bodyBatteryHighestValue", 0)
                    bb_lowest = health_response.get("bodyBatteryLowestValue", 0)
                    stress_avg = health_response.get("averageStressLevel", 0)
                    stress_max = health_response.get("maxStressLevel", 0)
                    rhr = health_response.get("restingHeartRate", 0)

                    if bb_highest or bb_lowest or stress_avg or stress_max or rhr:
                        health_data.append({
                            "date": date_str,
                            "body_battery_highest": bb_highest,
                            "body_battery_lowest": bb_lowest,
                            "stress_average": stress_avg,
                            "stress_max": stress_max,
                            "resting_heart_rate": rhr,
                        })
                        logger.info(f"[INFO] Synchronizuji DAILY HEALTH pro datum {date_str}")
                    else:
                        logger.debug(f"[DEBUG] Žádná Daily Health data pro datum {date_str}")
                else:
                    logger.debug(f"[DEBUG] Žádná Daily Health data pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při Daily Health pro {date_str}: {str(e)[:50]}")

            current_date += timedelta(days=1)

        if health_data:
            append_or_update_csv(CSV_FILES["daily_health"], health_data)
            logger.info(f"[INFO] Synchronizovány DAILY HEALTH data: {len(health_data)} záznamů")
        else:
            logger.info("[INFO] Daily Health: bez dostupných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci daily health: {e}")


def sync_training_readiness(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje Training Readiness data pomocí garmin_obj.get_training_readiness(date_str).

    Extrahuje pole: score, recoveryTime, sleepScore, hrvFactorPercent.
    Ukládá raw JSON odpovědi pro debugging.

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji TRAINING READINESS pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        readiness_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                readiness_response = garmin_obj.get_training_readiness(date_str)

                # Fallback: pokud je odpověď prázdná, zkus endpoint s display_name
                if not readiness_response:
                    display_name = getattr(garmin_obj, "display_name", None)
                    if display_name:
                        try:
                            readiness_response = garmin_obj.connectapi(
                                f"/metrics-service/metrics/trainingreadiness/{date_str}",
                                params={"displayName": display_name}
                            )
                        except Exception:
                            pass

                if readiness_response:
                    save_raw_response(f"training_readiness_{date_str}", "get_training_readiness", readiness_response)

                # Odpověď může být seznam nebo slovník
                if isinstance(readiness_response, list) and readiness_response:
                    record = readiness_response[0]
                elif isinstance(readiness_response, dict):
                    record = readiness_response
                else:
                    record = None

                if record and isinstance(record, dict):
                    score = record.get("score", 0)
                    recovery_time = record.get("recoveryTime", 0)
                    sleep_score = record.get("sleepScore", 0)
                    hrv_factor_percent = record.get("hrvFactorPercent", 0)

                    if score or recovery_time or sleep_score or hrv_factor_percent:
                        readiness_data.append({
                            "date": date_str,
                            "score": score,
                            "recovery_time": recovery_time,
                            "sleep_score": sleep_score,
                            "hrv_factor_percent": hrv_factor_percent,
                        })
                        logger.info(f"[INFO] Synchronizuji TRAINING READINESS pro datum {date_str}: score={score}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při stahování Training Readiness pro {date_str}: {str(e)[:100]}")

            current_date += timedelta(days=1)

        if readiness_data:
            append_or_update_csv(CSV_FILES["training_readiness"], readiness_data)
            logger.info(f"[INFO] Synchronizovány TRAINING READINESS data: {len(readiness_data)} záznamů")
        else:
            logger.info("[INFO] Training Readiness: bez dostupných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci training readiness: {e}")


def sync_training_status_history(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje historická Training Status data den po dni pomocí
    garmin_obj.get_training_status(date_str).

    Extrahuje: acuteLoad, trainingStatus, recoveryTime, vo2Max
    z mostRecentTrainingStatus -> latestTrainingStatusData.

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji TRAINING STATUS HISTORII pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        status_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                status_response = garmin_obj.get_training_status(date_str)

                # Fallback: pokud je odpověď prázdná, zkus endpoint s display_name
                if not status_response:
                    display_name = getattr(garmin_obj, "display_name", None)
                    if display_name:
                        try:
                            status_response = garmin_obj.connectapi(
                                f"/trainingstatus-service/trainingstatus/latest/{display_name}"
                            )
                        except Exception:
                            pass

                if status_response and isinstance(status_response, dict):
                    most_recent = status_response.get("mostRecentTrainingStatus", {})
                    latest_data = {}
                    if isinstance(most_recent, dict):
                        latest_data = most_recent.get("latestTrainingStatusData", {})
                        if not isinstance(latest_data, dict):
                            latest_data = {}

                    acute_load = (
                        latest_data.get("acuteLoad")
                        or most_recent.get("acuteLoad")
                        or status_response.get("acuteLoad", 0)
                    )
                    training_status = (
                        latest_data.get("trainingStatus")
                        or most_recent.get("trainingStatus")
                        or status_response.get("trainingStatus", "")
                    )
                    recovery_time = (
                        latest_data.get("recoveryTime")
                        or most_recent.get("recoveryTime")
                        or status_response.get("recoveryTime", 0)
                    )
                    vo2_max = (
                        latest_data.get("vo2Max")
                        or most_recent.get("vo2Max")
                        or status_response.get("vo2Max", 0)
                    )

                    status_data.append({
                        "date": date_str,
                        "acute_load": acute_load or 0,
                        "training_status": training_status or "",
                        "recovery_time": recovery_time or 0,
                        "vo2_max": vo2_max or 0,
                    })

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při Training Status History pro {date_str}: {str(e)[:100]}")

            current_date += timedelta(days=1)

        # Odfiltruj záznamy, kde jsou všechny hodnoty nulové/prázdné
        meaningful = [r for r in status_data if r.get("acute_load") or r.get("training_status") or r.get("recovery_time")]
        if meaningful:
            append_or_update_csv(CSV_FILES["training_status"], meaningful)
            logger.info(f"[INFO] Synchronizovány TRAINING STATUS HISTORY data: {len(meaningful)} záznamů")
        else:
            logger.info("[INFO] Training Status History: bez smysluplných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci training status history: {e}")


def sync_load_focus(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje Load Focus data - složení tréninkového zatížení
    pomocí garminconnect get_training_status() per-day.

    Extrahuje z mostRecentTrainingStatus.latestTrainingStatusData:
    anaerobicLoad, highAerobicLoad, lowAerobicLoad, recoveryTime, vo2Max.

    Args:
        garmin_obj: Garminconnect Garmin objekt
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji LOAD FOCUS pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        load_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                status_response = garmin_obj.get_training_status(date_str)

                if status_response and isinstance(status_response, dict):
                    lts = {}
                    mts = status_response.get("mostRecentTrainingStatus", {})
                    if isinstance(mts, dict):
                        lts = mts.get("latestTrainingStatusData", {}) or {}

                    anaerobic_load = (lts.get("anaerobicLoad")
                                      or status_response.get("anaerobicLoad", 0) or 0)
                    high_aerobic_load = (lts.get("highAerobicLoad")
                                         or status_response.get("highAerobicLoad", 0) or 0)
                    low_aerobic_load = (lts.get("lowAerobicLoad")
                                        or status_response.get("lowAerobicLoad", 0) or 0)
                    recovery_time = (lts.get("recoveryTime")
                                     or mts.get("recoveryTime", 0) if isinstance(mts, dict) else 0
                                     or status_response.get("recoveryTime", 0) or 0)
                    vo2_max = (lts.get("vo2Max")
                               or status_response.get("vo2Max", 0) or 0)

                    if any([anaerobic_load, high_aerobic_load, low_aerobic_load, recovery_time, vo2_max]):
                        load_data.append({
                            "date": date_str,
                            "anaerobic_load": anaerobic_load,
                            "high_aerobic_load": high_aerobic_load,
                            "low_aerobic_load": low_aerobic_load,
                            "recovery_time": recovery_time,
                            "vo2_max": vo2_max,
                        })
                        logger.info(f"[INFO] Synchronizuji LOAD FOCUS pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při stahování load focus pro {date_str}: {str(e)[:100]}")

            current_date += timedelta(days=1)

        if load_data:
            append_or_update_csv(CSV_FILES["load_focus"], load_data)
            logger.info(f"[INFO] Synchronizovány LOAD FOCUS data: {len(load_data)} záznamů")
        else:
            logger.info("[INFO] Load Focus: bez dostupných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci load focus: {e}")


def sync_lactate_threshold(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje Lactate Threshold data
    z /metrics-service/metrics/lactatethreshold/report.
    Pokud data není k dispozici, zkus fallback na get_user_summary().

    Extrahuje: lactateThresholdHeartRate, lactateThresholdSpeed.

    Args:
        garmin_obj: Garminconnect Garmin objekt
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji LACTATE THRESHOLD pro datumový rozsah {start_date} až {end_date}...")

    try:
        lactate_data = []

        # Pokus 1: Hlavní endpoint
        try:
            lactate_response = garmin_obj.connectapi(
                "/metrics-service/metrics/lactatethreshold/report",
                params={}
            )

            if lactate_response:
                logger.debug(f"[DEBUG] Lactate Threshold response preview: {str(lactate_response)[:500]}")

                if isinstance(lactate_response, list):
                    for record in lactate_response:
                        if isinstance(record, dict):
                            record_date = record.get("date") or datetime.now().strftime("%Y-%m-%d")
                            if isinstance(record_date, str) and len(record_date) > 10:
                                record_date = record_date[:10]
                            lt_heart_rate = record.get("lactateThresholdHeartRate", 0) or 0
                            lt_speed = record.get("lactateThresholdSpeed", 0) or 0
                            lactate_data.append({
                                "date": record_date,
                                "lt_heart_rate": lt_heart_rate,
                                "lt_speed": lt_speed,
                            })
                elif isinstance(lactate_response, dict):
                    record_date = lactate_response.get("date") or datetime.now().strftime("%Y-%m-%d")
                    if isinstance(record_date, str) and len(record_date) > 10:
                        record_date = record_date[:10]
                    lt_heart_rate = lactate_response.get("lactateThresholdHeartRate", 0) or 0
                    lt_speed = lactate_response.get("lactateThresholdSpeed", 0) or 0
                    if lt_heart_rate or lt_speed:
                        lactate_data.append({
                            "date": record_date,
                            "lt_heart_rate": lt_heart_rate,
                            "lt_speed": lt_speed,
                        })
        except Exception as e:
            logger.debug(f"[DEBUG] Lactate primary endpoint failed: {str(e)[:100]}")

        # Pokus 2: Fallback na get_user_summary() pokud hlavní vrátí prázdné data
        if not lactate_data:
            logger.debug("[DEBUG] Zkouším fallback na get_user_summary() pro lactate threshold")
            current_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            while current_date <= end_date_obj and len(lactate_data) < 8:
                date_str = current_date.strftime("%Y-%m-%d")
                try:
                    daily_response = garmin_obj.get_user_summary(date_str)
                    if daily_response and isinstance(daily_response, dict):
                        lt_hr = daily_response.get("lactateThresholdHeartRate", 0) or 0
                        lt_sp = daily_response.get("lactateThresholdSpeed", 0) or 0
                        if lt_hr or lt_sp:
                            lactate_data.append({
                                "date": date_str,
                                "lt_heart_rate": lt_hr,
                                "lt_speed": lt_sp,
                            })
                    random_sleep()
                except Exception as e:
                    logger.debug(f"[DEBUG] Chyba při fallback lactate {date_str}: {str(e)[:50]}")

                current_date += timedelta(days=1)

        if lactate_data:
            append_or_update_csv(CSV_FILES["lactate_threshold"], lactate_data)
            logger.info(f"[INFO] Synchronizovány LACTATE THRESHOLD data: {len(lactate_data)} záznamů")
        else:
            logger.info("[INFO] Lactate Threshold: bez dostupných dat v API")

        random_sleep()

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci lactate threshold: {e}")


def sync_training_status(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    [KEY FIX] Synchronizuje Training Status data pomocí garminconnect get_training_status().

    Iteruje každý den v rozsahu, volá get_training_status(date_str) a extrahuje
    z mostRecentTrainingStatus.latestTrainingStatusData:
    trainingStatus, acuteLoad, recoveryTime, loadFocus.

    Args:
        garmin_obj: Garminconnect Garmin objekt
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji TRAINING STATUS pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        status_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                status_response = garmin_obj.get_training_status(date_str)

                # Fallback: pokud je odpověď prázdná, zkus endpoint s display_name
                if not status_response:
                    display_name = getattr(garmin_obj, "display_name", None)
                    if display_name:
                        try:
                            status_response = garmin_obj.connectapi(
                                f"/trainingstatus-service/trainingstatus/latest/{display_name}"
                            )
                        except Exception:
                            pass

                if status_response and isinstance(status_response, dict):
                    mts = status_response.get("mostRecentTrainingStatus", {}) or {}
                    lts = mts.get("latestTrainingStatusData", {}) or {} if isinstance(mts, dict) else {}

                    training_status = (lts.get("trainingStatus")
                                       or mts.get("trainingStatus") if isinstance(mts, dict) else None
                                       or status_response.get("trainingStatus", "") or "")
                    acute_load = (lts.get("acuteLoad")
                                  or mts.get("acuteLoad") if isinstance(mts, dict) else None
                                  or status_response.get("acuteLoad", 0) or 0)
                    recovery_time = (lts.get("recoveryTime")
                                     or mts.get("recoveryTime") if isinstance(mts, dict) else None
                                     or status_response.get("recoveryTime", 0) or 0)
                    load_focus = (lts.get("loadFocus")
                                  or mts.get("loadFocus") if isinstance(mts, dict) else None
                                  or status_response.get("loadFocus", "") or "")

                    if training_status or acute_load or recovery_time:
                        status_data.append({
                            "date": date_str,
                            "training_status": training_status,
                            "acute_load": acute_load,
                            "recovery_time": recovery_time,
                            "load_focus": load_focus,
                        })
                        logger.info(f"[INFO] Synchronizuji TRAINING STATUS pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při stahování training status pro {date_str}: {str(e)[:100]}")

            current_date += timedelta(days=1)

        if status_data:
            append_or_update_csv(CSV_FILES["training_status"], status_data)
            logger.info(f"[INFO] Synchronizovány TRAINING STATUS data: {len(status_data)} záznamů")
        else:
            logger.info("[INFO] Training Status: bez dostupných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci training status: {e}")


def sync_movement(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje data o pohybu - kroky, cíl kroků a vstoupená patra
    pomocí garminconnect get_user_summary().

    Extrahuje:
    - totalSteps (celkový počet kroků)
    - stepsGoal (denní cíl kroků)
    - floorsAscended (vystoupaná patra)

    Args:
        garmin_obj: Garminconnect Garmin objekt
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji POHYB pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        movement_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                movement_response = garmin_obj.get_user_summary(date_str)

                if movement_response and isinstance(movement_response, dict):
                    total_steps = movement_response.get("totalSteps", 0)
                    steps_goal = movement_response.get("stepsGoal", 0)
                    floors_ascended = movement_response.get("floorsAscended", 0)

                    movement_data.append({
                        "date": date_str,
                        "steps": total_steps,
                        "steps_goal": steps_goal,
                        "floors_ascended": floors_ascended,
                    })
                    logger.info(f"[INFO] Synchronizuji POHYB pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při stahování pohybových dat pro {date_str}: {str(e)[:50]}")

            current_date += timedelta(days=1)

        if movement_data:
            append_or_update_csv(CSV_FILES["movement"], movement_data)
            logger.info(f"[INFO] Synchronizovány POHYB data: {len(movement_data)} záznamů")
        else:
            logger.info("[INFO] Pohyb: bez dostupných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci pohybu: {e}")


def sync_intensity(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje data o intenzitě aktivity - mírná a vysoká intenzita
    pomocí garminconnect get_user_summary().

    Extrahuje:
    - moderateIntensityMinutes (mírná intenzita v minutách)
    - vigorousIntensityMinutes (vysoká intenzita v minutách)
    - Vypočítá celkovou intenzitu

    Args:
        garmin_obj: Garminconnect Garmin objekt
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji INTENZITU pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        intensity_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                intensity_response = garmin_obj.get_user_summary(date_str)

                if intensity_response and isinstance(intensity_response, dict):
                    moderate_min = intensity_response.get("moderateIntensityMinutes", 0)
                    vigorous_min = intensity_response.get("vigorousIntensityMinutes", 0)
                    total_intensity_min = moderate_min + vigorous_min

                    intensity_data.append({
                        "date": date_str,
                        "moderate_min": moderate_min,
                        "vigorous_min": vigorous_min,
                        "total_intensity_min": total_intensity_min,
                    })
                    logger.info(f"[INFO] Synchronizuji INTENZITU pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při stahování intenzitních dat pro {date_str}: {str(e)[:50]}")

            current_date += timedelta(days=1)

        if intensity_data:
            append_or_update_csv(CSV_FILES["intensity"], intensity_data)
            logger.info(f"[INFO] Synchronizovány INTENZITA data: {len(intensity_data)} záznamů")
        else:
            logger.info("[INFO] Intenzita: bez dostupných dat v API")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci intenzity: {e}")


def sync_heart_rate(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """
    Synchronizuje detailní data o tepové frekvenci pomocí garminconnect get_heart_rates().

    Extrahuje:
    - Souhrn: maxHeartRate, minHeartRate, restingHeartRate
    - Časová řada: heartRateValues (seznam [timestamp, tep])
    - Výpočet: Průměrný denní tep z časové řady

    Ukládá:
    - heart_rate_summary.csv: date, max_hr, min_hr, resting_hr, avg_hr
    - heart_rate_details.csv (volitelně): timestamp, heart_rate

    Args:
        garmin_obj: Garminconnect Garmin objekt
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
    """
    logger.info(f"[INFO] Synchronizuji TEPOVOU FREKVENCI pro datumový rozsah {start_date} až {end_date}...")

    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        summary_data = []
        details_data = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                hr_response = garmin_obj.get_heart_rates(date_str)

                if hr_response and isinstance(hr_response, dict):
                    max_hr = hr_response.get("maxHeartRate", 0)
                    min_hr = hr_response.get("minHeartRate", 0)
                    resting_hr = hr_response.get("restingHeartRate", 0)

                    heart_rate_values = hr_response.get("heartRateValues", [])
                    avg_hr = 0

                    if isinstance(heart_rate_values, list) and len(heart_rate_values) > 0:
                        valid_values = []
                        for item in heart_rate_values:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                timestamp, hr_value = item[0], item[1]
                                if hr_value and hr_value > 0:
                                    valid_values.append(hr_value)
                                    details_data.append({
                                        "date": date_str,
                                        "timestamp": timestamp,
                                        "heart_rate": hr_value,
                                    })

                        if valid_values:
                            avg_hr = sum(valid_values) / len(valid_values)

                    if max_hr or min_hr or resting_hr or avg_hr:
                        summary_data.append({
                            "date": date_str,
                            "max_hr": max_hr,
                            "min_hr": min_hr,
                            "resting_hr": resting_hr,
                            "avg_hr": round(avg_hr, 1),
                        })
                        logger.info(f"[INFO] Synchronizuji TEPOVOU FREKVENCI pro datum {date_str}")

                random_sleep()
            except Exception as e:
                logger.debug(f"[DEBUG] Chyba při stahování tepové frekvence pro {date_str}: {str(e)[:100]}")

            current_date += timedelta(days=1)

        if summary_data:
            append_or_update_csv(CSV_FILES["heart_rate_summary"], summary_data)
            logger.info(f"[INFO] Synchronizovány TEPOVÁ DATA (souhrn): {len(summary_data)} záznamů")
        else:
            logger.info("[INFO] Tepová data (souhrn): bez dostupných dat v API")

        if details_data:
            append_or_update_csv(CSV_FILES["heart_rate_details"], details_data)
            logger.info(f"[INFO] Synchronizovány TEPOVÁ DATA (detaily): {len(details_data)} meření")

    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci tepové frekvence: {e}")


# ============================================================================
# HLAVNÍ FUNKCE
# ============================================================================

def main():
    """Hlavní funkce pro synchronizaci všech Garmin dat."""

    # 1. Zajisti existenci datových složek (bez mazání existujících dat)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Načti přihlašovací údaje z .env
    email, password, _ = load_credentials()

    # 3. Ověření se do Garmin Connect
    garmin_obj = authenticate(email, password)

    # 4. Vypočítej datumový rozsah pro stahování (Incremental Sync)
    end_date = datetime.now()

    # Detekce posledního existujícího data ze sleep.csv nebo activities.csv
    last_date = None
    for csv_key in ("sleep", "activities"):
        csv_path = CSV_FILES[csv_key]
        if csv_path.exists():
            try:
                df_check = pd.read_csv(csv_path)
                if "date" in df_check.columns and not df_check.empty:
                    latest = pd.to_datetime(df_check["date"]).max()
                    if last_date is None or latest > last_date:
                        last_date = latest
            except Exception as e:
                logger.warning(f"[WARN] Chyba při čtení {csv_path} pro detekci posledního data: {e}")

    if last_date is not None:
        # Pojistka: 3 dny dozadu – Garmin občas zpětně upravuje data
        start_date = last_date - timedelta(days=3)
        logger.info(f"[INFO] Nalezena existující data. Stahuji pouze chybějící dny od {start_date.strftime('%Y-%m-%d')}.")
    else:
        # Nová instalace: stáhni posledních INITIAL_BACKFILL_DAYS dní
        start_date = end_date - timedelta(days=INITIAL_BACKFILL_DAYS)
        logger.info(f"[INFO] Žádná existující data. Zahajuji backfill za posledních {INITIAL_BACKFILL_DAYS} dní.")

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"[INFO] Zahájuji synchronizaci od {start_date_str} do {end_date_str}")

    # 5. Synchronizuj všechny metriky
    sync_activities(garmin_obj, start_date_str, end_date_str)
    sync_hrv(garmin_obj, start_date_str, end_date_str)
    sync_vo2_max(garmin_obj, start_date_str, end_date_str)
    sync_sleep(garmin_obj, start_date_str, end_date_str)
    sync_daily_health(garmin_obj, start_date_str, end_date_str)
    sync_training_readiness(garmin_obj, start_date_str, end_date_str)
    sync_training_status(garmin_obj, start_date_str, end_date_str)
    sync_training_status_history(garmin_obj, start_date_str, end_date_str)
    sync_load_focus(garmin_obj, start_date_str, end_date_str)
    sync_lactate_threshold(garmin_obj, start_date_str, end_date_str)
    sync_heart_rate(garmin_obj, start_date_str, end_date_str)
    sync_movement(garmin_obj, start_date_str, end_date_str)
    sync_intensity(garmin_obj, start_date_str, end_date_str)

    logger.info("[INFO] ✅ Synchronizace dokončena!")


if __name__ == "__main__":
    main()