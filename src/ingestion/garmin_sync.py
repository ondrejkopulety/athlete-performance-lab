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

import io
import json
import logging
import os
import random
import sys
import time
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on sys.path for config import ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import requests.exceptions
from dotenv import load_dotenv
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

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

def random_sleep(min_seconds: int = 3, max_seconds: int = 6) -> None:
    """Náhodná pauza mezi požadavky, aby nás Garmin nezablokoval."""
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)


# Exceptions that indicate Garmin rate limiting / connection issues
_RATE_LIMIT_EXCEPTIONS = (
    ConnectionResetError,
    TimeoutError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    GarminConnectTooManyRequestsError,
)

# HTTP status codes worth retrying (server errors only; 429 is FATAL)
_RETRYABLE_HTTP_CODES = {500, 502, 503, 504}

# Backoff delays in seconds: 1st retry = 30s, 2nd = 120s, 3rd = 300s
_BACKOFF_DELAYS = [30, 120, 300]


class GarminRateLimitError(Exception):
    """Raised when Garmin API rate limit is exhausted after all retries.

    This is a FATAL error – the script must terminate immediately to avoid
    further hammering the Garmin API and escalating the cooldown window.
    """
    pass


class GarminNotFoundError(Exception):
    """Raised when Garmin API returns 404 – data not available for this account."""
    pass


class GarminAuthError(Exception):
    """Raised when stored tokens are missing or invalid.

    Sync se musí přeskočit – opakované neautorizované pokusy jsou
    nejrychlejší cesta k banu.
    """
    pass


def _extract_http_status(exc: Exception) -> int | None:
    """Try to extract an HTTP status code from various exception types."""
    # requests.exceptions.HTTPError → .response.status_code
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None:
            return getattr(resp, "status_code", None)
    # GarminConnectConnectionError – status may be embedded in message
    err_str = str(exc)
    for code in (404, 429, 500, 502, 503, 504):
        if str(code) in err_str:
            return code
    return None


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception warrants a retry with backoff.

    NOTE: HTTP 429 is NOT retryable – it is fatal. See _is_hard_429().
    """
    # Pure network / timeout errors are always retryable
    if isinstance(exc, (ConnectionResetError, TimeoutError,
                        requests.exceptions.ReadTimeout,
                        requests.exceptions.ConnectionError)):
        return True
    # GarminConnectTooManyRequestsError is 429 → NOT retryable (fatal)
    if isinstance(exc, GarminConnectTooManyRequestsError):
        return False
    # HTTP errors: retry only on specific status codes (5xx)
    status = _extract_http_status(exc)
    if status is not None:
        return status in _RETRYABLE_HTTP_CODES
    # Fallback: check message for known timeout patterns (but NOT 429)
    err_str = str(exc).lower()
    return "timeout" in err_str or "connection" in err_str


def _is_not_found(exc: Exception) -> bool:
    """Return True if the exception represents an HTTP 404."""
    return _extract_http_status(exc) == 404


def _is_hard_429(exc: Exception) -> bool:
    """Return True if the exception is a definitive HTTP 429 (not a timeout)."""
    if isinstance(exc, GarminConnectTooManyRequestsError):
        return True
    status = _extract_http_status(exc)
    return status == 429


def polite_api_call(api_fn, *args, metric_name="unknown", **kwargs):
    """
    Wrapper for Garmin API calls with exponential backoff.

    - 404 errors → raises GarminNotFoundError immediately (no retry).
    - Hard 429 (Too Many Requests) → raises GarminRateLimitError immediately.
      The caller / main() must catch this and terminate the script.
    - Other retryable errors (5xx, timeouts) → retry with backoff.
    - Other errors → re-raised immediately.
    """
    last_exception = None

    # First attempt
    try:
        return api_fn(*args, **kwargs)
    except Exception as e:
        if _is_not_found(e):
            raise GarminNotFoundError(
                f"Metrika {metric_name} nenalezena (HTTP 404)"
            ) from e
        if _is_hard_429(e):
            raise GarminRateLimitError(
                f"Rate limit hit (HTTP 429) u metriky {metric_name}. Wait 60 min."
            ) from e
        if _is_retryable(e):
            last_exception = e
        else:
            raise

    # Retry with backoff (only for 5xx / timeouts – NOT 429)
    for attempt, delay in enumerate(_BACKOFF_DELAYS, 1):
        logger.warning(
            f"[RETRY] {metric_name}: pokus {attempt}/{len(_BACKOFF_DELAYS)}, "
            f"chyba: {type(last_exception).__name__}. Čekám {delay}s..."
        )
        time.sleep(delay)
        try:
            return api_fn(*args, **kwargs)
        except Exception as e:
            if _is_not_found(e):
                raise GarminNotFoundError(
                    f"Metrika {metric_name} nenalezena (HTTP 404)"
                ) from e
            if _is_hard_429(e):
                raise GarminRateLimitError(
                    f"Rate limit hit (HTTP 429) u metriky {metric_name}. Wait 60 min."
                ) from e
            if _is_retryable(e):
                last_exception = e
            else:
                raise

    raise GarminRateLimitError(
        f"Garmin API nás odřízlo u metriky {metric_name} po {len(_BACKOFF_DELAYS)} pokusech."
    ) from last_exception


# CSV soubory, kde deduplikace podle 'date' dává smysl (jedna hodnota na den)
_DATE_DEDUP_FILES = {"sleep", "hrv", "daily_health"}


def _choose_dedup_subset(df: pd.DataFrame, csv_path: Path) -> list[str] | None:
    """
    Dynamicky zvolí klíč pro deduplikaci:
      1) activity_id – pro soubory s jednotlivými aktivitami
      2) timestamp   – pro soubory s více záznamy denně (heart_rate_details apod.)
      3) date        – fallback pro agregované denní metriky (sleep, hrv, daily_health)
    """
    if 'activity_id' in df.columns:
        return ['activity_id']
    if 'timestamp' in df.columns:
        return ['timestamp']
    if 'date' in df.columns:
        return ['date']
    return None


def append_or_update_csv(csv_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Přidá nebo aktualizuje CSV soubor.

    Deduplikace je dynamická:
      - activity_id / timestamp → zachová více záznamů na den (dvoufázové tréninky).
      - date → fallback pro denní agregáty (sleep, hrv, daily_health).

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

            df_merged = pd.concat([df_existing, df_new], ignore_index=True)

            dedup_subset = _choose_dedup_subset(df_merged, csv_path)
            if dedup_subset:
                # Normalizuj date sloupec, pokud je součástí klíče
                if 'date' in dedup_subset:
                    df_merged['date'] = pd.to_datetime(df_merged['date'])
                df_merged = df_merged.drop_duplicates(subset=dedup_subset, keep='last')

            # Seřad podle date, pokud existuje
            if 'date' in df_merged.columns:
                df_merged['date'] = pd.to_datetime(df_merged['date'])
                df_merged = df_merged.sort_values('date')
                df_merged['date'] = df_merged['date'].astype(str)

            df_merged.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"[WARN] Chyba při slučování CSV {csv_path}: {e}")
            # Fallback: přepsat soubor
            pd.DataFrame(data).to_csv(csv_path, index=False)
    else:
        # Vytvoř nový soubor
        pd.DataFrame(data).to_csv(csv_path, index=False)


# Placeholder values that signal credentials are not configured
_CREDENTIAL_PLACEHOLDERS = {
    "",
    "your_email@example.com",
    "your_password",
    "your_email",
    "example@example.com",
}


def load_credentials():
    """Načte přihlašovací údaje z .env souboru.

    Returns:
        Tuple (email, password, display_name), or (None, None, None) when
        credentials are absent or contain placeholder values.
    """
    load_dotenv()

    email = os.getenv("GARMIN_EMAIL") or ""
    password = os.getenv("GARMIN_PASSWORD") or ""

    if not email or not password:
        return None, None, None

    if email.lower() in _CREDENTIAL_PLACEHOLDERS or password.lower() in _CREDENTIAL_PLACEHOLDERS:
        return None, None, None

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
    Ověření se Garmin Connect API výhradně pomocí lokálně uložených tokenů.

    Volá POUZE api.login(TOKEN_STORE). Žádné záložní mechanismy
    (garth.refresh_oauth2, čisté api.login()) nejsou povoleny – každý
    neautorizovaný síťový pokus riskuje HTTP 429 a prodloužený ban.

    Pokud tokeny chybí nebo jsou neplatné, vyhodí GarminAuthError. Volající
    musí sync přeskočit – další neautorizované pokusy by riskovaly ban.
    """
    logger.info("[INFO] Ověřuji se do Garmin Connect (pouze lokální tokeny)...")

    api = Garmin(email, password)

    try:
        api.login(TOKEN_STORE)
        logger.info("[INFO] Session úspěšně obnovena z uložených tokenů")
        return api
    except Exception as e:
        # Dřív tu bylo sys.exit(1). To je v CLI v pořádku, ale SystemExit
        # nedědí z Exception, takže by proletěla i přes ošetření v main()
        # a zabila celý uvicorn proces, kdyby sync běžel z API.
        raise GarminAuthError(
            f"Nepodařilo se přihlásit pomocí lokálních tokenů: {e}. "
            f"Tokeny v {TOKEN_STORE}/ jsou neplatné nebo chybí – "
            f"spusť scripts/seed_token.py pro vygenerování nových."
        ) from e


def sync_activities(
    garmin_obj: Garmin,
    start_date: str,
    end_date: str,
    force_redownload_ids: Optional[List] = None,
) -> None:
    """
    Synchronizuje aktivitní data pomocí garmin_obj.get_activities_by_date().

    Args:
        garmin_obj: Přihlášený objekt Garmin
        start_date: Počáteční datum (YYYY-MM-DD)
        end_date: Konečné datum (YYYY-MM-DD)
        force_redownload_ids: Seznam activity_id pro vynucené stažení FIT souborů
    """
    if force_redownload_ids is None:
        force_redownload_ids = []
    logger.info(f"[INFO] Synchronizuji AKTIVITY pro datumový rozsah {start_date} až {end_date}...")

    try:
        activities = polite_api_call(
            garmin_obj.get_activities_by_date, start_date, end_date,
            metric_name="AKTIVITY"
        )

        activities_data = []
        fit_count = 0

        if activities and isinstance(activities, list):
            for activity in activities:
                # Sjednocení na UTC: Garmin API vrací startTimeGMT (UTC)
                # i startTimeLocal. Pro konzistenci s FIT parserem (UTC)
                # odvozujeme date z UTC, aby aktivity kolem půlnoci
                # neměly v různých tabulkách různá data.
                start_time_gmt = activity.get("startTimeGMT", "")
                start_time_local = activity.get("startTimeLocal", "")

                # Primárně date z UTC; fallback na local
                if start_time_gmt:
                    activity_date_str = start_time_gmt[:10]
                elif start_time_local:
                    activity_date_str = start_time_local[:10]
                else:
                    activity_date_str = ""

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
                    "start_time_utc": start_time_gmt,
                    "epoc_load": activity.get("trainingLoad", 0.0),
                    "aerobic_te": activity.get("aerobicTrainingEffect", 0.0),
                    "anaerobic_te": activity.get("anaerobicTrainingEffect", 0.0),
                })

                # Pokus se stáhnout a rozbalit FIT soubor
                fit_path = FIT_DIR / f"activity_{activity_id}.fit"
                _force_this = activity_id in force_redownload_ids
                if fit_path.exists() and not _force_this:
                    logger.debug(f"[DEBUG] FIT soubor už existuje, přeskakuji stahování: {fit_path}")
                else:
                    if _force_this and fit_path.exists():
                        logger.info(f"[REDOWNLOAD] Vynucené opětovné stažení FIT pro {activity_id}")
                    # Retry mechanismus pro Garmin processing lag:
                    # nový FIT soubor nemusí být ihned k dispozici po doběhu.
                    _FIT_RETRY_DELAY = 30   # sekund
                    _FIT_MAX_RETRIES = 3
                    fit_downloaded = False

                    for _fit_attempt in range(1, _FIT_MAX_RETRIES + 1):
                        try:
                            fit_data = polite_api_call(
                                garmin_obj.download_activity,
                                activity_id, dl_fmt=garmin_obj.ActivityDownloadFormat.ORIGINAL,
                                metric_name="FIT DOWNLOAD"
                            )
                            if fit_data and len(fit_data) > 0:
                                if zipfile.is_zipfile(io.BytesIO(fit_data)):
                                    with zipfile.ZipFile(io.BytesIO(fit_data)) as zf:
                                        for name in zf.namelist():
                                            if name.lower().endswith(".fit"):
                                                fit_path.write_bytes(zf.read(name))
                                                fit_count += 1
                                                fit_downloaded = True
                                else:
                                    fit_path.write_bytes(fit_data)
                                    fit_count += 1
                                    fit_downloaded = True
                                break  # úspěch → konec retry smyčky
                            else:
                                # Prázdná data – Garmin ještě nezpracoval FIT
                                if _fit_attempt < _FIT_MAX_RETRIES:
                                    logger.info(
                                        f"[RETRY] FIT pro {activity_id}: prázdná odpověď, "
                                        f"pokus {_fit_attempt}/{_FIT_MAX_RETRIES}. "
                                        f"Čekám {_FIT_RETRY_DELAY}s (Garmin processing lag)..."
                                    )
                                    time.sleep(_FIT_RETRY_DELAY)
                                else:
                                    logger.warning(
                                        f"[WARN] FIT pro {activity_id}: prázdná odpověď "
                                        f"i po {_FIT_MAX_RETRIES} pokusech. Přeskakuji."
                                    )
                        except GarminNotFoundError:
                            # FIT ještě neexistuje na API – retry
                            if _fit_attempt < _FIT_MAX_RETRIES:
                                logger.info(
                                    f"[RETRY] FIT pro {activity_id}: 404, "
                                    f"pokus {_fit_attempt}/{_FIT_MAX_RETRIES}. "
                                    f"Čekám {_FIT_RETRY_DELAY}s..."
                                )
                                time.sleep(_FIT_RETRY_DELAY)
                            else:
                                logger.debug(
                                    f"[DEBUG] FIT pro {activity_id}: 404 i po "
                                    f"{_FIT_MAX_RETRIES} pokusech."
                                )
                            continue
                        except GarminRateLimitError:
                            logger.warning(
                                f"[RATE LIMIT] FIT download pro {activity_id}: "
                                f"rate limit – ukládám dosavadní data a ukončuji."
                            )
                            if activities_data:
                                append_or_update_csv(CSV_FILES["activities"], activities_data)
                                logger.info(f"[PARTIAL] Uloženo {len(activities_data)} aktivit před ukončením.")
                            raise
                        except Exception as e:
                            logger.warning(
                                f"[WARN] FIT pro {activity_id} "
                                f"(pokus {_fit_attempt}): {str(e)[:80]}"
                            )
                            if _fit_attempt < _FIT_MAX_RETRIES:
                                time.sleep(_FIT_RETRY_DELAY)
                            break

                logger.info(f"[INFO] Synchronizuji AKTIVITU pro datum {activity_date_str}")
                random_sleep()

        if activities_data:
            append_or_update_csv(CSV_FILES["activities"], activities_data)
            logger.info(f"[INFO] Synchronizovány AKTIVITY: {len(activities_data)} záznamů, {fit_count} FIT souborů")
        else:
            logger.info("[INFO] Žádné aktivity k synchronizaci")

    except GarminRateLimitError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci aktivit: {e}")


def _daily_range(start_date: str, end_date: str):
    """Vydává 'YYYY-MM-DD' od start do end včetně."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def sync_daily_metric(
    garmin_obj: Garmin,
    start_date: str,
    end_date: str,
    *,
    label: str,
    csv_key: str,
    fetch: Callable[[Garmin, str], Any],
    parse: Callable[[Any, str], Optional[dict]],
    per_day_error_level: int = logging.WARNING,
) -> None:
    """
    Sdílený skelet pro denní Garmin metriky: smyčka přes dny → ``fetch`` →
    ``parse`` → CSV. Odstraňuje ~30 řádků boilerplate na metriku a hlavně
    sjednocuje ošetření rate-limitu (částečný zápis + propagace 429).

    ``fetch(garmin_obj, date_str)`` volá API (smí vyhodit Garmin* výjimky),
    ``parse(response, date_str)`` vrátí řádek do CSV nebo ``None`` (den se
    přeskočí). Metriky s vlastní odchylkou (VO2 s 30denním fallbackem,
    tepová frekvence se dvěma výstupy, aktivity s FIT soubory) tenhle skelet
    nepoužívají.
    """
    logger.info(f"[INFO] Synchronizuji {label} pro datumový rozsah {start_date} až {end_date}...")

    rows: list[dict] = []
    try:
        for date_str in _daily_range(start_date, end_date):
            try:
                response = fetch(garmin_obj, date_str)
                row = parse(response, date_str)
                if row is not None:
                    rows.append(row)
                    logger.info(f"[INFO] Synchronizuji {label} pro datum {date_str}")
                random_sleep()
            except GarminNotFoundError:
                logger.debug(f"[DEBUG] {label} pro {date_str}: 404, přeskakuji den")
            except GarminRateLimitError:
                logger.warning(
                    f"[RATE LIMIT] Garmin API nás odřízlo u metriky {label}. "
                    f"Uloženo {len(rows)} záznamů. Ukončuji skript."
                )
                if rows:
                    append_or_update_csv(CSV_FILES[csv_key], rows)
                raise
            except Exception as e:
                logger.log(per_day_error_level, f"[WARN] Chyba při synchronizaci {label} pro {date_str}: {str(e)[:100]}")

        if rows:
            append_or_update_csv(CSV_FILES[csv_key], rows)
            logger.info(f"[INFO] Synchronizovány {label} data: {len(rows)} záznamů")
        else:
            logger.info(f"[INFO] {label}: bez dostupných dat v API")

    except GarminRateLimitError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci {label}: {e}")


def _parse_hrv(resp: Any, date_str: str) -> Optional[dict]:
    if not (resp and isinstance(resp, dict)):
        return None
    summary = resp.get("hrvSummary", {})
    readings = resp.get("hrvReadings", [])
    if not (readings or (isinstance(summary, dict) and summary.get("lastNightAvg"))):
        return None
    return {
        "date": date_str,
        "weekly_avg": summary.get("weeklyAvg", 0),
        "last_night_avg": summary.get("lastNightAvg", 0),
        "last_night_5min_high": summary.get("lastNight5MinHigh", 0),
        "status": summary.get("status", ""),
        "feedback_text": summary.get("feedbackPhrase", ""),
        "sample_count": len(readings),
    }


def sync_hrv(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """HRV data (garmin_obj.get_hrv_data)."""
    sync_daily_metric(
        garmin_obj, start_date, end_date,
        label="HRV", csv_key="hrv",
        fetch=lambda g, d: polite_api_call(g.get_hrv_data, d, metric_name="HRV"),
        parse=_parse_hrv,
    )


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
                max_metrics = polite_api_call(
                    garmin_obj.get_max_metrics, date_str, metric_name="VO2 MAX"
                )
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
            except GarminNotFoundError:
                pass
            except GarminRateLimitError:
                raise
            except Exception as e:
                logger.warning(f"[WARN] Chyba při stahování VO2 Max pro {date_str}: {str(e)[:100]}")
            return None

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                vo2_value = _fetch_and_parse_vo2(date_str)
            except GarminRateLimitError:
                logger.warning(
                    f"[RATE LIMIT] Garmin API nás odřízlo u metriky VO2 MAX. "
                    f"Uloženo {len(vo2_data)} záznamů. Ukončuji skript."
                )
                if vo2_data:
                    append_or_update_csv(CSV_FILES["vo2_max"], vo2_data)
                raise
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
                try:
                    vo2_value = _fetch_and_parse_vo2(date_str)
                except GarminRateLimitError:
                    logger.warning(
                        f"[RATE LIMIT] Garmin API nás odřízlo u metriky VO2 MAX (fallback). "
                        f"Uloženo {len(vo2_data)} záznamů. Ukončuji skript."
                    )
                    if vo2_data:
                        append_or_update_csv(CSV_FILES["vo2_max"], vo2_data)
                    raise
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

    except GarminRateLimitError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci VO2 Max: {e}")


def _parse_sleep(resp: Any, date_str: str) -> Optional[dict]:
    if not (resp and "dailySleepDTO" in resp):
        return None
    daily_sleep = resp["dailySleepDTO"]
    sleep_scores = daily_sleep.get("sleepScores", {})
    sleep_score = 0
    if isinstance(sleep_scores, dict) and "overall" in sleep_scores:
        sleep_score = sleep_scores["overall"].get("value", 0)
    sleep_time_seconds = daily_sleep.get("sleepTimeSeconds", 0)
    duration_minutes = sleep_time_seconds // 60 if sleep_time_seconds else 0
    if not (sleep_score > 0 or duration_minutes > 0):
        return None
    return {
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
    }


def sync_sleep(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """Spánková data (garmin_obj.get_sleep_data); sleep_score = sleepScores['overall']['value']."""
    sync_daily_metric(
        garmin_obj, start_date, end_date,
        label="SPÁNEK", csv_key="sleep",
        fetch=lambda g, d: polite_api_call(g.get_sleep_data, d, metric_name="SPÁNEK"),
        parse=_parse_sleep,
    )


def _parse_daily_health(resp: Any, date_str: str) -> Optional[dict]:
    if not (resp and isinstance(resp, dict)):
        return None
    bb_highest = resp.get("bodyBatteryHighestValue", 0)
    bb_lowest = resp.get("bodyBatteryLowestValue", 0)
    stress_avg = resp.get("averageStressLevel", 0)
    stress_max = resp.get("maxStressLevel", 0)
    rhr = resp.get("restingHeartRate", 0)
    if not (bb_highest or bb_lowest or stress_avg or stress_max or rhr):
        return None
    return {
        "date": date_str,
        "body_battery_highest": bb_highest,
        "body_battery_lowest": bb_lowest,
        "stress_average": stress_avg,
        "stress_max": stress_max,
        "resting_heart_rate": rhr,
    }


def sync_daily_health(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """Body Battery / stres / klidový tep (garmin_obj.get_user_summary)."""
    sync_daily_metric(
        garmin_obj, start_date, end_date,
        label="DAILY HEALTH", csv_key="daily_health",
        fetch=lambda g, d: polite_api_call(g.get_user_summary, d, metric_name="DAILY HEALTH"),
        parse=_parse_daily_health,
        per_day_error_level=logging.DEBUG,
    )


def _fetch_training_readiness(garmin_obj: Garmin, date_str: str) -> Any:
    resp = polite_api_call(
        garmin_obj.get_training_readiness, date_str, metric_name="TRAINING READINESS"
    )
    # Fallback: prázdná odpověď → zkus endpoint s display_name
    if not resp:
        display_name = getattr(garmin_obj, "display_name", None)
        if display_name:
            try:
                resp = garmin_obj.connectapi(
                    f"/metrics-service/metrics/trainingreadiness/{date_str}",
                    params={"displayName": display_name},
                )
            except Exception:
                pass
    if resp:
        save_raw_response(f"training_readiness_{date_str}", "get_training_readiness", resp)
    return resp


def _parse_training_readiness(resp: Any, date_str: str) -> Optional[dict]:
    if isinstance(resp, list) and resp:
        record = resp[0]
    elif isinstance(resp, dict):
        record = resp
    else:
        return None
    if not isinstance(record, dict):
        return None
    score = record.get("score", 0)
    recovery_time = record.get("recoveryTime", 0)
    sleep_score = record.get("sleepScore", 0)
    hrv_factor_percent = record.get("hrvFactorPercent", 0)
    if not (score or recovery_time or sleep_score or hrv_factor_percent):
        return None
    return {
        "date": date_str,
        "score": score,
        "recovery_time": recovery_time,
        "sleep_score": sleep_score,
        "hrv_factor_percent": hrv_factor_percent,
    }


def sync_training_readiness(garmin_obj: Garmin, start_date: str, end_date: str) -> None:
    """Training Readiness (score/recoveryTime/sleepScore/hrvFactorPercent)."""
    sync_daily_metric(
        garmin_obj, start_date, end_date,
        label="TRAINING READINESS", csv_key="training_readiness",
        fetch=_fetch_training_readiness,
        parse=_parse_training_readiness,
        per_day_error_level=logging.DEBUG,
    )


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
    sync_daily_metric(
        garmin_obj, start_date, end_date,
        label="POHYB", csv_key="movement",
        fetch=lambda g, d: polite_api_call(g.get_user_summary, d, metric_name="POHYB"),
        parse=_parse_movement,
        per_day_error_level=logging.DEBUG,
    )


def _parse_movement(resp: Any, date_str: str) -> Optional[dict]:
    if not (resp and isinstance(resp, dict)):
        return None
    return {
        "date": date_str,
        "steps": resp.get("totalSteps", 0),
        "steps_goal": resp.get("stepsGoal", 0),
        "floors_ascended": resp.get("floorsAscended", 0),
    }


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
    sync_daily_metric(
        garmin_obj, start_date, end_date,
        label="INTENZITA", csv_key="intensity",
        fetch=lambda g, d: polite_api_call(g.get_user_summary, d, metric_name="INTENZITA"),
        parse=_parse_intensity,
        per_day_error_level=logging.DEBUG,
    )


def _parse_intensity(resp: Any, date_str: str) -> Optional[dict]:
    if not (resp and isinstance(resp, dict)):
        return None
    moderate_min = resp.get("moderateIntensityMinutes", 0)
    vigorous_min = resp.get("vigorousIntensityMinutes", 0)
    return {
        "date": date_str,
        "moderate_min": moderate_min,
        "vigorous_min": vigorous_min,
        "total_intensity_min": moderate_min + vigorous_min,
    }


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
                hr_response = polite_api_call(
                    garmin_obj.get_heart_rates, date_str, metric_name="TEPOVÁ FREKVENCE"
                )

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
            except GarminNotFoundError:
                logger.debug(f"[DEBUG] Tepová frekvence pro {date_str}: 404, přeskakuji den")
            except GarminRateLimitError:
                logger.warning(
                    f"[RATE LIMIT] Garmin API nás odřízlo u metriky TEPOVÁ FREKVENCE. "
                    f"Uloženo {len(summary_data)} záznamů. Ukončuji skript."
                )
                if summary_data:
                    append_or_update_csv(CSV_FILES["heart_rate_summary"], summary_data)
                if details_data:
                    append_or_update_csv(CSV_FILES["heart_rate_details"], details_data)
                raise
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

    except GarminRateLimitError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Chyba při synchronizaci tepové frekvence: {e}")


# ============================================================================
# HLAVNÍ FUNKCE
# ============================================================================

def main() -> set:
    """Hlavní funkce pro synchronizaci všech Garmin dat.

    Returns:
        Set of activity_id values that were synced (new or updated).
        Empty set if sync was skipped or no activities were synced.
    """
    dirty_ids: set = set()

    # 1. Zajisti existenci datových složek (bez mazání existujících dat)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Načti přihlašovací údaje z .env
    email, password, _ = load_credentials()

    # Přímá kontrola: pokud jsou credentials prázdné, přeskoč sync a pokračuj v pipeline
    if not email or not password:
        logger.info("ℹ️  Garmin credentials missing. Skipping cloud sync.")
        return dirty_ids

    # 3. Ověření se do Garmin Connect – pokud selže, pokračuj s lokálními soubory
    try:
        garmin_obj = authenticate(email, password)
    except Exception as auth_err:
        logger.warning(f"[WARN] Garmin login failed: {auth_err}. Skipping cloud sync and continuing pipeline.")
        return dirty_ids

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
        # ── Smart Incremental Sync: FIT existence check (last 3 days) ────
        # Pro posledních 3 dní zkontroluj fyzickou existenci FIT souborů
        # na disku. Pokud soubor chybí, naplánuj download.
        backfill_days = 2  # pokryjeme včerejšek i dnešek
        force_redownload_ids: list = []
        activities_csv = CSV_FILES["activities"]
        if activities_csv.exists():
            try:
                df_act = pd.read_csv(activities_csv)
                if "activity_id" in df_act.columns and "date" in df_act.columns and not df_act.empty:
                    cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    recent = df_act[df_act["date"].astype(str) >= cutoff_date]
                    missing_fit = 0
                    for _, row in recent.iterrows():
                        aid = row.get("activity_id")
                        if aid is None or pd.isna(aid):
                            continue
                        aid = int(aid) if isinstance(aid, float) else aid
                        fit_path = FIT_DIR / f"activity_{aid}.fit"
                        if not fit_path.exists():
                            missing_fit += 1
                            force_redownload_ids.append(aid)
                            logger.info(
                                f"[FIT CHECK] Chybí FIT soubor pro aktivitu {aid} "
                                f"(datum: {row.get('date', '?')})"
                            )
                    if missing_fit > 0:
                        logger.info(
                            f"[FIT CHECK] {missing_fit} aktivit z posledních 7 dní "
                            f"nemá FIT soubor na disku – vynucuji stažení."
                        )
                    else:
                        logger.info("[FIT CHECK] Všechny FIT soubory za posledních 7 dní existují.")
            except Exception as e:
                logger.warning(f"[WARN] FIT existence check selhal: {e}")

        start_date = last_date - timedelta(days=backfill_days)
        logger.info(
            f"[INFO] Nalezena existující data. Sync okno: "
            f"{start_date.strftime('%Y-%m-%d')} → dnes (backfill={backfill_days}d)."
        )
    else:
        # Nová instalace: stáhni posledních INITIAL_BACKFILL_DAYS dní
        start_date = end_date - timedelta(days=INITIAL_BACKFILL_DAYS)
        force_redownload_ids = []
        logger.info(f"[INFO] Žádná existující data. Zahajuji backfill za posledních {INITIAL_BACKFILL_DAYS} dní.")

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"[INFO] Zahájuji synchronizaci od {start_date_str} do {end_date_str}")

    # 5. Synchronizuj všechny metriky.
    #    GarminRateLimitError (429) je FATÁLNÍ – okamžité ukončení.
    #    Ostatní chyby → přeskoč metriku a pokračuj.
    try:
        sync_activities(garmin_obj, start_date_str, end_date_str,
                        force_redownload_ids=force_redownload_ids)
    except GarminNotFoundError:
        logger.info(
            "[INFO] Metrika AKTIVITY nenalezena (404). Přeskakuji."
        )
    except GarminRateLimitError as e:
        # Dřív se tu volalo sys.exit(1). To je v pořádku pro CLI, ale
        # zabilo by celý uvicorn proces, kdyby sync běžel z API. Výjimka
        # nechá rozhodnutí na volajícím.
        logger.error(f"[FATAL] {e}")
        raise
    except Exception as e:
        logger.warning(f"[WARN] Chyba při synchronizaci AKTIVITY: {e}. Přeskakuji.")

    _sync_steps: List[tuple] = [
        ("HRV", sync_hrv),
        ("VO2 MAX", sync_vo2_max),
        ("SPÁNEK", sync_sleep),
        ("DAILY HEALTH", sync_daily_health),
        ("TRAINING READINESS", sync_training_readiness),
        # Training Status / Status History / Load Focus / Lactate Threshold:
        # Garmin na těchto endpointech pro tenhle účet vrací 404 / prázdno,
        # takže se nesynchronizují (funkce byly odstraněny 8/2026).
        ("TEPOVÁ FREKVENCE", sync_heart_rate),
        ("POHYB", sync_movement),
        ("INTENZITA", sync_intensity),
    ]

    for metric_name, sync_fn in _sync_steps:
        try:
            sync_fn(garmin_obj, start_date_str, end_date_str)
            random_sleep(4, 8)
        except GarminNotFoundError:
            logger.info(
                f"[INFO] Metrika {metric_name} nenalezena (404). Přeskakuji."
            )
        except GarminRateLimitError as e:
            logger.error(
                f"[FATAL] Rate limit hit u metriky {metric_name}. "
                f"Wait 60 min. Detail: {e}"
            )
            raise
        except Exception as e:
            logger.warning(
                f"[WARN] Chyba při synchronizaci {metric_name}: {e}. Přeskakuji."
            )

    logger.info("[INFO] ✅ Synchronizace dokončena!")

    # Collect dirty activity IDs from the activities CSV (all IDs in the sync window)
    try:
        activities_csv = CSV_FILES["activities"]
        if activities_csv.exists():
            df_act = pd.read_csv(activities_csv)
            if "activity_id" in df_act.columns and "date" in df_act.columns:
                mask = df_act["date"].astype(str) >= start_date_str
                synced = df_act.loc[mask, "activity_id"].dropna()
                dirty_ids = set(int(x) if not isinstance(x, int) else x for x in synced)
    except Exception as e:
        logger.warning(f"[WARN] Nepodařilo se načíst dirty IDs: {e}")

    # Persist dirty IDs to a marker file for downstream pipeline stages
    dirty_ids_file = SUMMARIES_DIR / ".dirty_activity_ids.json"
    try:
        dirty_ids_file.write_text(json.dumps(list(dirty_ids), default=str))
        logger.info(f"[INFO] Dirty IDs ({len(dirty_ids)}) uloženy do {dirty_ids_file}")
    except Exception as e:
        logger.warning(f"[WARN] Nepodařilo se uložit dirty IDs: {e}")

    return dirty_ids


if __name__ == "__main__":
    main()
