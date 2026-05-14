"""
Strava Original File Exporter
=============================
Hromadný export originálních souborů (.fit/.gpx/.tcx/.zip) ze Stravy.
Používá stravalib pro metadata a requests se session cookie pro RAW stažení.

Funkce:
  - Dynamická detekce přípony z Content-Disposition hlavičky
  - Kompletní pagination (všechny aktivity, nejen posledních 30)
  - Robustní error handling (expirovaná cookie, síťové chyby)
  - Kontrola integrity (přeskočení existujících + nulových souborů)
  - Inteligentní rate limiting (100 req / 15 min)
  - Logging do souboru i konzole
"""

import os
import re
import sys
import time
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv
from stravalib.client import Client

# Načte proměnné z .env souboru (přepíše případné env proměnné shellu)
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# KONFIGURACE
# ---------------------------------------------------------------------------
def _require_env_int(key: str) -> int:
    """Načte env proměnnou a převede ji na int; selže srozumitelnou chybou."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        raise EnvironmentError(
            f"Chybí povinná proměnná prostředí '{key}'. Zkontroluj .env soubor."
        )
    try:
        return int(raw)
    except ValueError:
        raise EnvironmentError(
            f"Proměnná '{key}' musí být celé číslo, ale obsahuje: {raw!r}"
        )


def _require_env_str(key: str) -> str:
    """Načte env proměnnou jako string; selže srozumitelnou chybou pokud chybí."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        raise EnvironmentError(
            f"Chybí povinná proměnná prostředí '{key}'. Zkontroluj .env soubor."
        )
    return raw


CLIENT_ID: int = _require_env_int("STRAVA_CLIENT_ID")
CLIENT_SECRET: str = _require_env_str("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN: str = _require_env_str("STRAVA_REFRESH_TOKEN")
SESSION_COOKIE: str = _require_env_str("STRAVA_SESSION_COOKIE")
DOWNLOAD_DIR = "data/fit/strava_originals"
LOG_FILE = "logs/export_debug.log"

# Rate-limit: Strava povoluje ~100 požadavků / 15 minut
RATE_LIMIT_MAX = 95  # necháme si rezervu 5 req
RATE_LIMIT_WINDOW = 15 * 60  # 900 sekund
SLEEP_BETWEEN_REQUESTS = 2  # sekundy mezi požadavky
SLEEP_RATE_LIMIT_PAUSE = 120  # sekundy čekání po dosažení limitu

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("strava_export")
logger.setLevel(logging.DEBUG)

# Formát logů
_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Handler: soubor (DEBUG+)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# Handler: konzole (INFO+)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# RATE LIMITER
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sleduje počet požadavků a automaticky pozastaví export při blížícím se limitu."""

    def __init__(self, max_requests: int, window_seconds: int, pause_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.pause_seconds = pause_seconds
        self.timestamps: list[float] = []

    def _prune(self):
        """Odstraní timestampy starší než okno."""
        cutoff = time.time() - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    def wait_if_needed(self):
        """Pokud se blížíme limitu, počká potřebnou dobu."""
        self._prune()
        if len(self.timestamps) >= self.max_requests:
            oldest = min(self.timestamps)
            wait_until = oldest + self.window_seconds
            wait_secs = max(0, wait_until - time.time()) + 5  # +5s rezerva
            logger.warning(
                "Rate limit dosažen (%d/%d za posledních %d s). "
                "Čekám %.0f s...",
                len(self.timestamps),
                self.max_requests,
                self.window_seconds,
                wait_secs,
            )
            time.sleep(wait_secs)
            self._prune()

    def record(self):
        """Zaznamená provedený požadavek."""
        self.timestamps.append(time.time())

    @property
    def remaining(self) -> int:
        self._prune()
        return max(0, self.max_requests - len(self.timestamps))


rate_limiter = RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW, SLEEP_RATE_LIMIT_PAUSE)

# ---------------------------------------------------------------------------
# INICIALIZACE KLIENTA
# ---------------------------------------------------------------------------
client = Client()


def refresh_access_token() -> str:
    """Obnoví přístupový token přes OAuth2 refresh flow."""
    # --- Sanity Check ---
    secret_preview = CLIENT_SECRET[:4] + "***" if len(CLIENT_SECRET) > 4 else "***"
    token_preview = REFRESH_TOKEN[:4] + "***" if len(REFRESH_TOKEN) > 4 else "***"
    logger.info(
        "Ověřuji konfiguraci: ID=%d, Secret=%s, Token=%s",
        CLIENT_ID,
        secret_preview,
        token_preview,
    )

    logger.info("Obnovuji přístupový token...")
    try:
        response = client.refresh_access_token(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            refresh_token=REFRESH_TOKEN,
        )
        logger.info("Token úspěšně obnoven.")
        return response["access_token"]
    except Exception as e:
        logger.error("Nepodařilo se obnovit token: %s", e)
        raise


# ---------------------------------------------------------------------------
# DETEKCE PŘÍPONY
# ---------------------------------------------------------------------------

def _detect_extension(response: requests.Response) -> str:
    """
    Zjistí příponu souboru z hlavičky Content-Disposition.
    Fallback: Content-Type → .fit jako poslední možnost.
    """
    # 1) Content-Disposition: attachment; filename="xyz.fit.gz"
    cd = response.headers.get("Content-Disposition", "")
    if cd:
        match = re.search(r'filename[*]?=["\']?([^"\';]+)', cd, re.IGNORECASE)
        if match:
            fname = match.group(1).strip()
            # Může být vícenásobná přípona (.fit.gz)
            suffixes = Path(fname).suffixes
            if suffixes:
                ext = "".join(suffixes)  # ".fit.gz" nebo ".gpx"
                logger.debug("Přípona z Content-Disposition: %s (filename=%s)", ext, fname)
                return ext

    # 2) Fallback na Content-Type
    ct = response.headers.get("Content-Type", "").lower()
    ct_map = {
        "application/zip": ".zip",
        "application/x-gzip": ".fit.gz",
        "application/gzip": ".fit.gz",
        "application/octet-stream": ".fit",
        "application/gpx+xml": ".gpx",
        "application/xml": ".gpx",
        "text/xml": ".tcx",
        "application/vnd.garmin.tcx+xml": ".tcx",
    }
    for key, ext in ct_map.items():
        if key in ct:
            logger.debug("Přípona z Content-Type: %s (Content-Type=%s)", ext, ct)
            return ext

    logger.warning("Nelze zjistit příponu – použiji .fit jako fallback.")
    return ".fit"


# ---------------------------------------------------------------------------
# DETEKCE EXPIROVANÉ COOKIE
# ---------------------------------------------------------------------------

def _is_login_redirect(response: requests.Response) -> bool:
    """Detekuje, zda odpověď je přesměrování na login stránku (expirovaná cookie)."""
    # Strava vrátí 302 → login, nebo 200 s HTML login stránkou
    if response.status_code in (301, 302, 303, 307, 308):
        location = response.headers.get("Location", "")
        if "login" in location.lower() or "oauth" in location.lower():
            return True

    ct = response.headers.get("Content-Type", "").lower()
    if "text/html" in ct:
        # Čteme jen prvních 2 KB pro kontrolu
        peek = response.content[:2048].decode("utf-8", errors="ignore").lower()
        if "login" in peek or "sign in" in peek or "<title>strava</title>" in peek:
            return True

    return False


# ---------------------------------------------------------------------------
# STAŽENÍ SOUBORU
# ---------------------------------------------------------------------------

def download_original(
    activity_id: int, session_cookie: str, base_path_no_ext: str
) -> tuple[bool, str | None]:
    """
    Stáhne originální soubor ze Stravy.

    Args:
        activity_id: ID aktivity na Stravě.
        session_cookie: Hodnota _strava4_session cookie.
        base_path_no_ext: Cesta k souboru BEZ přípony (ta se doplní dynamicky).

    Returns:
        (success, final_filepath) – True a cesta, pokud se povedlo.
    """
    url = f"https://www.strava.com/activities/{activity_id}/export_original"
    cookies = {"_strava4_session": session_cookie}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }

    rate_limiter.wait_if_needed()

    try:
        # allow_redirects=False – chceme chytit 302 → login
        response = requests.get(
            url,
            cookies=cookies,
            headers=headers,
            stream=True,
            allow_redirects=False,
            timeout=60,
        )
        rate_limiter.record()

        logger.debug(
            "GET %s → %d | Content-Type: %s | Content-Disposition: %s",
            url,
            response.status_code,
            response.headers.get("Content-Type", "?"),
            response.headers.get("Content-Disposition", "?"),
        )

        # --- Kontrola expirované cookie ---
        if _is_login_redirect(response):
            logger.error(
                "Session cookie expirovala! Odpověď je login redirect pro ID %s. "
                "Aktualizuj SESSION_COOKIE a spusť znovu.",
                activity_id,
            )
            return False, None

        if response.status_code == 404:
            logger.warning("Aktivita %s nemá originální soubor (404).", activity_id)
            return False, None

        if response.status_code == 429:
            logger.warning(
                "HTTP 429 Too Many Requests pro ID %s. Čekám %d s...",
                activity_id,
                SLEEP_RATE_LIMIT_PAUSE,
            )
            time.sleep(SLEEP_RATE_LIMIT_PAUSE)
            return False, None

        if response.status_code != 200:
            logger.error(
                "Neočekávaný status %d pro ID %s.", response.status_code, activity_id
            )
            return False, None

        # --- Dynamická přípona ---
        ext = _detect_extension(response)
        final_path = base_path_no_ext + ext

        # --- Stažení do souboru ---
        with open(final_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # --- Kontrola nulové velikosti ---
        file_size = os.path.getsize(final_path)
        if file_size == 0:
            logger.warning(
                "Stažený soubor %s má nulovou velikost – mažu.", final_path
            )
            os.remove(final_path)
            return False, None

        return True, final_path

    except requests.exceptions.Timeout:
        logger.error("Timeout při stahování ID %s.", activity_id)
        return False, None
    except requests.exceptions.ConnectionError as e:
        logger.error("Síťová chyba při stahování ID %s: %s", activity_id, e)
        return False, None
    except requests.exceptions.RequestException as e:
        logger.error("HTTP chyba při stahování ID %s: %s", activity_id, e)
        return False, None
    except OSError as e:
        logger.error("Chyba I/O při zápisu ID %s: %s", activity_id, e)
        return False, None


# ---------------------------------------------------------------------------
# KONTROLA EXISTUJÍCÍCH SOUBORŮ
# ---------------------------------------------------------------------------

def _find_existing_file(base_path_no_ext: str) -> str | None:
    """
    Hledá existující soubor s libovolnou příponou odpovídající base_path_no_ext.
    Vrací cestu k nalezenému souboru, nebo None.
    """
    parent = os.path.dirname(base_path_no_ext)
    prefix = os.path.basename(base_path_no_ext)
    if not os.path.isdir(parent):
        return None
    for fname in os.listdir(parent):
        if fname.startswith(prefix) and fname != prefix:
            full = os.path.join(parent, fname)
            size = os.path.getsize(full)
            if size > 0:
                return full
            else:
                # Nulový soubor – smažeme a stáhneme znovu
                logger.warning("Existující soubor %s má nulovou velikost – mažu.", full)
                os.remove(full)
    return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Strava Original Export – START")
    logger.info("Výstupní adresář: %s", DOWNLOAD_DIR)
    logger.info("=" * 60)

    # 1. Získání platného tokenu
    access_token = refresh_access_token()
    client.access_token = access_token

    # 2. Načtení VŠECH aktivit (limit=None → kompletní historie)
    logger.info("Načítám kompletní seznam aktivit ze Stravy...")
    try:
        activities = list(client.get_activities(limit=None))
    except Exception as e:
        logger.error("Nepodařilo se načíst aktivity: %s", e)
        sys.exit(1)

    total = len(activities)
    logger.info("Nalezeno %d aktivit.", total)

    downloaded = 0
    skipped = 0
    failed = 0
    failed_ids: list[int] = []

    for i, activity in enumerate(activities, start=1):
        activity_id = activity.id
        date_str = activity.start_date.strftime("%Y-%m-%d")
        safe_name = "".join(
            c for c in str(activity.name) if c.isalnum() or c in (" ", "_")
        ).strip()
        base_filename = f"{date_str}_{safe_name}_{activity_id}"
        base_path = os.path.join(DOWNLOAD_DIR, base_filename)

        # --- Kontrola: soubor už existuje a má nenulovou velikost? ---
        existing = _find_existing_file(base_path)
        if existing:
            logger.debug(
                "[%d/%d] Přeskakuji %s – existuje (%d KB).",
                i,
                total,
                os.path.basename(existing),
                os.path.getsize(existing) // 1024,
            )
            skipped += 1
            continue

        logger.info(
            "[%d/%d] Stahuji: %s (ID: %s) | Zbývá rate-limit: %d",
            i,
            total,
            base_filename,
            activity_id,
            rate_limiter.remaining,
        )

        success, final_path = download_original(activity_id, SESSION_COOKIE, base_path)

        if success and final_path:
            size_kb = os.path.getsize(final_path) // 1024
            logger.info(
                "  -> OK: %s (%d KB)", os.path.basename(final_path), size_kb
            )
            downloaded += 1
        else:
            logger.warning("  -> CHYBA u ID %s.", activity_id)
            failed += 1
            failed_ids.append(activity_id)

        # Pauza mezi požadavky
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # --- Souhrn ---
    logger.info("=" * 60)
    logger.info("HOTOVO")
    logger.info("  Celkem aktivit:  %d", total)
    logger.info("  Staženo:         %d", downloaded)
    logger.info("  Přeskočeno:      %d", skipped)
    logger.info("  Chyby:           %d", failed)
    if failed_ids:
        logger.info("  Neúspěšná ID:    %s", ", ".join(str(x) for x in failed_ids))
    logger.info("=" * 60)


def _connection_test():
    """
    Rychlý test připojení: obnoví token a vypíše prvních 5 aktivit.
    Slouží k ověření, že .env je správně nastaven a API komunikuje.
    """
    print("\n" + "=" * 60)
    print("  STRAVA CONNECTION TEST")
    print("=" * 60)

    try:
        access_token = refresh_access_token()
    except EnvironmentError as e:
        print(f"\n[CHYBA KONFIGURACE] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CHYBA TOKENU] {e}")
        sys.exit(1)

    client.access_token = access_token
    print(f"\nToken obnoven. Načítám prvních 5 aktivit...\n")

    try:
        activities = list(client.get_activities(limit=5))
    except Exception as e:
        print(f"[CHYBA API] Nepodařilo se načíst aktivity: {e}")
        sys.exit(1)

    if not activities:
        print("[VAROVÁNÍ] Žádné aktivity nebyly nalezeny – účet je prázdný?")
        return

    print(f"  {'#':<4} {'Datum':<12} {'Typ':<20} {'Název':<40} {'ID'}")
    print("  " + "-" * 95)
    for idx, act in enumerate(activities, start=1):
        date_str = act.start_date.strftime("%Y-%m-%d") if act.start_date else "?"
        sport = str(act.sport_type or act.type or "?")[:20]
        name = str(act.name or "")[:40]
        print(f"  {idx:<4} {date_str:<12} {sport:<20} {name:<40} {act.id}")

    print("\n" + "=" * 60)
    print("  Spojení funguje. Spusť main() pro hromadný export.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--test":
        _connection_test()
    else:
        main()