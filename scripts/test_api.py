#!/usr/bin/env python3
"""
Smoke test – ověření platnosti uložených Garmin Connect tokenů.

Skript NEPROVÁDÍ nové přihlašování přes síť.
Načte existující session z .garminconnect/ a stáhne jeden malý
datový bod (denní souhrn) – postačuje k ověření, že tokeny fungují.
"""

import os
import sys
from datetime import date

from dotenv import load_dotenv
from garminconnect import Garmin, GarminConnectAuthenticationError

TOKEN_STORE = ".garminconnect"

# ── 1) Načtení přihlašovacích údajů ze souboru .env ─────────
load_dotenv()
email    = os.getenv("GARMIN_EMAIL")
password = os.getenv("GARMIN_PASSWORD")

if not email or not password:
    sys.exit("✗  GARMIN_EMAIL nebo GARMIN_PASSWORD nejsou nastaveny v .env")

# ── 2) Inicializace klienta ──────────────────────────────────
api = Garmin(email, password)

try:
    # ── 3) Bezpečný login – POUZE z uložené session ──────────
    #       Pokud tokeny v TOKEN_STORE neexistují nebo jsou prošlé,
    #       knihovna vyhodí výjimku (neprovádí nový síťový login).
    api.login(TOKEN_STORE)
    print("✅ Tokeny úspěšně načteny ze složky", TOKEN_STORE + "/")

    # ── 4) Minimalistický dotaz – dnešní denní souhrn ────────
    today = date.today().isoformat()   # např. "2026-05-14"
    summary = api.get_user_summary(today)

    steps      = summary.get("totalSteps",            "N/A")
    resting_hr = summary.get("restingHeartRate",      "N/A")
    calories   = summary.get("activeKilocalories",    "N/A")
    stress     = summary.get("averageStressLevel",    "N/A")

    print("✅ Připojení k API funguje\n")
    print(f"   📅 Datum:          {today}")
    print(f"   👣 Kroky:          {steps}")
    print(f"   ❤️  Klidový tep:    {resting_hr} bpm")
    print(f"   🔥 Aktivní kcal:   {calories} kcal")
    print(f"   😰 Průměrný stres: {stress}")
    print("\nHotovo – tokeny jsou platné, hlavní skripty lze spustit.")

except GarminConnectAuthenticationError as e:
    print("✗  Autentizace selhala – tokeny jsou pravděpodobně prošlé.")
    print(f"   Detail: {e}")
    print("   → Spusť znovu seed_token.py pro obnovu session.")
    sys.exit(1)

except FileNotFoundError:
    print(f"✗  Složka {TOKEN_STORE}/ neexistuje nebo neobsahuje tokeny.")
    print("   → Nejprve spusť seed_token.py.")
    sys.exit(1)

except Exception as e:
    print(f"✗  Neočekávaná chyba: {e}")
    sys.exit(1)
