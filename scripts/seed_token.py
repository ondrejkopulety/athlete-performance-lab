#!/usr/bin/env python3
"""
Jednorázový skript pro získání a uložení Garmin Connect session tokenů.
Využívá garminconnect 0.3.x s nativním curl_cffi + ua-generator
(TLS fingerprinting řeší knihovna sama – žádný manuální hack hlaviček).
"""

import shutil
from pathlib import Path
from dotenv import load_dotenv
import os

TOKEN_STORE = ".garminconnect"

# ── 1) Smazat starou session ────────────────────────────────
token_dir = Path(TOKEN_STORE)
if token_dir.exists():
    shutil.rmtree(token_dir)
    print(f"✓ Složka {TOKEN_STORE}/ smazána – čistý start.")
else:
    print(f"ℹ Složka {TOKEN_STORE}/ neexistuje, pokračuji.")

# ── 2) Načíst přihlašovací údaje ────────────────────────────
load_dotenv()
email = os.getenv("GARMIN_EMAIL")
password = os.getenv("GARMIN_PASSWORD")

if not email or not password:
    raise SystemExit("✗ GARMIN_EMAIL nebo GARMIN_PASSWORD nejsou nastaveny v .env")

# ── 3) Nativní login přes curl_cffi ─────────────────────────
from garminconnect import Garmin

print(f"⏳ Zahajuji NATIVNÍ login přes curl_cffi (garminconnect 0.3.x)...")
print(f"   účet: {email}")

api = Garmin(email, password)

try:
    api.login(TOKEN_STORE)
    print("✓ Login úspěšný – tokeny uloženy do", TOKEN_STORE + "/")

    # Rychlý smoke-test: stáhni jméno uživatele
    name = api.get_full_name()
    print(f"✓ Ověřeno – přihlášen jako: {name}")

    print("\nHotovo! Nyní můžeš spustit hlavní skripty – použijí uložené tokeny.")
except Exception as e:
    print(f"✗ Login selhal: {e}")
    print("  → Zkontroluj přihlašovací údaje v .env")
    print("  → Případně zkus jinou IP (mobilní hotspot) pro obejití rate-limitu.")
