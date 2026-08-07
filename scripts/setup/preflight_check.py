"""
Preflight diagnostics – local-only environment check before Garmin API login.
No Garmin endpoints are contacted.
"""

import sys
import os

def check_python_version():
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 14):
        print(f"❌ Python verze: {version_str}  (doporučeno < 3.14, ideálně 3.12)")
    else:
        print(f"✅ Python verze: {version_str}")


def check_virtualenv():
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print(f"✅ Virtuální prostředí aktivní: {sys.prefix}")
    else:
        print("❌ Virtuální prostředí NENÍ aktivní (sys.prefix == sys.base_prefix)")


def check_env_variables():
    from dotenv import load_dotenv

    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.isfile(dotenv_path):
        print("❌ Soubor .env nenalezen")
        return

    load_dotenv(dotenv_path)

    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    if email:
        local, _, domain = email.partition("@")
        masked = local[:3] + "***@" + domain if len(local) > 3 else local[0] + "***@" + domain
        print(f"✅ GARMIN_EMAIL: {masked}")
    else:
        print("❌ GARMIN_EMAIL není nastavena")

    if password:
        print(f"✅ GARMIN_PASSWORD: nastavena ({len(password)} znaků)")
    else:
        print("❌ GARMIN_PASSWORD není nastavena")


def check_public_ip():
    import requests

    try:
        resp = requests.get("https://api.ipify.org", timeout=5)
        resp.raise_for_status()
        ip = resp.text.strip()
        print(f"✅ Veřejná IP adresa: {ip}")
        print("   ⚠️  Ověř, že tato IP odpovídá tvému mobilnímu hotspotu, NE domácí Wi-Fi!")
    except Exception as e:
        print(f"❌ Nelze zjistit veřejnou IP: {e}")


def main():
    print("=" * 55)
    print("  🔍  PREFLIGHT CHECK – lokální diagnostika prostředí")
    print("=" * 55)
    print()

    check_python_version()
    check_virtualenv()
    print()
    check_env_variables()
    print()
    check_public_ip()

    print()
    print("=" * 55)
    print("  Hotovo. Pokud je vše ✅, můžeš spustit přihlášení.")
    print("=" * 55)


if __name__ == "__main__":
    main()
