"""
strava_auth.py  –  OAuth2 refresh flow pro Stravu (jedno místo)
==============================================================

Dřív žila obnova access tokenu ve dvou kopiích: ``strava_map.py`` (pipeline,
párování odkazů) a ``strava_client.py`` (samostatný stahovač originálních
FIT souborů). Rozcházely se v tom, jestli po sobě přepisují ``.env`` a jak
řeší chyby. Teď je to jedna funkce.

Strava při obnově občas vrátí **nový** refresh token a starý zneplatní –
musí se uložit do ``.env``, jinak příští běh selže na autentizaci. Zápis je
atomický (``.env.tmp`` + ``os.replace``), aby souběh (noční cron + ruční
spuštění) ``.env`` nepoškodil. ``persist_rotation=False`` ten zápis vypne –
používá ho ``--dry-run``, který nemá mít žádný vedlejší efekt.
"""

from __future__ import annotations

import logging
import os

import requests

from config.settings import PROJECT_ROOT, STRAVA_TOKEN_URL

log = logging.getLogger("strava_auth")

_ENV_PATH = PROJECT_ROOT / ".env"
_HTTP_TIMEOUT = 30


class StravaAuthError(RuntimeError):
    """Chybí nebo nefunguje OAuth konfigurace – krok se má přeskočit, ne spadnout."""


def _env(key: str) -> str:
    val = (os.environ.get(key) or "").strip()
    if not val:
        raise StravaAuthError(f"Chybí {key} v .env – krok Strava přeskočen.")
    return val


def _persist_refresh_token(new_token: str) -> None:
    """
    Přepíše ``STRAVA_REFRESH_TOKEN`` v ``.env``. Atomicky: nový obsah se
    zapíše do ``.env.tmp`` a teprve ``os.replace`` ho přesune na místo, takže
    souběžný čtenář vidí buď starý, nebo nový soubor, nikdy půlku.
    Best-effort – když ``.env`` nejde zapsat, jen varujeme.
    """
    try:
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning(
            "Nový refresh token nelze uložit (%s) – ulož STRAVA_REFRESH_TOKEN ručně.", exc
        )
        return

    out, replaced = [], False
    for line in lines:
        if line.startswith("STRAVA_REFRESH_TOKEN="):
            out.append(f"STRAVA_REFRESH_TOKEN={new_token}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"STRAVA_REFRESH_TOKEN={new_token}")

    tmp = _ENV_PATH.with_suffix(_ENV_PATH.suffix + ".tmp")
    try:
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        os.replace(tmp, _ENV_PATH)
        os.environ["STRAVA_REFRESH_TOKEN"] = new_token
        log.info("Strava vrátila nový refresh token – uložen do .env.")
    except OSError as exc:
        log.warning(
            "Nový refresh token nelze uložit (%s) – ulož STRAVA_REFRESH_TOKEN ručně.", exc
        )
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def refresh_access_token(*, persist_rotation: bool = True) -> str:
    """
    OAuth2 refresh flow – vrátí platný access token.

    ``persist_rotation=False``: i když Strava vrátí nový refresh token,
    neuloží se do ``.env`` (jen se zaloguje). Pro ``--dry-run``, který nemá
    nic měnit.
    """
    client_id = _env("STRAVA_CLIENT_ID")
    client_secret = _env("STRAVA_CLIENT_SECRET")
    refresh_token = _env("STRAVA_REFRESH_TOKEN")

    try:
        resp = requests.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=_HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise StravaAuthError(f"Strava OAuth nedostupné: {exc}") from exc

    if resp.status_code != 200:
        raise StravaAuthError(
            f"Strava OAuth vrátilo {resp.status_code}: {resp.text[:200]} "
            "(zkontroluj STRAVA_CLIENT_ID/_SECRET/_REFRESH_TOKEN)"
        )

    body = resp.json()
    new_refresh = body.get("refresh_token")
    if new_refresh and new_refresh != refresh_token:
        if persist_rotation:
            _persist_refresh_token(new_refresh)
        else:
            os.environ["STRAVA_REFRESH_TOKEN"] = new_refresh
            log.warning(
                "Strava vrátila nový refresh token, ale --dry-run ho neukládá do .env. "
                "Spusť ostrý běh, nebo ulož STRAVA_REFRESH_TOKEN ručně."
            )
    token = body.get("access_token")
    if not token:
        raise StravaAuthError("Strava OAuth neposlalo access_token.")
    return token
