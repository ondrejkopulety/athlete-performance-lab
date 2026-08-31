"""
sport.py  –  kanonický název sportu z FIT session zprávy
=========================================================

Jediné místo, které rozhoduje, jak se `sport` + `sub_sport` složí do řetězce
uloženého v `activities.sport`.

Proč vlastní modul: původní kód (čtyři nezávislé kopie téhož `if s: … if ss: …`)
produkoval tři různé tvary téhož sportu –

    "hiking"            sub_sport ve zprávě chyběl
    "hiking/generic"    sub_sport tam byl
    "64/94"             enum hodnota mimo tabulku fitparse

– takže `GROUP BY sport` počítal jednu turistiku jako dvě a squash jako
neznámé číslo. Kanonický tvar je proto **vždy** `"hlavní/pod"`, malými
písmeny, s `generic` jako výchozím pod-sportem.

Na metriky to nemá vliv: všichni konzumenti sloupce (`load.py` koeficient
turistiky, `activity.py` výběr kardio aktivit, `quality.py` efektivita,
`exports.py` cycling_summary, API filtr) hledají podřetězec v celém řetězci,
takže "hiking" i "hiking/generic" jim matchovaly stejně už předtím. Jediná
výjimka – `fit_parser` splituje na "/" kvůli prahu rychlosti – kanonickému
tvaru vyhovuje líp než holému.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fitparse.profile import FIELD_TYPES

if TYPE_CHECKING:
    import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# ENUM HODNOTY NAD RÁMEC FITPARSE
# ─────────────────────────────────────────────────────────────────────────────
# fitparse 1.2.0 má profilovou tabulku vygenerovanou ze staršího FIT SDK:
# sport zná do 48, sub_sport do 59. Novější hodnoty vrátí `get_values()` jako
# syrový int, ze kterého původní kód udělal `str(64)`.
#
# Hodnoty níž jsou opsané z parseru GoldenCheetah (src/FileIO/FitRideFile.cpp,
# funkce getSport() / getSubSport()), který profil udržuje aktuální.
# Doplňujeme jen kódy, které se v datech skutečně vyskytly, plus ty, které
# s nimi sousedí – hádat celý zbytek tabulky nemá smysl.
FIT_SPORT_EXTRA: dict[int, str] = {
    62: "hiit",
    64: "racket",
    65: "wheelchair_push_walk",
    73: "hockey",
    84: "pickleball",
}

FIT_SUB_SPORT_EXTRA: dict[int, str] = {
    62: "breathing",
    91: "ice",
    94: "squash",
}

DEFAULT_SUB_SPORT = "generic"

# Různé zdroje (Strava sport_type, novější FIT enumy, ruční zápis) pojmenují
# kolo mimo kanonickou rodinu "cycling/*": Strava posílá "MountainBikeRide"
# → "mountain_biking", "GravelRide" → "gravel_cycling", "Ride" → "ride".
# Bez překladu je konzument hledající podřetězec "cycl" nenajde – MTB pak
# vypadne z kardio aktivit a cardiac_drift/durability ho vyhodnotí jako běh.
# Klíč je vždy resolved hlavní sport malými písmeny.
SPORT_ALIASES: dict[str, tuple[str, str]] = {
    "mountain_biking":  ("cycling", "mountain"),
    "mountainbiking":   ("cycling", "mountain"),
    "mtb":              ("cycling", "mountain"),
    "gravel_cycling":   ("cycling", "gravel"),
    "gravel_ride":      ("cycling", "gravel"),
    "road_cycling":     ("cycling", "road"),
    "virtual_cycling":  ("cycling", "virtual"),
    "virtual_ride":     ("cycling", "virtual"),
    "ride":             ("cycling", DEFAULT_SUB_SPORT),
    "bike":             ("cycling", DEFAULT_SUB_SPORT),
    "biking":           ("cycling", DEFAULT_SUB_SPORT),
    "e_bike_ride":      ("cycling", "e_bike"),
    "handcycling":      ("cycling", "hand"),
}

# Vlastní tabulka fitparse – ptáme se jí dřív než na doplňky výš, aby zdrojem
# pravdy zůstala knihovna a případný její upgrade nás přebil.
_FITPARSE_SPORT: dict[int, str] = FIELD_TYPES["sport"].values
_FITPARSE_SUB_SPORT: dict[int, str] = FIELD_TYPES["sub_sport"].values


def _resolve(value: object, base: dict[int, str], extra: dict[int, str]) -> str | None:
    """
    Jméno sportu z hodnoty, kterou vrátil fitparse.

    Známý enum už fitparse rozložil na řetězec, neznámý zůstal jako int.
    Vrací None jen pro chybějící hodnotu – nula je platný enum (`generic`),
    takže se nesmí testovat pravdivostí.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        name = base.get(value) or extra.get(value)
        # Nerozpoznaný kód pojmenujeme tak, aby bylo poznat, že je neznámý –
        # tvrdit "racket" u kódu, který jsme neověřili, by bylo horší než
        # přiznat neznalost.
        return (name or f"unknown_{value}").strip().lower()
    name = str(value).strip().lower()
    return name or None


def normalize_sport(sport: object, sub_sport: object) -> str:
    """
    Kanonický `"hlavní/pod"` z hodnot `sport` a `sub_sport` session zprávy.

        ("cycling", "gravel_cycling")  → "cycling/gravel_cycling"
        ("hiking",  None)              → "hiking/generic"
        (64, 94)                       → "racket/squash"
        (None, None)                   → ""

    Prázdný řetězec znamená, že session zpráva sport neuvedla vůbec –
    volající se pak zachová stejně jako dřív (žádný sport není chyba,
    jen chybějící metadata).
    """
    main = _resolve(sport, _FITPARSE_SPORT, FIT_SPORT_EXTRA)
    sub = _resolve(sub_sport, _FITPARSE_SUB_SPORT, FIT_SUB_SPORT_EXTRA)

    if main is None:
        # Sub-sport bez hlavního sportu nedává smysl jako "/squash";
        # ber ho jako hlavní, ať se informace neztratí.
        if sub is None:
            return ""
        main, sub = sub, None

    # Sjednocení názvů z jiných zdrojů na kanonickou rodinu (viz SPORT_ALIASES).
    # Alias nepřebíjí konkrétnější sub-sport, který zpráva uvedla explicitně.
    if main in SPORT_ALIASES:
        alias_main, alias_sub = SPORT_ALIASES[main]
        main = alias_main
        if sub in (None, DEFAULT_SUB_SPORT):
            sub = alias_sub

    return f"{main}/{sub or DEFAULT_SUB_SPORT}"


# Garmin ukládá elektrokolo jako sport `cycling` se sub-sportem `e_bike_fitness`
# (28) nebo `e_bike_mountain` (47), takže v kanonickém názvu je vždycky podřetězec
# `e_bike`. Regex bere i tvary `ebike` / `e-bike` pro případ ručně zadaného sportu.
EBIKE_SPORT_PATTERN = r"e[-_ ]?bike"
_EBIKE_RE = re.compile(EBIKE_SPORT_PATTERN, re.IGNORECASE)


def is_ebike(sport: object) -> bool:
    """
    True, pokud je aktivita elektrokolo.

    Elektro se počítá do zátěže a formy (TRIMP → CTL/ATL/TSB, minuty v zónách),
    ale NE do objemu cyklistiky (km, převýšení, počet jízd) ani do metrik, které
    stojí na poměru výkon/rychlost ↔ tep – ten motor rozbíjí. Konkrétně mimo:
    efektivitu (TRIMP/km), cardiac drift, DFA prahy, odhad LTHR, tepovou křivku,
    HRR a VAM. Jediné místo, kde se elektro rozpoznává – volá `normalize_sport`
    výš přes uložený `activities.sport`.
    """
    if sport is None:
        return False
    return bool(_EBIKE_RE.search(str(sport)))


# ─────────────────────────────────────────────────────────────────────────────
# KLASIFIKACE SPORTU  –  jediný zdroj pro celou analytiku i API
# ─────────────────────────────────────────────────────────────────────────────
# Regexy jsou psané tak, aby fungovaly jak pro pandas `str.contains`, tak pro
# SQL `Activity.sport.op("~*")`. Dřív žila stejná logika v pěti kopiích
# (EFFICIENCY_SPORT_PATTERN v quality.py, CYCLING_SPORT_PATTERN v exports.py
# i dashboard.py, CARDIO_SPORTS v activity.py, ad-hoc `"cycling" in sport`),
# každá s trochu jiným výčtem – MTB nebo "Ride" ze Stravy pak propadl.
CYCLING_SPORT_REGEX = r"cycl|biking|ride"
RUNNING_SPORT_REGEX = r"run(?!way)|jog"
CARDIO_SPORT_REGEX = rf"(?:{CYCLING_SPORT_REGEX})|(?:{RUNNING_SPORT_REGEX})"

_CYCLING_RE = re.compile(CYCLING_SPORT_REGEX, re.IGNORECASE)
_RUNNING_RE = re.compile(RUNNING_SPORT_REGEX, re.IGNORECASE)


def is_cycling_sport(sport: object) -> bool:
    """True pro jakoukoli formu kola (silnice, gravel, MTB, dráha, i e-bike)."""
    return sport is not None and bool(_CYCLING_RE.search(str(sport)))


def is_running_sport(sport: object) -> bool:
    """True pro běh (silniční, terénní, na páse)."""
    return sport is not None and bool(_RUNNING_RE.search(str(sport)))


def is_cardio_sport(sport: object) -> bool:
    """
    True pro kardio aktivity s vypovídajícím vztahem výkon/rychlost ↔ tep –
    tj. kolo a běh. Elektrokolo se sem nepočítá (motor ten vztah rozbíjí).
    """
    return (is_cycling_sport(sport) or is_running_sport(sport)) and not is_ebike(sport)


def _sport_mask(sport: pd.Series, regex: str, *, exclude_ebike: bool) -> pd.Series:
    """
    Vektorová klasifikace nad ``activities["sport"]``. Vrací bool Series se
    stejným indexem jako vstup.

    Import pandas je uvnitř schválně – ``sport.py`` je ingest primitiv a nemá
    si tahat pandas do každého importu jen kvůli téhle jedné funkci.
    """
    s = sport.astype("string").fillna("")
    mask = s.str.contains(regex, case=False, regex=True, na=False)
    if exclude_ebike:
        mask &= ~s.str.contains(EBIKE_SPORT_PATTERN, case=False, regex=True, na=False)
    return mask


def cycling_mask(sport: pd.Series, *, exclude_ebike: bool = True) -> pd.Series:
    """Bool Series pro cyklistiku (bez elektrokola, pokud ``exclude_ebike``)."""
    return _sport_mask(sport, CYCLING_SPORT_REGEX, exclude_ebike=exclude_ebike)


def cardio_mask(sport: pd.Series, *, exclude_ebike: bool = True) -> pd.Series:
    """Bool Series pro kardio (kolo + běh)."""
    return _sport_mask(sport, CARDIO_SPORT_REGEX, exclude_ebike=exclude_ebike)
