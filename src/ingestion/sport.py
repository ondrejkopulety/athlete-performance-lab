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

from typing import Optional

from fitparse.profile import FIELD_TYPES

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

# Vlastní tabulka fitparse – ptáme se jí dřív než na doplňky výš, aby zdrojem
# pravdy zůstala knihovna a případný její upgrade nás přebil.
_FITPARSE_SPORT: dict[int, str] = FIELD_TYPES["sport"].values
_FITPARSE_SUB_SPORT: dict[int, str] = FIELD_TYPES["sub_sport"].values


def _resolve(value: object, base: dict[int, str], extra: dict[int, str]) -> Optional[str]:
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

    return f"{main}/{sub or DEFAULT_SUB_SPORT}"
