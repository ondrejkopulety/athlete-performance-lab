"""
calendar.py  –  denní osa analytiky
====================================

Jediné místo, které rozhoduje, za které dny se metriky počítají.

Dvě pravidla, na kterých záleží:

  1. **Konec je vždy dnešek.** I když se dnes netrénovalo, den musí v tabulce
     existovat – jinak by se zahodila ranní biometrie (HRV, spánek, RHR),
     která dorazí ze zápěstí dřív než jakákoli aktivita.

  2. **Začátek je nejstarší záznam z JAKÉHOKOLI zdroje**, ne jen z aktivit.
     Původní implementace startovala první aktivitou, takže biometrie
     z období před prvním zaznamenaným tréninkem se ztrácela.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from src.db import repository as repo


def today_local() -> date:
    """
    Dnešní datum v lokálním čase atleta.

    Garmin reportuje denní data v lokálním čase, takže UTC by kolem půlnoci
    posouvalo dny (a v létě by „dnešek" končil ve 2:00 ráno).
    """
    return pd.Timestamp.now().normalize().date()


def build_calendar(
    session: Session,
    start: date | None = None,
    end: date | None = None,
) -> pd.DatetimeIndex:
    """
    Souvislá denní osa (bez děr) od nejstaršího dat po dnešek.

    `start` / `end` slouží pro inkrementální přepočet dílčího okna.
    Vrací prázdný index, pokud v databázi nejsou žádná data.
    """
    end = end or today_local()
    if start is None:
        start = repo.min_data_date(session)
    if start is None:
        return pd.DatetimeIndex([], name="date")
    if start > end:
        start = end

    idx = pd.date_range(start=start, end=end, freq="D")
    idx.name = "date"
    return idx


def lookback_start(dirty_from: date, lookback_days: int) -> date:
    """
    Počátek okna, které je potřeba načíst, aby rolling okna měla plný kontext.

    Přepočítáváme sice jen dny od `dirty_from`, ale monotony (7 d),
    ACWR (28 d), polarizace (14 d) nebo HRV z-score (30 d) potřebují vidět
    i historii před nimi – jinak by na hraně okna vyšla jiná čísla než
    při plném přepočtu.
    """
    return dirty_from - timedelta(days=lookback_days)
