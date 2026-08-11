"""
calendar.py  –  denní osa analytiky
====================================

Jediné místo, které rozhoduje, za které dny se metriky počítají.

Dvě pravidla, na kterých záleží:

  1. **Konec je vždy dnešek.** I když se dnes netrénovalo, den musí v tabulce
     existovat – jinak by se zahodila ranní biometrie (HRV, spánek, RHR),
     která dorazí ze zápěstí dřív než jakákoli aktivita.

  2. **Začátek je první aktivita**, ne první biometrie.

     Chvíli to bylo naopak – osa začínala nejstarším záznamem z jakéhokoli
     zdroje, aby se neztratila biometrie z doby před prvním tréninkem.
     Jenže Apple Health sahá do roku 2017, kdežto tréninková data začínají
     až 2022, takže vzniklo 1495 dní, kde TRIMP je nula, CTL i ATL jsou
     nula, a readiness_score proto vychází na pevných 75 bodů (fixní bod
     vzorce při TSB = 0, ne zapsaná konstanta). To vypadá jako data, ale
     není to nic – jen tvar rovnice. Průměry a korelace přes celou historii
     to táhlo k té konstantě.

     Biometrie bez jediného tréninku se tedy zahazuje záměrně: bez zátěže
     nemá připravenost k čemu být připravená.
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
    Souvislá denní osa (bez děr) od první aktivity po dnešek.

    `start` / `end` slouží pro inkrementální přepočet dílčího okna.
    Vrací prázdný index, pokud v databázi nejsou žádné aktivity.
    """
    end = end or today_local()
    if start is None:
        start = repo.min_activity_date(session)
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
