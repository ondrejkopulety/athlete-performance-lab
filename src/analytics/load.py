"""
load.py  –  Tréninková zátěž (PMC): TRIMP, CTL/ATL/TSB, ACWR
=============================================================

Vzorce jsou převzaté beze změny z původního athlete_analytics.py.

Poznámka k inkrementalitě: denní metriky se počítají **vždy nad celou
historií**, ne jen nad změněným oknem. Je to levné (řádově jednotky
milisekund na 1600 dní) a hlavně to je jediný způsob, jak zaručit
identický výsledek – ef_trend používá ffill s neomezenou pamětí a EMA
s 42denní konstantou nese vliv řádově rok zpátky, takže žádné rozumné
lookback okno by nedalo přesně stejná čísla.

Drahá část analytiky nikdy nebyly denní agregace, ale per-activity
metriky nad vteřinovými daty (DFA-alpha1, RSA, cardiac drift) – a ty
inkrementální jsou, viz activity.py a metrics_version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    ACWR_ACUTE_DAYS,
    ACWR_CHRONIC_DAYS,
    ATL_DAYS,
    CTL_DAYS,
    CTL_RAMP_WARN,
)

# Pěší sporty dostávají lineární koeficient 0.6 (−40 %). Bez něj 6–8hodinová
# Z1 túra nafoukne TRIMP natolik, že rozbije celý PMC. Lineární koeficient
# (na rozdíl od dřívějšího logaritmického tlumení) zachovává tvar rozdělení,
# takže Banisterův model i ACWR zůstávají matematicky v pořádku.
HIKING_TRIMP_COEFFICIENT = 0.6
HIKING_SPORT_PATTERN = "hiking|walking"

# EPOC-vážený TRIMP: 10 % EPOC skóre navrch, aby ACWR vnímalo i intenzitu,
# nejen objem.
EPOC_WEIGHT = 0.1


def ema_decay(series: pd.Series, span: int) -> pd.Series:
    """
    Exponenciální průměr s pravou časovou konstantou alpha = 1/N.

    pandas defaultně používá alpha = 2/(span+1); Banisterův PMC vyžaduje
    1/N, jinak nesedí 42denní/7denní konstanty.
    """
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.ewm(alpha=1.0 / span, adjust=False).mean()


def build_daily_load(
    activities: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Denní agregace tréninkové zátěže na zadanou osu.

    Vrací DataFrame indexovaný datem se sloupci trimp, trimp_epoc,
    epoc_score_daily, recovery_tax_hours_daily.

    Dny bez tréninku mají trimp = 0 (to je fyziologicky správně – netrénoval
    jsem, zátěž je nula). Biometrické sloupce se tu záměrně needitují,
    ty zůstávají NULL, dokud nedorazí data ze zápěstí.
    """
    empty = pd.DataFrame(
        0.0,
        index=calendar,
        columns=["trimp", "trimp_epoc", "epoc_score_daily", "recovery_tax_hours_daily"],
    )
    empty.index.name = "date"
    if activities is None or activities.empty:
        return empty

    df = activities.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in ("total_trimp", "epoc_score", "recovery_tax_hours"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    # Přednost má TRIMP přepočítaný z klidového tepu platného k datu
    # aktivity; total_trimp z parseru (pevná konstanta) slouží jako záloha
    # pro aktivity, které ještě nemají odvozené metriky.
    if "trimp_adjusted" in df.columns:
        adjusted = pd.to_numeric(df["trimp_adjusted"], errors="coerce")
        df["total_trimp"] = adjusted.fillna(df["total_trimp"])

    sport = df["sport"].astype(str).str.lower().fillna("") if "sport" in df.columns else pd.Series("", index=df.index)
    is_hiking = sport.str.contains(HIKING_SPORT_PATTERN, na=False)
    if is_hiking.any():
        df.loc[is_hiking, "total_trimp"] = df.loc[is_hiking, "total_trimp"] * HIKING_TRIMP_COEFFICIENT

    daily = df.groupby("date").agg(
        trimp=("total_trimp", "sum"),
        epoc_score_daily=("epoc_score", "sum"),
        recovery_tax_hours_daily=("recovery_tax_hours", "sum"),
    )
    daily = daily.reindex(calendar, fill_value=0.0)
    daily.index.name = "date"
    daily["trimp_epoc"] = daily["trimp"] + daily["epoc_score_daily"] * EPOC_WEIGHT
    return daily[["trimp", "trimp_epoc", "epoc_score_daily", "recovery_tax_hours_daily"]]


def compute_ctl_atl_tsb(daily: pd.DataFrame) -> pd.DataFrame:
    """
    CTL (fitness), ATL (únava), TSB (forma).

    TSB záměrně používá **včerejší** CTL a ATL – reprezentuje ranní formu
    PŘED dnešním tréninkem, což je stav, na základě kterého se ráno
    rozhoduje, jak tvrdě jet.
    """
    daily = daily.copy()
    daily["ctl"] = ema_decay(daily["trimp"], CTL_DAYS)
    daily["atl"] = ema_decay(daily["trimp"], ATL_DAYS)
    daily["tsb"] = daily["ctl"].shift(1) - daily["atl"].shift(1)
    return daily


def compute_acwr(daily: pd.DataFrame) -> pd.DataFrame:
    """
    ACWR = 7denní průměr / 28denní průměr z EPOC-vážené TRIMP.

    Bez clipování – hodnoty > 2.0 jsou legitimní signál přetrénování
    a konzument (dashboard, chatbot) si je má vyhodnotit sám.

    Navíc CTL ramp rate = týdenní přírůstek CTL.
    """
    daily = daily.copy()
    trimp_col = daily["trimp_epoc"] if "trimp_epoc" in daily.columns else daily["trimp"]

    acute = trimp_col.rolling(ACWR_ACUTE_DAYS, min_periods=ACWR_ACUTE_DAYS).mean()
    chronic = trimp_col.rolling(ACWR_CHRONIC_DAYS, min_periods=ACWR_CHRONIC_DAYS).mean()
    daily["acwr"] = (acute / chronic.replace(0, np.nan)).round(2)

    if "ctl" in daily.columns:
        daily["ctl_ramp_rate"] = (daily["ctl"] - daily["ctl"].shift(7)).round(2)
        daily["ctl_ramp_warning"] = daily["ctl_ramp_rate"] > CTL_RAMP_WARN
    else:
        daily["ctl_ramp_rate"] = np.nan
        daily["ctl_ramp_warning"] = False

    return daily


def recommend(tsb: float) -> str:
    """Slovní doporučení podle TSB (používá CLI shrnutí i coach advice)."""
    if pd.isna(tsb):
        return "Nedostatek dat."
    if tsb < -30:
        return "Vysoké riziko zranění! Okamžitý odpočinek."
    elif tsb < -10:
        return "Optimální tréninková zátěž."
    elif tsb <= 5:
        return "Udržovací režim."
    elif tsb <= 10:
        return "Čerstvost – můžeš zvýšit zátěž."
    else:
        return "Čerstvost / Připravenost na výkon."
