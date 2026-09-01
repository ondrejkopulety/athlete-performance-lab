"""
quality.py  –  Kvalita tréninku
================================

Monotony, Strain, Whoop logaritmický strain, Seilerova polarizace (80/20)
s penalizací za Z3 „junk miles", efficiency index a fatigue index.

Vzorce jsou převzaté beze změny z původního athlete_analytics.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import MONOTONY_WINDOW
from src.ingestion.sport import cycling_mask

# Okno pro trend efektivity (dny). Lokální konstanta, ne ze settings –
# historická hodnota, kterou se výsledky drží.
EFFICIENCY_TREND_WINDOW = 14

# Grade-Adjusted Distance: metr stoupání se do jmenovatele efektivity počítá
# jako ekvivalent GRADE_ADJ_M_PER_ASCENT_M metru roviny. TRIMP (z tepu) už
# vyšší cenu kopce zachycuje, takže dělení HOLÝMI km dělá z kopcovitého dne
# falešnou „skrytou únavu" (vysoký TRIMP/km) – fatigue_index pak čte profil
# trasy jako fyziologii. Faktor 10 (≈ energetická cena výškového metru vůči
# jízdě po rovině) je hrubý, ale vrací terénní citlivost do řádu šumu.
GRADE_ADJ_M_PER_ASCENT_M = 10.0

# Strop REPORTOVANÉ monotonie. Při nízké odchylce a vysoké průměrné zátěži
# by hodnota nereálně explodovala; epsilon brání dělení nulou.
MONOTONY_CAP = 4.0
MONOTONY_EPSILON = 1e-5
# Strop monotonie vstupující do STRAINU. Foster strain se počítá z neořezané
# monotonie (viz compute_monotony_strain), ale degenerovaný týden se sedmi
# identickými nenulovými dny má SD ≈ 0 a poměr mean/SD by vyletěl do milionů.
# 10 je nad jakoukoli reálnou týdenní monotonií (naměřené maximum ~3,5),
# takže reálná data se nedotkne a stláčí jen tenhle patologický případ.
STRAIN_MONOTONY_CAP = 10.0

ZONE_COLUMNS = ["time_in_z1", "time_in_z2", "time_in_z3", "time_in_z4", "time_in_z5"]

# TRIMP/km má smysl porovnávat jen v rámci kola (míchání sportů by metriku
# znehodnotilo – kolo dává zhruba poloviční TRIMP/km oproti běhu). Elektrokolo
# je taky mimo (motor sráží TRIMP/km na zlomek). Klasifikace sportu žije
# jednotně v src/ingestion/sport.py – viz cycling_mask.


def compute_monotony_strain(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Fosterova monotonie a strain nad 7denním oknem denního TRIMP.

        Monotony = mean(TRIMP) / std(TRIMP)      (populační SD, ddof=0)
        Strain   = Monotony_raw × sum(TRIMP)
        Whoop    = 21 × (1 − exp(−0.005 × denní TRIMP))

    Dvě věci podle původní definice (Foster & Fitz-Gerald 1996):

      • **Populační směrodatná odchylka** (``ddof=0``). Pandas default je
        výběrová (``ddof=1``), což monotonii nad 7denním oknem podhodnocuje
        faktorem √(6/7) ≈ 0,93 – pásmo „vysoká > 2.0" by se rozsvěcelo
        později, než Foster zamýšlel.

      • **Strain z NEOŘEZANÉ monotonie.** Strop 4.0 (``MONOTONY_CAP``) platí
        jen na reportovanou ``monotony``; do strainu jde surová hodnota.
        V nejmonotónnějších týdnech (kde monotonie narazí na strop) je
        strain jinak uměle stlačený právě tam, kde má nejvíc varovat.
    """
    daily = daily.copy()
    roll = daily["trimp"].rolling(window=MONOTONY_WINDOW, min_periods=MONOTONY_WINDOW)
    roll_mean = roll.mean()
    roll_std = roll.std(ddof=0)
    roll_sum = roll.sum()

    monotony_raw = roll_mean / (roll_std + MONOTONY_EPSILON)
    daily["monotony"] = monotony_raw.clip(upper=MONOTONY_CAP).round(2)
    daily["strain"] = (monotony_raw.clip(upper=STRAIN_MONOTONY_CAP) * roll_sum).round(1)
    daily["whoop_strain"] = (21 * (1 - np.exp(-0.005 * daily["trimp"]))).round(2)
    return daily


def compute_polarization(daily: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    """
    14denní klouzavý rozbor rozložení intenzity.

    Low = Z1+Z2, High = Z4+Z5. Jmenovatelem je čas ve VŠECH zónách včetně Z3,
    aby „šedá zóna" polarizační skóre správně snižovala.

    (Dřívější `polarization_efficiency` zrušena – algebraicky vycházela
    ≈ 105 − 2·z3_junk_pct, tedy jen převrácený `z3_junk_pct` bez vlastní
    informace.)
    """
    daily = daily.copy()

    if activities is None or activities.empty or not all(c in activities.columns for c in ZONE_COLUMNS):
        for col in ("polarization_low_pct", "polarization_high_pct", "z3_junk_pct"):
            daily[col] = np.nan
        return daily

    m = activities[["date", *ZONE_COLUMNS]].copy()
    m["date"] = pd.to_datetime(m["date"])
    for c in ZONE_COLUMNS:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    m["low_min"] = m["time_in_z1"] + m["time_in_z2"]
    m["mid_min"] = m["time_in_z3"]
    m["high_min"] = m["time_in_z4"] + m["time_in_z5"]

    zone_daily = m.groupby("date")[["low_min", "mid_min", "high_min"]].sum()
    zone_daily = zone_daily.reindex(daily.index, fill_value=0)

    low_14d = zone_daily["low_min"].rolling(14, min_periods=3).sum()
    mid_14d = zone_daily["mid_min"].rolling(14, min_periods=3).sum()
    high_14d = zone_daily["high_min"].rolling(14, min_periods=3).sum()
    total_14d = (low_14d + mid_14d + high_14d).replace(0, np.nan)

    daily["polarization_low_pct"] = ((low_14d / total_14d) * 100).round(1)
    daily["polarization_high_pct"] = ((high_14d / total_14d) * 100).round(1)

    daily["z3_junk_pct"] = ((mid_14d / total_14d) * 100).round(1)
    return daily


def compute_efficiency_index(daily: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiency index = TRIMP / grade-adjusted km, agregovaně za den.

    Jmenovatel je vzdálenost upravená o převýšení
    (``km + ascent_m/1000 × GRADE_ADJ_M_PER_ASCENT_M``), ne holé km.
    TRIMP z tepu už vyšší cenu stoupání zachycuje, takže dělení holými km
    dělalo z kopcovitého dne falešnou „skrytou únavu" (vysoký TRIMP/km) a
    ``fatigue_index`` pak četl profil trasy jako fyziologii. Když
    ``ascent_m`` chybí, spadne to zpět na holé km.

    Agregace přes součty (ne průměr per-activity) záměrně: jinak by krátká
    intenzivní jízda s vysokým TRIMP/km převážila celodenní objem.

    Trend se počítá nad řídkými daty (jen dny s jízdou), aby dny volna
    metriku uměle nevyhlazovaly.

    Jízdy bez tepu se vynechávají: Strava import bez HR má čas v zónách 0,
    tedy total_trimp i trimp_adjusted 0, a sum(TRIMP)/sum(km) pak vyjde 0.
    Nula není „dokonalá efektivita" – je to chybějící vstup, který se dřív
    šířil do ef_trend (ffill) a fatigue_index (dělení nulou).
    """
    daily = daily.copy()
    daily["daily_efficiency"] = np.nan
    daily["ef_trend"] = np.nan

    if activities is None or activities.empty:
        return daily
    if not {"distance_km", "total_trimp", "sport"} <= set(activities.columns):
        return daily

    cardio = activities.loc[cycling_mask(activities["sport"])].copy()
    if cardio.empty:
        return daily

    cardio["date"] = pd.to_datetime(cardio["date"])
    cardio["distance_km"] = pd.to_numeric(cardio["distance_km"], errors="coerce")
    cardio["total_trimp"] = pd.to_numeric(cardio["total_trimp"], errors="coerce")
    cardio = cardio.dropna(subset=["distance_km", "total_trimp"])
    cardio = cardio.loc[cardio["distance_km"] > 0]

    # Jízda bez tepu: buď chybí avg_hr, nebo z ní nevznikla žádná zátěž.
    if "avg_hr" in cardio.columns:
        cardio = cardio.loc[pd.to_numeric(cardio["avg_hr"], errors="coerce").notna()]
    cardio = cardio.loc[cardio["total_trimp"] > 0]
    if cardio.empty:
        return daily

    ascent = pd.to_numeric(cardio.get("ascent_m"), errors="coerce").fillna(0.0).clip(lower=0.0)
    cardio["_gad_km"] = cardio["distance_km"] + ascent / 1000.0 * GRADE_ADJ_M_PER_ASCENT_M

    agg = cardio.groupby("date").agg(
        _sum_trimp=("total_trimp", "sum"),
        _sum_km=("_gad_km", "sum"),
    )
    eff = (agg["_sum_trimp"] / agg["_sum_km"]).to_frame("daily_efficiency")

    # Trend se počítá na ose od první do poslední jízdy – ne přes celý
    # kalendář, aby se ffill nešířil do období po posledním tréninku.
    ride_range = pd.date_range(eff.index.min(), eff.index.max(), freq="D")
    eff = eff.reindex(ride_range)
    eff["ef_trend"] = (
        eff["daily_efficiency"]
        .rolling(window=EFFICIENCY_TREND_WINDOW, min_periods=3)
        .mean()
        .ffill()
    )
    eff = eff.round(4).reindex(daily.index)

    daily["daily_efficiency"] = eff["daily_efficiency"]
    daily["ef_trend"] = eff["ef_trend"]
    return daily


def compute_fatigue_index(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Skrytá únava: dnešní efektivita vůči 7dennímu průměru.

    > 1.0 znamená, že organismus reaguje hůř na stejnou mechanickou práci.
    """
    daily = daily.copy()
    if "daily_efficiency" not in daily.columns:
        daily["fatigue_index"] = np.nan
        return daily

    eff = daily["daily_efficiency"]
    eff_7d = eff.rolling(7, min_periods=3).mean()
    # mask() nastaví jmenovatel na NaN, když je nula nebo záporný (žádný
    # trénink v okně) → fatigue_index bezpečně zůstane NaN místo dělení nulou.
    daily["fatigue_index"] = (eff / eff_7d.mask(eff_7d <= 0)).round(3)
    return daily
