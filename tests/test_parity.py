"""
Regresní kontrola proti historickému výstupu
=============================================

Původní úkol tohohle souboru byl prokázat, že refaktor z CSV do databáze
nezměnil čísla. Ten úkol je splněný: v době refaktoru sedělo 33 ze 37
sloupců na bit a čtyři odchylky měly vlastní zdůvodnění.

Od té doby se metriky **záměrně** změnily:

  • TRIMP se počítá z klidového tepu platného k datu aktivity, ne z pevné
    konstanty 41 bpm → posunulo se CTL, ATL, TSB, ACWR, monotony, strain
  • biometrie má nově i Apple Health, takže dny před srpnem 2025 mají
    hodnoty tam, kde dřív byly prázdné → rhr_day, spánek, recovery, readiness
  • vlajka klidového tepu je relativní → stress_flags, illness_warning

Porovnávat sloupec po sloupci proti souboru, který popisuje jiný model,
by znamenalo test složený převážně z výjimek – a takový test nic nehlídá.
Zůstávají proto jen tvrzení, která platí bez ohledu na změnu vzorců:

  1. žádný den z historie se neztratil
  2. metriky nezávislé na klidovém tepu se nezměnily vůbec
  3. metriky závislé na klidovém tepu se posunuly jen v očekávaném řádu

Bod 3 je tu ten podstatný: chytí, kdyby změna utekla do nesmyslu, ale
nepadá kvůli tomu, že změna vůbec nastala.

Vzorce samotné hlídá tests/test_metrics.py (fyziologické invarianty)
a tests/test_rhr_baseline.py (přepočet TRIMP proti parseru).
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import SUMMARIES_DIR
from src.db import repository as repo

BASELINE_CSV = SUMMARIES_DIR / "athlete_readiness.csv"
RENAME = {"CTL": "ctl", "ATL": "atl", "TSB": "tsb", "fluid_loss_L_daily": "fluid_loss_l_daily"}

# Metriky odvozené jen z času v tepových zónách. Zóny jsou pevné, takže
# na změnu klidového tepu ani na doplnění biometrie reagovat nesmí.
RHR_INDEPENDENT = ["polarization_low_pct", "polarization_high_pct"]

# Maximální přijatelný posun metrik závislých na TRIMP, v jejich vlastních
# jednotkách. Klidový tep se zvedl ze 41 na 47–52, takže tepová rezerva
# klesla o ~5 % – a o zhruba tolik klesl i TRIMP. Naměřeno: CTL p95 3,9.
MAX_SHIFT = {"ctl": 8.0, "atl": 15.0, "tsb": 12.0}


BASELINE_ACTIVITIES_CSV = SUMMARIES_DIR / "master_high_res_summary.csv"


def _comparable_until(session):
    """
    Poslední den, který smí baseline popisovat.

    Sada aktivit se od vzniku baseline změnila dvěma způsoby: sync stáhl
    nové tréninky a oprava deduplikace vyměnila jeden Strava soubor za
    Garmin originál (jiný soubor = jiné minuty v zónách). Obojí legitimně
    mění metriky od svého data dál – rolling okna se dívají dozadu, takže
    starší dny zůstávají nedotčené.

    Bez téhle hranice by test padal na změnách, které jsou v pořádku.
    """
    from datetime import timedelta

    if not BASELINE_ACTIVITIES_CSV.exists():
        return None
    baseline_ids = set(
        pd.read_csv(
            BASELINE_ACTIVITIES_CSV, dtype={"activity_id": str},
            usecols=["activity_id"], low_memory=False,
        )["activity_id"]
    )
    activities = repo.read_activities(session, with_metrics=False)
    if activities.empty:
        return None

    unknown = activities[~activities["activity_id"].isin(baseline_ids)]
    if unknown.empty:
        return None
    return min(unknown["date"]) - timedelta(days=1)


@pytest.fixture(scope="module")
def frames():
    from src.db.session import SessionLocal, check_connection

    if not BASELINE_CSV.exists():
        pytest.skip(f"Baseline {BASELINE_CSV} neexistuje")
    if not check_connection():
        pytest.skip("Databáze neběží")

    csv = pd.read_csv(BASELINE_CSV, low_memory=False)
    csv["date"] = pd.to_datetime(csv["date"]).dt.date
    csv = csv.rename(columns=RENAME).set_index("date")

    s = SessionLocal()
    try:
        db = repo.read_daily_metrics(s)
        cutoff = _comparable_until(s)
    finally:
        s.close()
    if db.empty:
        pytest.skip("Žádné denní metriky – spusť `python scripts/main.py analyze`")

    db["date"] = pd.to_datetime(db["date"]).dt.date
    db = db.set_index("date")

    common = csv.index.intersection(db.index)
    if cutoff is not None:
        common = common[common <= cutoff]
        if len(common) == 0:
            pytest.skip("Všechny dny baseline jsou ovlivněné novými daty")
    return csv, db, common


def test_no_historical_day_was_lost(frames):
    """Databáze musí obsahovat každý den, který byl v historickém výstupu."""
    csv, db, _ = frames  # záměrně mimo okno srovnatelnosti – ztráta dne je vždy chyba
    missing = set(csv.index) - set(db.index)
    assert not missing, f"Chybí dny: {sorted(missing)[:5]}"


def test_history_was_extended_not_replaced(frames):
    """Nová data smí historii jen rozšířit, ne zkrátit."""
    csv, db, _ = frames
    assert min(db.index) <= min(csv.index)
    assert max(db.index) >= max(csv.index)


@pytest.mark.parametrize("column", RHR_INDEPENDENT)
def test_zone_based_metrics_are_untouched(frames, column):
    """
    Polarizace vychází z minut v zónách. Zóny se nezměnily, takže tyhle
    hodnoty musí sedět přesně – kdyby ne, změna klidového tepu prosákla
    někam, kam neměla.
    """
    csv, db, common = frames
    a = pd.to_numeric(csv.loc[common, column], errors="coerce")
    b = pd.to_numeric(db.loc[common, column], errors="coerce")

    both = a.notna() & b.notna()
    assert both.any()
    assert (a[both] - b[both]).abs().max() < 1e-6


@pytest.mark.parametrize("column", sorted(MAX_SHIFT))
def test_load_metrics_shifted_within_expected_range(frames, column):
    """
    TRIMP se změnil záměrně, ale posun musí odpovídat velikosti změny
    klidového tepu. Výrazně větší rozdíl by znamenal chybu v přepočtu,
    ne jinou vstupní konstantu.
    """
    csv, db, common = frames
    a = pd.to_numeric(csv.loc[common, column], errors="coerce")
    b = pd.to_numeric(db.loc[common, column], errors="coerce")

    both = a.notna() & b.notna()
    assert both.any()

    delta = b[both] - a[both]
    assert delta.abs().max() <= MAX_SHIFT[column], (
        f"{column}: posun až {delta.abs().max():.1f} přesahuje očekávaných "
        f"{MAX_SHIFT[column]} – zkontroluj přepočet TRIMP"
    )
    # Směr: vyšší klidový tep = menší tepová rezerva = nižší zátěž
    assert delta.median() <= 0.5


def test_biometrics_were_only_added(frames):
    """
    Apple Health smí prázdné dny doplnit, ale nesmí přepsat den, kde už
    Garmin hodnotu měl – ten má prioritu.
    """
    csv, db, common = frames
    a = pd.to_numeric(csv.loc[common, "rhr_day"], errors="coerce")
    b = pd.to_numeric(db.loc[common, "rhr_day"], errors="coerce")

    lost = a.notna() & b.isna()
    assert not lost.any(), f"Ztraceno {int(lost.sum())} dní s klidovým tepem"

    both = a.notna() & b.notna()
    assert (a[both] - b[both]).abs().max() < 1e-6, "Garmin hodnoty se změnily"


def test_rest_days_have_zero_not_null(frames):
    """Dny volna: NULL → 0. Netrénoval jsem, zátěž je nula, ne neznámo."""
    csv, db, common = frames
    for column in ("epoc_score_daily", "recovery_tax_hours_daily"):
        a = pd.to_numeric(csv.loc[common, column], errors="coerce")
        b = pd.to_numeric(db.loc[common, column], errors="coerce")
        only_db = a.isna() & b.notna()
        assert (b[only_db] == 0).all()
        assert not (a.notna() & b.isna()).any()
