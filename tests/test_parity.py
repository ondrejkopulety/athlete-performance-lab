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
  • parser slučuje fragmenty Strava exportů, doplňuje mezery v tepu a počítá
    i stacionární tep → minuty v zónách vzrostly, u fotbalu z 1,9 na 92,9
    minuty; s nimi CTL, ATL a polarizace
  • osa začíná první aktivitou, ne první biometrií → dny před 2022-01-28
    v databázi nejsou, ačkoli v baseline CSV jsou

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

from config.settings import DATA_DIR
from src.db import repository as repo

# Zmrazená kopie, ne živý export – jinak by se výstup porovnával sám se sebou
BASELINE_DIR = DATA_DIR / "_baseline"
BASELINE_CSV = BASELINE_DIR / "athlete_readiness.csv"
RENAME = {"CTL": "ctl", "ATL": "atl", "TSB": "tsb", "fluid_loss_L_daily": "fluid_loss_l_daily"}

# Metriky odvozené z času v tepových zónách. Dřív se porovnávaly na bit –
# zóny byly pevné, takže na klidový tep reagovat nesměly. Od opravy parseru
# to neplatí: fragmentované Strava soubory přispívaly zlomkem svých minut
# (fotbal 1,9 místo 92,9), takže baseline popisuje jiné rozložení intenzity.
ZONE_DERIVED = ["polarization_low_pct", "polarization_high_pct"]

# Maximální přijatelný posun metrik závislých na TRIMP, v jejich vlastních
# jednotkách. Na posun působí dva vlivy proti sobě:
#
#   • klidový tep se zvedl ze 41 na 47–52, takže tepová rezerva klesla
#     o ~5 % a s ní i TRIMP  → dolů
#   • parser přestal zahazovat minuty fragmentovaných Strava souborů
#     a stacionárních sportů → nahoru, a mnohem víc
#
# Druhý vliv převažuje: naměřeno CTL medián +6,6 / max +47,5, ATL +3,7 /
# +85,2. Meze mají headroom, aby test nepadal na běžném kolísání dat, ale
# chytil řádovou chybu (odstraněný strop 120 s, vypnutá pojistka).
MAX_SHIFT = {"ctl": 60.0, "atl": 100.0, "tsb": 45.0}

# Akumulátory zátěže: oprava parseru jim může jen přidat, takže se u nich
# navíc hlídá směr. TSB je rozdíl CTL a ATL – přidaná zátěž zvedne ATL
# rychleji než CTL, takže legitimně klesá (naměřeno až −37,8) a směr ani
# hloubku propadu u něj tvrdit nelze.
ACCUMULATORS = ("ctl", "atl")

# Kolik smí akumulátor klesnout. Klesat má jen vlivem klidového tepu, což
# jsou jednotky bodů (naměřeno CTL −1,0, ATL −5,6). Větší propad by znamenal,
# že se někde ztratila zátěž, ne že se přepočítala.
MAX_DROP = 12.0


BASELINE_ACTIVITIES_CSV = BASELINE_DIR / "master_high_res_summary.csv"


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


def test_no_day_inside_the_axis_was_lost(frames):
    """
    Uvnitř osy nesmí chybět ani jeden den, který byl v historickém výstupu.

    Dny před začátkem osy chybět mají: baseline sahá do roku 2017, kdy
    ještě žádný trénink neexistoval, a takové dny se dnes zahazují záměrně
    (viz src/analytics/calendar.py). Ztráta dne uvnitř osy je pořád chyba.
    """
    csv, db, _ = frames  # záměrně mimo okno srovnatelnosti
    start = min(db.index)
    missing = {d for d in csv.index if d >= start} - set(db.index)
    assert not missing, f"Chybí dny: {sorted(missing)[:5]}"


def test_history_reaches_at_least_as_far_forward(frames):
    """Nová data smí historii dopředu jen rozšířit, ne zkrátit."""
    csv, db, _ = frames
    assert max(db.index) >= max(csv.index)


@pytest.mark.parametrize("column", ZONE_DERIVED)
def test_zone_based_metrics_kept_their_coverage(frames, column):
    """
    Oprava parseru smí polarizaci posunout, ale ne ji nikde zahodit.

    Den, který v baseline hodnotu měl, ji musí mít i teď: minuty v zónách
    po opravě jen přibyly, takže jmenovatel čtrnáctidenního okna nemůže
    spadnout na nulu tam, kde dřív nula nebyl.
    """
    csv, db, common = frames
    a = pd.to_numeric(csv.loc[common, column], errors="coerce")
    b = pd.to_numeric(db.loc[common, column], errors="coerce")

    lost = a.notna() & b.isna()
    assert not lost.any(), f"{column}: ztraceno {int(lost.sum())} dní"


def test_zone_minutes_only_grew():
    """
    Součet minut v zónách po opravě parseru jen roste.

    Tohle je vlastní tvrzení opravy: sloučení fragmentů, doplněný tep ani
    započítaný stacionární tep nemůžou žádné aktivitě minuty ubrat. Kdyby
    některá klesla, je to regrese – proto se porovnává aktivita po aktivitě,
    ne až agregát.
    """
    if not BASELINE_ACTIVITIES_CSV.exists():
        pytest.skip(f"Baseline {BASELINE_ACTIVITIES_CSV} neexistuje")
    from src.db.session import SessionLocal, check_connection

    if not check_connection():
        pytest.skip("Databáze neběží")

    zones = ["time_in_z1", "time_in_z2", "time_in_z3", "time_in_z4", "time_in_z5"]
    old = pd.read_csv(
        BASELINE_ACTIVITIES_CSV, dtype={"activity_id": str},
        usecols=["activity_id", *zones], low_memory=False,
    ).set_index("activity_id")

    s = SessionLocal()
    try:
        new = repo.read_activities(s, with_metrics=False)
    finally:
        s.close()
    if new.empty:
        pytest.skip("Žádné aktivity – spusť `python scripts/main.py load`")
    new = new.set_index("activity_id")

    common = old.index.intersection(new.index)
    assert len(common) > 0

    old_sum = old.loc[common, zones].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    new_sum = new.loc[common, zones].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Tolerance na zaokrouhlení minut; pod ní jde o tutéž hodnotu.
    shrank = new_sum < old_sum - 0.05
    assert not shrank.any(), (
        f"{int(shrank.sum())} aktivit ztratilo minuty v zónách, "
        f"např. {shrank[shrank].index[:3].tolist()}"
    )


def test_zone_minutes_never_exceed_duration():
    """
    Součet minut v zónách nesmí přerůst délku aktivity.

    Zóny i aktivní čas se v parseru plní na jedné a téže podmínce, takže
    tohle je strukturální vlastnost, ne odhad. Kdyby padlo, znamená to, že
    se některá vteřina započítala do zóny, ale ne do trvání – přesně ten
    druh chyby, kvůli které fotbal vycházel na 1,9 minuty ze 65.

    Tolerance je na zaokrouhlení: pět sloupců po dvou desetinných místech
    dá dohromady až 0,025 min. Naměřeno: nejhorší případ 0,01 min (0,6 s).
    """
    from src.db.session import SessionLocal, check_connection

    if not check_connection():
        pytest.skip("Databáze neběží")

    zones = ["time_in_z1", "time_in_z2", "time_in_z3", "time_in_z4", "time_in_z5"]
    s = SessionLocal()
    try:
        acts = repo.read_activities(s, with_metrics=False)
    finally:
        s.close()
    if acts.empty:
        pytest.skip("Žádné aktivity – spusť `python scripts/main.py load`")

    total = acts[zones].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    duration = pd.to_numeric(acts["duration_minutes"], errors="coerce")

    over = total - duration
    worst = over.max()
    assert worst <= 0.05, (
        f"{int((over > 0.05).sum())} aktivit má víc minut v zónách než trvání, "
        f"nejhorší o {worst:.2f} min"
    )


@pytest.mark.parametrize("column", sorted(MAX_SHIFT))
def test_load_metrics_shifted_within_expected_range(frames, column):
    """
    Zátěž se změnila záměrně, ale posun musí zůstat v řádu, který odpovídá
    doplněným minutám. Výrazně větší rozdíl by znamenal chybu v přepočtu,
    ne jiná vstupní data.
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

    if column in ACCUMULATORS:
        # Klesat smí jen vlivem vyššího klidového tepu – jednotky bodů.
        assert delta.min() >= -MAX_DROP, (
            f"{column}: propad až {delta.min():.1f} – někde se ztratila zátěž"
        )
        # Parser dřív zahazoval minuty, takže na medián může jen přidat.
        assert delta.median() > 0, (
            f"{column}: medián posunu {delta.median():.1f} – doplněné minuty "
            f"se do zátěže nepromítly"
        )


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
    # Vědomá odchylka: recovery_tax_hours_daily se tu už netestuje. Metrika
    # byla zrušena jako vymyšlená (Spearman s TRIMP 0.9955, tedy žádná
    # informace navíc; proti Garminovu měření RMSE 32.0 h, což je
    # k nerozeznání od nejlepší možné konstanty 32.1 h).
    # V baseline CSV sloupec zůstává, v databázi pro něj místo není.
    csv, db, common = frames
    for column in ("epoc_score_daily",):
        a = pd.to_numeric(csv.loc[common, column], errors="coerce")
        b = pd.to_numeric(db.loc[common, column], errors="coerce")
        only_db = a.isna() & b.notna()
        assert (b[only_db] == 0).all()
        assert not (a.notna() & b.isna()).any()
