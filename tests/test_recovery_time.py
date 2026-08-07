"""
Garmin recovery time místo vymyšlené regenerační daně
======================================================

`recovery_tax_hours = min(96, 0.08 × TRIMP^1.2)` byla vymyšlená veličina:
Spearman s `total_trimp` 0.9955, tedy monotónní přeznačkování zátěže bez
jakékoli informace navíc. Proti Garminovu naměřenému recovery time vyšla
RMSE 32.0 h – k nerozeznání od nejlepší možné konstanty (32.1 h), zatímco
přeškálované ATL, které v projektu už je, dá 26.4 h.

Nahradil ji `recovery_time_h` z `training_readiness.csv`, což je hodnota,
kterou spočítal Firstbeat v hodinkách. Rok se stahovala, aniž by se kdy
naimportovala.

Tyhle testy hlídají hlavně jedno: že se nikde nic **nedopočítává**. Garmin
data začínají 7. 8. 2025 a pro 2784 starších dní musí zůstat NULL. Kdyby
je někdo v budoucnu backfilloval modelem, vypadalo by odhadnuté číslo
úplně stejně jako měřené – a to je přesně chyba, kvůli které se ta stará
metrika rušila.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.db import repository as repo

# První den, kdy Garmin Training Readiness vůbec něco vrátil.
GARMIN_ERA_START = pd.Timestamp("2025-08-07")

READINESS_COLUMNS = ("recovery_time_h", "garmin_readiness_score", "garmin_hrv_factor_pct")


# ── Převod jednotek ────────────────────────────────────────────────────────

@pytest.mark.parametrize("day,minutes", [("2025-09-18", 1443), ("2026-01-28", 2123)])
def test_minutes_are_converted_to_hours(tmp_path, monkeypatch, day, minutes):
    """Garmin posílá minuty, DB drží hodiny. Uloženo na jedno desetinné místo."""
    from src.ingestion import biometrics_import as bi

    (tmp_path / "training_readiness.csv").write_text(
        "date,score,recovery_time,sleep_score,hrv_factor_percent\n"
        "2025-09-18,61,1443,80,55\n"
        "2026-01-28,42,2123,75,40\n"
    )
    monkeypatch.setattr(bi, "SUMMARIES_DIR", tmp_path)

    frame = bi.build_biometrics_frame().set_index("date")
    stored = frame.loc[pd.Timestamp(day).date(), "recovery_time_h"]
    # Tolerance 0.06, ne 0.05: uložená hodnota je zaokrouhlená na desetiny,
    # takže odchylka smí dosáhnout přesně půl kroku (1443/60 = 24.05 → 24.0).
    assert stored == pytest.approx(minutes / 60.0, abs=0.06)


def test_sleep_score_is_not_taken_from_readiness(tmp_path, monkeypatch):
    """
    training_readiness.csv má vlastní sleep_score, ale ten už chodí ze
    sleep.csv. Dvě mapování na týž cíl by se v merge(how='outer') přetloukla
    a bylo by nedohledatelné, které z nich vyhrálo.
    """
    from src.ingestion import biometrics_import as bi

    (tmp_path / "sleep.csv").write_text(
        "date,sleep_score,duration_minutes\n2025-09-18,91,470\n"
    )
    (tmp_path / "training_readiness.csv").write_text(
        "date,score,recovery_time,sleep_score,hrv_factor_percent\n2025-09-18,61,1443,12,55\n"
    )
    monkeypatch.setattr(bi, "SUMMARIES_DIR", tmp_path)

    frame = bi.build_biometrics_frame().set_index("date")
    row = frame.loc[pd.Timestamp("2025-09-18").date()]
    assert row["sleep_score"] == 91, "sleep_score musí zůstat ze sleep.csv"
    assert row["garmin_readiness_score"] == 61


def test_missing_file_is_not_fatal(tmp_path, monkeypatch):
    """Kdo nemá Garmin Training Readiness, tomu pipeline nesmí spadnout."""
    from src.ingestion import biometrics_import as bi

    (tmp_path / "hrv.csv").write_text("date,last_night_avg\n2025-09-18,68\n")
    monkeypatch.setattr(bi, "SUMMARIES_DIR", tmp_path)

    frame = bi.build_biometrics_frame()
    assert not frame.empty
    assert "recovery_time_h" not in frame.columns


# ── Nad reálnými daty ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def session_data():
    from src.db.session import SessionLocal, check_connection

    if not check_connection():
        pytest.skip("Databáze neběží")
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(scope="module")
def daily(session_data):
    df = repo.read_daily_metrics(session_data)
    if df.empty:
        pytest.skip("Žádné denní metriky")
    return df.set_index(pd.to_datetime(df["date"]))


def test_pre_garmin_era_stays_null(daily):
    """
    Jádro věci: pro dobu před hodinkami se recovery time NEDOPOČÍTÁVÁ.
    NULL je poctivější než modelem dopočítané číslo, které by v exportu
    ani v odpovědi chatbota nešlo odlišit od měřeného.
    """
    for column in READINESS_COLUMNS:
        if column not in daily.columns:
            pytest.skip(f"{column} zatím nenaimportované")
        before = daily.loc[daily.index < GARMIN_ERA_START, column]
        assert before.isna().all(), (
            f"{column}: {int(before.notna().sum())} dní před {GARMIN_ERA_START.date()} "
            "má hodnotu – něco to dopočítává"
        )


def test_garmin_era_is_populated(daily):
    """A naopak: v garminské éře to nesmí být prázdné."""
    if "recovery_time_h" not in daily.columns:
        pytest.skip("recovery_time_h zatím nenaimportované")
    era = daily.loc[daily.index >= GARMIN_ERA_START, "recovery_time_h"]
    if era.empty:
        pytest.skip("Databáze nesahá do garminské éry")
    assert era.notna().sum() >= 300, f"Jen {int(era.notna().sum())} dní z ~365"


def test_values_are_within_garmin_range(daily):
    """Firstbeat strop je 96 h (5760 min); nic mimo 0–96 nedává smysl."""
    if "recovery_time_h" not in daily.columns:
        pytest.skip("recovery_time_h zatím nenaimportované")
    values = daily["recovery_time_h"].dropna()
    if values.empty:
        pytest.skip("Žádné hodnoty")
    assert values.min() >= 0.0
    assert values.max() <= 96.0

    score = daily.get("garmin_readiness_score", pd.Series(dtype=float)).dropna()
    if not score.empty:
        assert score.between(0, 100).all()


def test_invented_metric_is_gone(session_data, daily):
    """Vymyšlený sloupec nesmí zůstat viset ani v jedné tabulce."""
    assert "recovery_tax_hours_daily" not in daily.columns

    activities = repo.read_activities(session_data, with_metrics=True)
    if activities.empty:
        pytest.skip("Žádné aktivity")
    assert "recovery_tax_hours" not in activities.columns
