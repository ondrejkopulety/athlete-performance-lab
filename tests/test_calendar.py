"""
Kalendář musí končit dneškem a začínat první aktivitou.

Konec je dnešek, aby se neztratila ranní biometrie ve dnech bez tréninku.

Začátek byl chvíli nejstarší záznam z jakéhokoli zdroje, ale Apple Health
sahá do roku 2017, kdežto tréninková data začínají 2022 – vzniklo tak 1495
dní s nulovým TRIMP, nulovým CTL a readiness_score na pevných 75 bodech
(fixní bod vzorce při TSB = 0). Biometrie bez jediného tréninku se proto
zahazuje záměrně; podrobněji v docstringu src/analytics/calendar.py.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from sqlalchemy import func, select

from src.analytics.calendar import build_calendar, today_local
from src.db.models import Activity, DailyBiometrics, DailyMetrics


def test_calendar_ends_today_even_without_training(session):
    calendar = build_calendar(session)
    assert len(calendar) > 0
    assert calendar.max().date() == today_local()


def test_calendar_starts_at_first_activity(session):
    oldest_activity = session.scalar(select(func.min(Activity.date)))

    calendar = build_calendar(session)
    assert calendar.min().date() == oldest_activity


def test_biometrics_before_first_training_are_dropped(session):
    """
    Biometrie z doby před prvním tréninkem na osu nepatří – bez zátěže
    nemá připravenost k čemu být připravená a vyšla by z ní konstanta.
    """
    oldest_activity = session.scalar(select(func.min(Activity.date)))
    oldest_biometric = session.scalar(select(func.min(DailyBiometrics.date)))
    if oldest_biometric is None or oldest_biometric >= oldest_activity:
        return  # dataset takový den neobsahuje

    calendar = build_calendar(session)
    assert calendar.min().date() > oldest_biometric


def test_calendar_has_no_gaps(session):
    calendar = build_calendar(session)
    deltas = pd.Series(calendar).diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(days=1)]


def test_persisted_metrics_reach_today(session):
    """Poslední den v daily_metrics musí být dnešek i bez dnešního tréninku."""
    last_metrics = session.scalar(select(func.max(DailyMetrics.date)))
    last_activity = session.scalar(select(func.max(Activity.date)))
    assert last_metrics == today_local()
    if last_activity is not None:
        assert last_metrics >= last_activity


def test_biometrics_only_days_are_kept(session):
    """
    Dny, kdy byla naměřená biometrie, ale netrénovalo se, musí v metrikách
    existovat – jinak by se ranní HRV a spánek zahodily.

    Platí to uvnitř osy, tedy od prvního tréninku dál. Biometrii z doby před
    ním osa nepokrývá záměrně (viz test_biometrics_before_first_training…).
    """
    start = build_calendar(session).min().date()
    biometric_days = {
        d for d in session.scalars(select(DailyBiometrics.date)).all() if d >= start
    }
    activity_days = set(session.scalars(select(Activity.date)).all())
    rest_days_with_biometrics = biometric_days - activity_days
    if not rest_days_with_biometrics:
        return  # dataset takový den neobsahuje

    metric_days = set(session.scalars(select(DailyMetrics.date)).all())
    assert rest_days_with_biometrics <= metric_days
