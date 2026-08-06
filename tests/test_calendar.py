"""
Kalendář musí končit dneškem a začínat nejstarším záznamem z JAKÉHOKOLI
zdroje. Obojí je oprava proti původní implementaci, která začínala první
aktivitou (biometrie z dřívějška se ztrácela).
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


def test_calendar_starts_at_oldest_record_of_any_source(session):
    oldest_activity = session.scalar(select(func.min(Activity.date)))
    oldest_biometric = session.scalar(select(func.min(DailyBiometrics.date)))
    expected = min(d for d in (oldest_activity, oldest_biometric) if d is not None)

    calendar = build_calendar(session)
    assert calendar.min().date() == expected


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
    """
    biometric_days = set(session.scalars(select(DailyBiometrics.date)).all())
    activity_days = set(session.scalars(select(Activity.date)).all())
    rest_days_with_biometrics = biometric_days - activity_days
    if not rest_days_with_biometrics:
        return  # dataset takový den neobsahuje

    metric_days = set(session.scalars(select(DailyMetrics.date)).all())
    assert rest_days_with_biometrics <= metric_days
