"""
Pojistka proti tichému rozjetí seznamů sloupců v analytics/pipeline.py vůči
skutečnému schématu v db/models.py.

Ty seznamy (ACTIVITY_METRIC_COLUMNS, DAILY_METRIC_COLUMNS, …) se udržují
ručně a řídí, co se do DB zapisuje. Když v modelu sloupec přibude/zmizí a
seznam se nedopraví, chyba se projeví až za běhu (nebo vůbec – hodnota se
tiše zahodí). Nepotřebuje DB, jen import.
"""

from __future__ import annotations

from src.analytics import pipeline
from src.db.models import ActivityMetrics, DailyMetrics


def _model_columns(model) -> set[str]:
    return {c.name for c in model.__table__.columns}


def test_activity_metric_columns_exist_in_model():
    cols = _model_columns(ActivityMetrics)
    unknown = [c for c in pipeline.ACTIVITY_METRIC_COLUMNS if c not in cols]
    assert not unknown, f"ACTIVITY_METRIC_COLUMNS mimo model: {unknown}"


def test_series_only_columns_are_subset_of_activity_metrics():
    unknown = [c for c in pipeline.SERIES_ONLY_COLUMNS if c not in pipeline.ACTIVITY_METRIC_COLUMNS]
    assert not unknown, f"SERIES_ONLY_COLUMNS nejsou v ACTIVITY_METRIC_COLUMNS: {unknown}"


def test_global_metric_columns_exist_in_model():
    cols = _model_columns(ActivityMetrics)
    unknown = [c for c in pipeline.GLOBAL_METRIC_COLUMNS if c not in cols]
    assert not unknown, f"GLOBAL_METRIC_COLUMNS mimo model: {unknown}"


def test_daily_metric_columns_exist_in_model():
    cols = _model_columns(DailyMetrics)
    unknown = [c for c in pipeline.DAILY_METRIC_COLUMNS if c not in cols]
    assert not unknown, f"DAILY_METRIC_COLUMNS mimo model: {unknown}"


def test_round_map_keys_are_known_daily_columns():
    unknown = [c for c in pipeline.ROUND_MAP if c not in pipeline.DAILY_METRIC_COLUMNS]
    assert not unknown, f"ROUND_MAP klíče mimo DAILY_METRIC_COLUMNS: {unknown}"
