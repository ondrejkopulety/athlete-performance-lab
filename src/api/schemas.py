"""Pydantic modely odpovědí. Volitelnost všude záměrná – chybějící
měření se posílá jako null, nikdy jako 0."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    database: bool
    activities: int
    days: int
    last_activity_date: date | None = None
    last_metrics_date: date | None = None


class DailyMetricsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    trimp: float | None = None
    trimp_epoc: float | None = None
    ctl: float | None = None
    atl: float | None = None
    tsb: float | None = None
    acwr: float | None = None
    ctl_ramp_rate: float | None = None
    ctl_ramp_warning: bool | None = None
    monotony: float | None = None
    strain: float | None = None
    whoop_strain: float | None = None
    daily_efficiency: float | None = None
    ef_trend: float | None = None
    fatigue_index: float | None = None
    polarization_low_pct: float | None = None
    polarization_high_pct: float | None = None
    z3_junk_pct: float | None = None
    polarization_efficiency: float | None = None
    readiness_score: float | None = None
    pure_recovery_score: float | None = None
    hrv_last_night: float | None = None
    hrv_weekly_avg: float | None = None
    hrv_cv_pct: float | None = None
    rhr_day: float | None = None
    avg_stress_day: float | None = None
    sleep_score_day: float | None = None
    sleep_duration_min: float | None = None
    sleep_need_min: float | None = None
    sleep_performance_pct: float | None = None
    max_hrr_60s_avg: float | None = None
    stress_flag_count: int | None = None
    illness_warning: bool | None = None
    stress_flags: str | None = None
    coach_advice: str | None = None
    epoc_score_daily: float | None = None
    recovery_tax_hours_daily: float | None = None
    fat_kcal_daily: float | None = None
    carb_kcal_daily: float | None = None
    fat_g_daily: float | None = None
    carb_g_daily: float | None = None
    fluid_loss_l_daily: float | None = None


class PmcPoint(BaseModel):
    """Odlehčený tvar pro PMC graf – frontend nepotřebuje 40 sloupců."""
    model_config = ConfigDict(from_attributes=True)

    date: date
    trimp: float | None = None
    ctl: float | None = None
    atl: float | None = None
    tsb: float | None = None
    acwr: float | None = None


class ActivityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    activity_id: str
    date: date
    start_time: datetime | None = None
    activity_name: str | None = None
    sport: str | None = None
    duration_minutes: float | None = None
    total_trimp: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    distance_km: float | None = None
    ascent_m: float | None = None
    descent_m: float | None = None
    avg_speed_kmh: float | None = None
    calories: float | None = None
    time_in_z1: float | None = None
    time_in_z2: float | None = None
    time_in_z3: float | None = None
    time_in_z4: float | None = None
    time_in_z5: float | None = None
    source: str | None = None


class ActivityDetailOut(ActivityOut):
    cardiac_drift: float | None = None
    max_hrr_60s: float | None = None
    durability_pct: float | None = None
    vam_m_per_h: float | None = None
    avg_gradient_pct: float | None = None
    climb_category: str | None = None
    aet_hr_dfa: int | None = None
    ant_hr_dfa: int | None = None
    dfa_quality: str | None = None
    resp_rate_rsa: float | None = None
    epoc_score: float | None = None
    recovery_tax_hours: float | None = None
    time_at_threshold_min: float | None = None
    critical_hr: float | None = None
    tati_score: float | None = None
    fat_g: float | None = None
    carb_g: float | None = None
    fluid_loss_l: float | None = None
    heat_flag: bool | None = None
    metrics_version: int | None = None


class RecordPoint(BaseModel):
    ts: datetime
    heart_rate: float | None = None
    speed: float | None = None
    power: float | None = None
    cadence: float | None = None
    altitude: float | None = None
    distance: float | None = None
    temperature: float | None = None


class SyncStatusOut(BaseModel):
    running: bool
    last_run: dict[str, Any] | None = None


class SyncTriggerOut(BaseModel):
    accepted: bool
    message: str
