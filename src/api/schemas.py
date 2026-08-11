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
    rhr_baseline_14d: float | None = None
    rhr_baseline_90d: float | None = None
    rhr_elevation_bpm: float | None = None
    rhr_source: str | None = None
    lthr_estimate: float | None = None
    avg_stress_day: float | None = None
    sleep_score_day: float | None = None
    sleep_duration_min: float | None = None
    sleep_need_min: float | None = None
    sleep_performance_pct: float | None = None
    max_hrr_60s_avg: float | None = None
    recovery_time_h: float | None = None
    garmin_readiness_score: float | None = None
    garmin_hrv_factor_pct: float | None = None
    stress_flag_count: int | None = None
    illness_warning: bool | None = None
    stress_flags: str | None = None
    coach_advice: str | None = None
    epoc_score_daily: float | None = None
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


# ── Dashboard ─────────────────────────────────────────────────────────────
# Krátká jména polí (d, km, asc, …) jsou schválně: aktivit jsou stovky a
# dlouhé klíče by v JSONu vážily víc než samotná data.


class DashboardToday(BaseModel):
    """Poslední den v kalendáři – to, co dashboard ukazuje nahoře."""
    model_config = ConfigDict(from_attributes=True)

    date: date
    readiness_score: float | None = None
    pure_recovery_score: float | None = None
    hrv_last_night: float | None = None
    hrv_weekly_avg: float | None = None
    hrv_cv_pct: float | None = None
    rhr_day: float | None = None
    rhr_baseline_14d: float | None = None
    rhr_elevation_bpm: float | None = None
    sleep_duration_min: float | None = None
    sleep_need_min: float | None = None
    sleep_performance_pct: float | None = None
    sleep_score_day: float | None = None
    avg_stress_day: float | None = None
    whoop_strain: float | None = None
    strain: float | None = None
    trimp: float | None = None
    ctl: float | None = None
    atl: float | None = None
    tsb: float | None = None
    acwr: float | None = None
    recovery_time_h: float | None = None
    garmin_readiness_score: float | None = None
    illness_warning: bool | None = None
    coach_advice: str | None = None


class DashboardBiometric(BaseModel):
    """Poslední naměřená hodnota a její reference (týdenní průměr, bazál,
    potřeba spánku) z téhož dne."""

    date: date
    value: float | None = None
    reference: float | None = None


class DashboardRide(BaseModel):
    id: str
    d: date
    dur: float | None = None      # minuty
    km: float | None = None
    avg: float | None = None      # tepová frekvence
    max: float | None = None
    asc: float | None = None      # převýšení v metrech
    trimp: float | None = None
    kcal: float | None = None
    z: list[float]                # minuty v zónách Z1–Z5


class DashboardActivity(BaseModel):
    """Odlehčený řádek pro grafy zón, stoupání a zotavovacího tepu."""

    id: str
    d: date
    z: list[float]
    up: float | None = None       # minuty do kopce
    asc: float | None = None
    grad: float | None = None     # průměrný sklon stoupání v %
    hrr: float | None = None      # max. pokles tepu za 60 s


class DashboardOut(BaseModel):
    generated_at: datetime
    today: DashboardToday | None = None
    last_known: dict[str, DashboardBiometric | None]
    # [datum, ctl, atl, tsb, trimp, strain, readiness, acwr]
    # Polarizace se nepošle: dashboard ji počítá z minut v zónách za zvolené
    # období, ne z klouzavé denní metriky – jinak by nesouhlasila s obdobím
    # zvoleným přepínačem.
    days: list[list[Any]]
    rides: list[DashboardRide]
    activities: list[DashboardActivity]


class SyncTriggerOut(BaseModel):
    accepted: bool
    message: str
