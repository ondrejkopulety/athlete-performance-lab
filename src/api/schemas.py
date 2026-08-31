"""Pydantic modely odpovědí. Volitelnost všude záměrná – chybějící
měření se posílá jako null, nikdy jako 0."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Alias, aby šlo pole pojmenovat "date" a zároveň ho typovat datem: bez něj
# se v anotaci "date | None" rozsvítí samotné pole (= None), ne typ.
DateT = date


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
    max_speed_kmh: float | None = None
    uphill_minutes: float | None = None
    downhill_minutes: float | None = None
    flat_minutes: float | None = None
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
    # Percentilové pořadí TRIMP vůči vlastní historii kardio aktivit –
    # podklad pro verdikt/gauge, viz activity.py:compute_trimp_load_percentile.
    trimp_load_percentile: float | None = None
    # Whoop strain je denní (daily_metrics), ne per-aktivita – posílá se
    # sem s počtem aktivit toho dne, aby UI vědělo, kdy hodnota nepatří
    # jen téhle jízdě.
    whoop_strain: float | None = None
    activities_same_day: int | None = None
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
    # Souřadnice pro mapu trasy v detailu aktivity (SVG polyline, bez
    # podkladové mapy – viz rozhodnutí v CLAUDE.md/konverzaci o mapě).
    position_lat: float | None = None
    position_long: float | None = None
    hr_zone: str | None = None


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


class RideCoverage(BaseModel):
    """
    Na jak úplných datech čísla jízdy stojí.

    Ven jde hlavně ``ok`` a ``note``: uživatel potřebuje odpověď na "můžu
    tomuhle číslu věřit", ne tři procenta k interpretaci. ``density`` je
    informativní a **nikdy** nespouští varování – řídký zápis Smart
    Recordingu není ztráta dat.
    """

    ok: bool
    pct: float | None = None      # pokrytí po ffillu; rozhoduje o platnosti
    density: float | None = None  # hustota vzorků; informativní
    gap: int | None = None        # nejdelší souvislá díra v sekundách
    note: str | None = None       # vysvětlení do detailu; None, když je vše v pořádku


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
    cov: RideCoverage | None = None


class DashboardActivity(BaseModel):
    """
    Odlehčený řádek pro grafy zón, stoupání, zotavovacího tepu a stránku
    Stats (souhrny za období, terénní a nutriční rozklad) – jediný seznam
    aktivit v dashboardu, který nemá historický limit.
    """

    id: str
    d: date
    z: list[float]
    up: float | None = None       # minuty do kopce
    down: float | None = None     # minuty z kopce
    flat: float | None = None     # minuty po rovině
    asc: float | None = None
    grad: float | None = None     # průměrný sklon stoupání v %
    hrr: float | None = None      # max. pokles tepu za 60 s
    km: float | None = None
    dur: float | None = None      # minuty
    kcal: float | None = None
    fat: float | None = None      # g tuků spálených při aktivitě
    carb: float | None = None     # g cukrů spálených při aktivitě
    # Nezáměrná Z3 v sekundách: čas v Z3 mimo souvislé bloky. None = bloky
    # pro tuhle aktivitu ještě spočítané nejsou, což není nula.
    z3u: int | None = None
    cov: RideCoverage | None = None


# ── Panely tepové křivky a souvislých bloků ───────────────────────────────


class CurvePoint(BaseModel):
    """
    Jeden bod křivky. ``hr = None`` znamená, že v období žádná jízda takhle
    dlouhé okno nemá – čára v grafu tam končí, v tabulce je pomlčka. Nikdy
    nula.
    """

    d: int                        # délka okna v sekundách
    hr: float | None = None
    activity_id: str | None = None
    date: DateT | None = None
    label: str | None = None


class LastMaxEffort(BaseModel):
    """Poslední jízda, která přispěla do referenčního bodu křivky."""

    date: DateT
    activity_id: str | None = None
    label: str | None = None
    duration_s: int
    days_ago: int
    stale: bool                   # přes hranici → zvýraznit


class CurvePeriod(BaseModel):
    label: str
    from_date: date | None = None
    to_date: date | None = None
    rides_total: int
    rides_excluded: int
    points: list[CurvePoint]
    last_max_effort: LastMaxEffort | None = None


class HrCurveOut(BaseModel):
    durations_s: list[int]
    complete_only: bool
    coverage_warn_pct: float
    period: CurvePeriod
    reference: CurvePeriod | None = None


class SegmentBucket(BaseModel):
    """Koš histogramu délek úseků – počet i součet času, obojí je důležité."""

    bucket: str
    count: int
    seconds: int


class BlockTotals(BaseModel):
    segment_count: int
    total_time_s: int
    time_in_long_blocks_s: int


class BlockSource(BaseModel):
    activity_id: str | None = None
    date: DateT | None = None
    label: str | None = None


class HrBlocksOut(BaseModel):
    requested_bpm: int            # hranice zóny spočítaná z LTHR
    threshold_bpm: int            # nejbližší práh z uložené mřížky
    zone: str
    lthr_bpm: int
    tolerance_s: int
    complete_only: bool
    coverage_warn_pct: float
    rides_total: int
    rides_excluded: int
    # None = v období není ani jedna jízda. Nula by znamenala "jel jsem, ale
    # nad práh se nedostal", a to je jiné tvrzení.
    longest_block_s: int | None = None
    longest_block: BlockSource | None = None
    previous_longest_block_s: int | None = None
    trend_pct: float | None = None
    hist: list[SegmentBucket]
    totals: BlockTotals


class ThresholdOut(BaseModel):
    """Nastavený práh a jeho stáří – číslo, na kterém visí zbytek dashboardu."""

    lthr_bpm: int
    hr_max_bpm: int
    valid_from: date
    note: str | None = None
    days_ago: int
    stale: bool
    stale_after_days: int
    source: str                   # "user" | "settings"
    zone_thresholds: dict[str, int]   # zóna → nejbližší práh z mřížky


class ThresholdIn(BaseModel):
    lthr_bpm: int = Field(ge=100, le=220)
    hr_max_bpm: int = Field(ge=120, le=230)
    valid_from: date | None = None
    note: str | None = None


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
