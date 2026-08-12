"""
SQLAlchemy 2.0 ORM modely
=========================

Návrhový princip: **surová fakta jsou oddělená od odvozených metrik.**

  activities        – co přišlo z FIT souboru; nikdy se nepřepočítává
  activity_metrics  – co jsme spočítali; nese metrics_version → přepočitatelné
  activity_hr_curve – tepová křivka (max. průměr za okno); nese calc_version
  activity_hr_blocks– souvislé bloky nad prahem; nese calc_version
  activity_hr_coverage – na jak úplných datech obojí stojí; nese calc_version
  athlete_threshold – historie nastavení LTHR a maximálního tepu
  records           – vteřinová data (TimescaleDB hypertable)
  daily_biometrics  – denní vstupy z Garmin Connect API (HRV, spánek, RHR, stres)
  daily_metrics     – denní výstup analytiky (dřívější athlete_readiness.csv)
  sync_state        – kde skončil poslední sync / analýza (klíč → JSONB)

Díky tomuhle rozdělení stačí při změně vzorce bumpnout verzi v
config.settings a pipeline sama přepočítá jen dotčené řádky.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITIES – surová fakta z FIT
# ═══════════════════════════════════════════════════════════════════════════
class Activity(Base):
    __tablename__ = "activities"

    activity_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Identita a čas
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    activity_name: Mapped[str | None] = mapped_column(Text)
    sport: Mapped[str | None] = mapped_column(String(64), index=True)

    # Zátěž a tep
    duration_minutes: Mapped[float | None] = mapped_column(Float)
    total_trimp: Mapped[float | None] = mapped_column(Float)
    avg_hr: Mapped[float | None] = mapped_column(Float)
    max_hr: Mapped[float | None] = mapped_column(Float)
    time_in_z1: Mapped[float | None] = mapped_column(Float)
    time_in_z2: Mapped[float | None] = mapped_column(Float)
    time_in_z3: Mapped[float | None] = mapped_column(Float)
    time_in_z4: Mapped[float | None] = mapped_column(Float)
    time_in_z5: Mapped[float | None] = mapped_column(Float)
    zone2_cap_used: Mapped[float | None] = mapped_column(Float)
    records_count: Mapped[int | None] = mapped_column(Integer)

    # Fyzické metriky
    distance_km: Mapped[float | None] = mapped_column(Float)
    ascent_m: Mapped[float | None] = mapped_column(Float)
    descent_m: Mapped[float | None] = mapped_column(Float)
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float)
    max_speed_kmh: Mapped[float | None] = mapped_column(Float)
    calories: Mapped[float | None] = mapped_column(Float)
    uphill_minutes: Mapped[float | None] = mapped_column(Float)

    # Senzory
    avg_cadence: Mapped[float | None] = mapped_column(Float)
    max_cadence: Mapped[float | None] = mapped_column(Float)
    avg_temp: Mapped[float | None] = mapped_column(Float)
    max_temp: Mapped[float | None] = mapped_column(Float)

    # Garmin výkonnostní metriky
    training_effect_aerobic: Mapped[float | None] = mapped_column(Float)
    training_effect_anaerobic: Mapped[float | None] = mapped_column(Float)
    vo2_max: Mapped[float | None] = mapped_column(Float)

    # Provenience
    source: Mapped[str | None] = mapped_column(String(16), index=True)  # garmin | strava
    fit_path: Mapped[str | None] = mapped_column(Text)
    fit_sha256: Mapped[str | None] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    metrics: Mapped["ActivityMetrics | None"] = relationship(
        back_populates="activity", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (Index("ix_activities_date_sport", "date", "sport"),)


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITY_METRICS – odvozené, přepočitatelné
# ═══════════════════════════════════════════════════════════════════════════
class ActivityMetrics(Base):
    __tablename__ = "activity_metrics"

    activity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("activities.activity_id", ondelete="CASCADE"), primary_key=True
    )
    metrics_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # A/B – drift a zotavení tepu
    cardiac_drift: Mapped[float | None] = mapped_column(Float)
    max_hrr_60s: Mapped[float | None] = mapped_column(Float)

    # P – durabilita
    durability_pct: Mapped[float | None] = mapped_column(Float)

    # J/T – stoupání
    vam_m_per_h: Mapped[float | None] = mapped_column(Float)
    avg_gradient_pct: Mapped[float | None] = mapped_column(Float)
    climb_category: Mapped[str | None] = mapped_column(String(16))

    # S/U/V – R-R fyziologie
    # aet_hr_dfa/ant_hr_dfa smí obsahovat jen práh spočítaný z R-R intervalů.
    # Odhad z linearity tep↔rychlost patří výhradně do aet_hr_proxy – dokud
    # se plnily oba stejnou hodnotou, vypadal proxy odhad jako výsledek DFA.
    aet_hr_dfa: Mapped[int | None] = mapped_column(Integer)
    ant_hr_dfa: Mapped[int | None] = mapped_column(Integer)
    aet_hr_proxy: Mapped[int | None] = mapped_column(Integer)
    # rr_ok | unreliable | synthetic_rr | no_rr | failed  (viz physio/cli.py)
    dfa_quality: Mapped[str | None] = mapped_column(String(16))
    resp_rate_rsa: Mapped[float | None] = mapped_column(Float)
    # R-R intervaly (ms) uložené při načtení FIT → DFA/RSA se počítají bez
    # opětovného otevírání FIT souboru na disku.
    rr_intervals_ms: Mapped[list[float] | None] = mapped_column(ARRAY(Float))

    # TRIMP přepočítaný z klidového tepu platného k datu aktivity.
    # activities.total_trimp zůstává tím, co spočítal parser s pevnou
    # konstantou; tohle je verzovaná odvozená hodnota.
    trimp_adjusted: Mapped[float | None] = mapped_column(Float)
    rhr_used: Mapped[float | None] = mapped_column(Float)

    # Nejlepší klouzavé průměry tepu – vstup pro odhad prahu (LTHR)
    best_20min_hr: Mapped[float | None] = mapped_column(Float)
    best_30min_hr: Mapped[float | None] = mapped_column(Float)
    best_60min_hr: Mapped[float | None] = mapped_column(Float)

    # Diagnostika DFA – ať je z dat vidět, kde metoda dává smysl
    dfa_alpha1_min: Mapped[float | None] = mapped_column(Float)
    dfa_alpha1_median: Mapped[float | None] = mapped_column(Float)
    dfa_window_count: Mapped[int | None] = mapped_column(Integer)

    # Diagnostika R-R (src/physio) – čím se posuzuje, jestli řada vůbec nese
    # variabilitu mezi tepy, a ne jen jestli ve FIT byly hrv zprávy.
    rr_beat_count: Mapped[int | None] = mapped_column(Integer)
    rr_artifact_pct: Mapped[float | None] = mapped_column(Float)
    rr_zero_diff_pct: Mapped[float | None] = mapped_column(Float)
    rr_unique_values: Mapped[int | None] = mapped_column(Integer)
    rr_lattice_coverage: Mapped[float | None] = mapped_column(Float)
    rr_authenticity: Mapped[str | None] = mapped_column(String(16))

    # R/Q/W – EPOC, práh, TATI
    epoc_score: Mapped[float | None] = mapped_column(Float)
    time_at_threshold_min: Mapped[float | None] = mapped_column(Float)
    tte_z4z5_min: Mapped[float | None] = mapped_column(Float)
    critical_hr: Mapped[float | None] = mapped_column(Float)
    tati_score: Mapped[float | None] = mapped_column(Float)

    # M/N/O – metabolismus, tekutiny, teplo
    fat_kcal: Mapped[float | None] = mapped_column(Float)
    carb_kcal: Mapped[float | None] = mapped_column(Float)
    fat_g: Mapped[float | None] = mapped_column(Float)
    carb_g: Mapped[float | None] = mapped_column(Float)
    fluid_loss_l: Mapped[float | None] = mapped_column(Float)
    heat_flag: Mapped[bool | None] = mapped_column(Boolean)

    activity: Mapped[Activity] = relationship(back_populates="metrics")


# ═══════════════════════════════════════════════════════════════════════════
# RECORDS – vteřinová data (TimescaleDB hypertable)
# ═══════════════════════════════════════════════════════════════════════════
class Record(Base):
    __tablename__ = "records"

    # PK musí obsahovat partitioning column (timestamp) – požadavek TimescaleDB.
    activity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), primary_key=True)

    heart_rate: Mapped[float | None] = mapped_column(Float)
    speed: Mapped[float | None] = mapped_column(Float)
    power: Mapped[float | None] = mapped_column(Float)
    cadence: Mapped[float | None] = mapped_column(Float)
    altitude: Mapped[float | None] = mapped_column(Float)
    distance: Mapped[float | None] = mapped_column(Float)
    temperature: Mapped[float | None] = mapped_column(Float)
    vertical_oscillation: Mapped[float | None] = mapped_column(Float)
    stance_time: Mapped[float | None] = mapped_column(Float)
    respiratory_rate: Mapped[float | None] = mapped_column(Float)
    hrv: Mapped[float | None] = mapped_column(Float)
    position_lat: Mapped[float | None] = mapped_column(Float)
    position_long: Mapped[float | None] = mapped_column(Float)
    hr_zone: Mapped[str | None] = mapped_column(String(4))
    is_active: Mapped[bool | None] = mapped_column(Boolean)
    trimp_increment: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_records_activity_ts", "activity_id", "timestamp"),)


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITY_HR_CURVE / ACTIVITY_HR_BLOCKS – předvýpočet pro dashboard
# ═══════════════════════════════════════════════════════════════════════════
# Obojí se z vteřinových dat nedá počítat za běhu, obojí je per-aktivita.
# Na rozdíl od activity_metrics je klíč složený – jedna aktivita má víc
# řádků (jeden na délku okna, resp. na kombinaci práh × tolerance), takže
# to nejsou sloupce v activity_metrics.
#
# Klíčové rozhodnutí: prahy v bpm, ne zóny. Zóny se odvozují z LTHR, které
# se mění; uložené zóny by při každé změně znamenaly přepočet historie.
class ActivityHrCurve(Base):
    """
    Tepová křivka – maximální průměrný tep za dané okno.

    Obdoba výkonové křivky, jen z tepu. Na LTHR nezávislá úplně: spočítá se
    jednou a platí, dokud se nezmění samotná data nebo calc_version.
    """

    __tablename__ = "activity_hr_curve"

    activity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("activities.activity_id", ondelete="CASCADE"), primary_key=True
    )
    duration_s: Mapped[int] = mapped_column(Integer, primary_key=True)

    max_mean_hr: Mapped[float] = mapped_column(Numeric(4, 1), nullable=False)

    calc_version: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ActivityHrBlocks(Base):
    """
    Souvislé bloky nad prahem – jak dlouho se nad daným tepem vydrží v kuse.

    ``bridge_tolerance_s`` je součástí klíče schválně: ukládají se obě
    varianty vedle sebe. Nula ukazuje surovou fragmentaci, 15 s realistickou
    souvislost úsilí, a rozdíl mezi nimi je sám o sobě informace.
    """

    __tablename__ = "activity_hr_blocks"

    activity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("activities.activity_id", ondelete="CASCADE"), primary_key=True
    )
    threshold_bpm: Mapped[int] = mapped_column(Integer, primary_key=True)
    bridge_tolerance_s: Mapped[int] = mapped_column(Integer, primary_key=True)

    longest_block_s: Mapped[int] = mapped_column(Integer, nullable=False)
    total_time_s: Mapped[int] = mapped_column(Integer, nullable=False)
    # Součet úseků delších než HR_BLOCK_LONG_S – "kolik z toho času byla
    # souvislá práce", ne posbírané vteřiny.
    time_in_long_blocks_s: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    median_segment_s: Mapped[float | None] = mapped_column(Numeric(6, 1))

    # Rozdělení délek úseků po koších (settings.HR_SEGMENT_BUCKETS_S).
    # Počty i součty sekund, protože každé říká něco jiného: 199 úseků pod
    # 30 s vypadá jinak než 18 minut, které dohromady dají.
    segment_hist_counts: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, server_default="{}"
    )
    segment_hist_seconds: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, server_default="{}"
    )

    calc_version: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ActivityHrCoverage(Base):
    """
    Na jak úplných datech stojí křivka a bloky téhle jízdy.

    Ukládají se sekundy, ne procenta: procenta jsou podíl dvou uložených
    čísel, kdežto z procent se rozsah zpátky nedostane – a rozdíl mezi
    "80 % z hodiny" a "80 % ze čtyř hodin" je pro důvěru v číslo podstatný.

    Dvě různá pokrytí vedle sebe schválně (viz hr_batch.ActivityCoverage):
    ``measured_s`` je hustota zápisu, ``usable_s`` je použitelnost mřížky po
    doplnění mezer. Varování visí výhradně na druhém – řídký zápis Smart
    Recordingu není ztráta dat.
    """

    __tablename__ = "activity_hr_coverage"

    activity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("activities.activity_id", ondelete="CASCADE"), primary_key=True
    )

    span_s: Mapped[int] = mapped_column(Integer, nullable=False)
    measured_s: Mapped[int] = mapped_column(Integer, nullable=False)
    usable_s: Mapped[int] = mapped_column(Integer, nullable=False)
    longest_gap_s: Mapped[int] = mapped_column(Integer, nullable=False)
    # Nejdelší okno, které jízdě v tepové křivce vyšlo; None = žádné.
    max_curve_duration_s: Mapped[int | None] = mapped_column(Integer)

    calc_version: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AthleteThreshold(Base):
    """
    Historie nastavení prahového a maximálního tepu.

    Historie, ne jeden přepisovaný řádek: dashboard ukazuje "nastaveno před
    N dny" a to N musí být z něčeho měřitelného. Zóny se odsud neodvozují –
    ZONES v settings.py jsou měřené z laktátového testu a mají přednost.
    LTHR řídí jen lookup prahu v panelu souvislých bloků, takže jeho změna
    nikdy nespustí přepočet uložených dat.
    """

    __tablename__ = "athlete_threshold"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    lthr_bpm: Mapped[int] = mapped_column(Integer, nullable=False)
    hr_max_bpm: Mapped[int] = mapped_column(Integer, nullable=False)
    # Odkdy hodnota platí – typicky datum testu, ne datum zápisu.
    valid_from: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    note: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ═══════════════════════════════════════════════════════════════════════════
# DAILY_BIOMETRICS – vstupy z Garmin Connect
# ═══════════════════════════════════════════════════════════════════════════
class DailyBiometrics(Base):
    """
    Denní biometrie. Složený klíč (date, source) záměrně – Apple Watch a
    Garmin měří klidový tep měřitelně jinak (překryv 25 dní, korelace 0.08,
    Apple čte o 3–6 bpm výš), takže se nesmí slévat do jedné hodnoty.
    Obě měření zůstávají uložená a čtenář si vybírá prioritou.
    """

    __tablename__ = "daily_biometrics"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(16), primary_key=True, default="garmin")

    hrv_last_night: Mapped[float | None] = mapped_column(Float)
    hrv_weekly_avg: Mapped[float | None] = mapped_column(Float)
    resting_heart_rate: Mapped[float | None] = mapped_column(Float)

    sleep_score: Mapped[float | None] = mapped_column(Float)
    sleep_duration_min: Mapped[float | None] = mapped_column(Float)
    sleep_deep_min: Mapped[float | None] = mapped_column(Float)
    sleep_light_min: Mapped[float | None] = mapped_column(Float)
    sleep_rem_min: Mapped[float | None] = mapped_column(Float)
    sleep_awake_min: Mapped[float | None] = mapped_column(Float)

    stress_average: Mapped[float | None] = mapped_column(Float)
    body_battery_max: Mapped[float | None] = mapped_column(Float)
    body_battery_min: Mapped[float | None] = mapped_column(Float)

    vo2_max: Mapped[float | None] = mapped_column(Float)
    steps: Mapped[int | None] = mapped_column(BigInteger)
    intensity_minutes: Mapped[float | None] = mapped_column(Float)

    # Training Readiness – Garminovy vlastní odhady (Firstbeat).
    # Apple je neposkytuje, takže u source='apple' zůstávají NULL.
    recovery_time_h: Mapped[float | None] = mapped_column(Float)
    garmin_readiness_score: Mapped[float | None] = mapped_column(Float)
    garmin_hrv_factor_pct: Mapped[float | None] = mapped_column(Float)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ═══════════════════════════════════════════════════════════════════════════
# DAILY_METRICS – výstup analytiky
# ═══════════════════════════════════════════════════════════════════════════
class DailyMetrics(Base):
    __tablename__ = "daily_metrics"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    metrics_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Zátěž (PMC)
    trimp: Mapped[float | None] = mapped_column(Float)
    trimp_epoc: Mapped[float | None] = mapped_column(Float)
    ctl: Mapped[float | None] = mapped_column(Float)
    atl: Mapped[float | None] = mapped_column(Float)
    tsb: Mapped[float | None] = mapped_column(Float)

    # Riziko
    acwr: Mapped[float | None] = mapped_column(Float)
    ctl_ramp_rate: Mapped[float | None] = mapped_column(Float)
    ctl_ramp_warning: Mapped[bool | None] = mapped_column(Boolean)

    # Kvalita tréninku
    monotony: Mapped[float | None] = mapped_column(Float)
    strain: Mapped[float | None] = mapped_column(Float)
    whoop_strain: Mapped[float | None] = mapped_column(Float)
    daily_efficiency: Mapped[float | None] = mapped_column(Float)
    ef_trend: Mapped[float | None] = mapped_column(Float)
    fatigue_index: Mapped[float | None] = mapped_column(Float)
    polarization_low_pct: Mapped[float | None] = mapped_column(Float)
    polarization_high_pct: Mapped[float | None] = mapped_column(Float)
    z3_junk_pct: Mapped[float | None] = mapped_column(Float)
    polarization_efficiency: Mapped[float | None] = mapped_column(Float)

    # Regenerace a biometrie (denormalizované pro rychlé čtení dashboardem)
    readiness_score: Mapped[float | None] = mapped_column(Float)
    pure_recovery_score: Mapped[float | None] = mapped_column(Float)
    hrv_last_night: Mapped[float | None] = mapped_column(Float)
    hrv_weekly_avg: Mapped[float | None] = mapped_column(Float)
    hrv_cv_pct: Mapped[float | None] = mapped_column(Float)
    rhr_day: Mapped[float | None] = mapped_column(Float)
    rhr_baseline_14d: Mapped[float | None] = mapped_column(Float)
    rhr_baseline_90d: Mapped[float | None] = mapped_column(Float)
    rhr_source: Mapped[str | None] = mapped_column(String(16))
    lthr_estimate: Mapped[float | None] = mapped_column(Float)
    rhr_elevation_bpm: Mapped[float | None] = mapped_column(Float)
    avg_stress_day: Mapped[float | None] = mapped_column(Float)
    sleep_score_day: Mapped[float | None] = mapped_column(Float)
    sleep_duration_min: Mapped[float | None] = mapped_column(Float)
    sleep_need_min: Mapped[float | None] = mapped_column(Float)
    sleep_performance_pct: Mapped[float | None] = mapped_column(Float)
    max_hrr_60s_avg: Mapped[float | None] = mapped_column(Float)

    # Garminovy vlastní odhady (Firstbeat) – měřené, ne dopočítané.
    # Dostupné až od 8/2025, pro starší dny zůstávají NULL.
    recovery_time_h: Mapped[float | None] = mapped_column(Float)
    garmin_readiness_score: Mapped[float | None] = mapped_column(Float)
    garmin_hrv_factor_pct: Mapped[float | None] = mapped_column(Float)

    # Varování
    stress_flag_count: Mapped[int | None] = mapped_column(Integer)
    illness_warning: Mapped[bool | None] = mapped_column(Boolean)
    stress_flags: Mapped[str | None] = mapped_column(Text)
    coach_advice: Mapped[str | None] = mapped_column(Text)

    # Denní součty z per-activity metrik
    epoc_score_daily: Mapped[float | None] = mapped_column(Float)
    fat_kcal_daily: Mapped[float | None] = mapped_column(Float)
    carb_kcal_daily: Mapped[float | None] = mapped_column(Float)
    fat_g_daily: Mapped[float | None] = mapped_column(Float)
    carb_g_daily: Mapped[float | None] = mapped_column(Float)
    fluid_loss_l_daily: Mapped[float | None] = mapped_column(Float)


# ═══════════════════════════════════════════════════════════════════════════
# SYNC_STATE – kde skončil poslední běh
# ═══════════════════════════════════════════════════════════════════════════
class SyncState(Base):
    __tablename__ = "sync_state"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
