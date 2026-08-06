"""Initial schema: activities, activity_metrics, records (hypertable),
daily_biometrics, daily_metrics, sync_state

Revision ID: 0001
Revises:
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    # ── activities – surová fakta z FIT ────────────────────────────────────
    op.create_table(
        "activities",
        sa.Column("activity_id", sa.String(64), primary_key=True),
        sa.Column("start_time", sa.DateTime(timezone=False)),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("activity_name", sa.Text()),
        sa.Column("sport", sa.String(64)),
        sa.Column("duration_minutes", sa.Float()),
        sa.Column("total_trimp", sa.Float()),
        sa.Column("avg_hr", sa.Float()),
        sa.Column("max_hr", sa.Float()),
        sa.Column("time_in_z1", sa.Float()),
        sa.Column("time_in_z2", sa.Float()),
        sa.Column("time_in_z3", sa.Float()),
        sa.Column("time_in_z4", sa.Float()),
        sa.Column("time_in_z5", sa.Float()),
        sa.Column("zone2_cap_used", sa.Float()),
        sa.Column("records_count", sa.Integer()),
        sa.Column("distance_km", sa.Float()),
        sa.Column("ascent_m", sa.Float()),
        sa.Column("descent_m", sa.Float()),
        sa.Column("avg_speed_kmh", sa.Float()),
        sa.Column("max_speed_kmh", sa.Float()),
        sa.Column("calories", sa.Float()),
        sa.Column("uphill_minutes", sa.Float()),
        sa.Column("avg_cadence", sa.Float()),
        sa.Column("max_cadence", sa.Float()),
        sa.Column("avg_temp", sa.Float()),
        sa.Column("max_temp", sa.Float()),
        sa.Column("training_effect_aerobic", sa.Float()),
        sa.Column("training_effect_anaerobic", sa.Float()),
        sa.Column("vo2_max", sa.Float()),
        sa.Column("source", sa.String(16)),
        sa.Column("fit_path", sa.Text()),
        sa.Column("fit_sha256", sa.String(64)),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_activities_date", "activities", ["date"])
    op.create_index("ix_activities_sport", "activities", ["sport"])
    op.create_index("ix_activities_source", "activities", ["source"])
    op.create_index("ix_activities_date_sport", "activities", ["date", "sport"])

    # ── activity_metrics – odvozené, verzované ─────────────────────────────
    op.create_table(
        "activity_metrics",
        sa.Column(
            "activity_id",
            sa.String(64),
            sa.ForeignKey("activities.activity_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("metrics_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("cardiac_drift", sa.Float()),
        sa.Column("max_hrr_60s", sa.Float()),
        sa.Column("durability_pct", sa.Float()),
        sa.Column("vam_m_per_h", sa.Float()),
        sa.Column("avg_gradient_pct", sa.Float()),
        sa.Column("climb_category", sa.String(16)),
        sa.Column("aet_hr_dfa", sa.Integer()),
        sa.Column("ant_hr_dfa", sa.Integer()),
        sa.Column("aet_hr_proxy", sa.Integer()),
        sa.Column("dfa_quality", sa.String(16)),
        sa.Column("resp_rate_rsa", sa.Float()),
        sa.Column("rr_intervals_ms", postgresql.ARRAY(sa.Float())),
        sa.Column("epoc_score", sa.Float()),
        sa.Column("recovery_tax_hours", sa.Float()),
        sa.Column("time_at_threshold_min", sa.Float()),
        sa.Column("tte_z4z5_min", sa.Float()),
        sa.Column("critical_hr", sa.Float()),
        sa.Column("tati_score", sa.Float()),
        sa.Column("fat_kcal", sa.Float()),
        sa.Column("carb_kcal", sa.Float()),
        sa.Column("fat_g", sa.Float()),
        sa.Column("carb_g", sa.Float()),
        sa.Column("fluid_loss_l", sa.Float()),
        sa.Column("heat_flag", sa.Boolean()),
    )
    op.create_index("ix_activity_metrics_version", "activity_metrics", ["metrics_version"])

    # ── records – vteřinová data, TimescaleDB hypertable ───────────────────
    op.create_table(
        "records",
        sa.Column("activity_id", sa.String(64), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=False), primary_key=True),
        sa.Column("heart_rate", sa.Float()),
        sa.Column("speed", sa.Float()),
        sa.Column("power", sa.Float()),
        sa.Column("cadence", sa.Float()),
        sa.Column("altitude", sa.Float()),
        sa.Column("distance", sa.Float()),
        sa.Column("temperature", sa.Float()),
        sa.Column("vertical_oscillation", sa.Float()),
        sa.Column("stance_time", sa.Float()),
        sa.Column("respiratory_rate", sa.Float()),
        sa.Column("hrv", sa.Float()),
        sa.Column("position_lat", sa.Float()),
        sa.Column("position_long", sa.Float()),
        sa.Column("hr_zone", sa.String(4)),
        sa.Column("is_active", sa.Boolean()),
        sa.Column("trimp_increment", sa.Float()),
    )
    op.create_index("ix_records_activity_ts", "records", ["activity_id", "timestamp"])

    # Hypertable: chunk po 7 dnech. Aktivity trvají hodiny, takže dotaz na
    # jednu aktivitu sáhne vždy jen do jednoho chunku.
    op.execute(
        "SELECT create_hypertable('records', 'timestamp', "
        "chunk_time_interval => INTERVAL '7 days', migrate_data => TRUE)"
    )

    # Komprese starších chunků. Segmentace po activity_id – dotazy jsou
    # vždy per-activity, takže dekomprese sáhne jen na relevantní segment.
    # Chráněno DO blokem: na edici bez komprese migrace nesmí spadnout.
    op.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE records SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'activity_id',
                timescaledb.compress_orderby = 'timestamp'
            );
            PERFORM add_compression_policy('records', INTERVAL '30 days');
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Komprese records nedostupná (%), pokračuji bez ní.', SQLERRM;
        END $$;
        """
    )

    # ── daily_biometrics ───────────────────────────────────────────────────
    op.create_table(
        "daily_biometrics",
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column("hrv_last_night", sa.Float()),
        sa.Column("hrv_weekly_avg", sa.Float()),
        sa.Column("resting_heart_rate", sa.Float()),
        sa.Column("sleep_score", sa.Float()),
        sa.Column("sleep_duration_min", sa.Float()),
        sa.Column("sleep_deep_min", sa.Float()),
        sa.Column("sleep_light_min", sa.Float()),
        sa.Column("sleep_rem_min", sa.Float()),
        sa.Column("sleep_awake_min", sa.Float()),
        sa.Column("stress_average", sa.Float()),
        sa.Column("body_battery_max", sa.Float()),
        sa.Column("body_battery_min", sa.Float()),
        sa.Column("vo2_max", sa.Float()),
        sa.Column("steps", sa.BigInteger()),
        sa.Column("intensity_minutes", sa.Float()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── daily_metrics ──────────────────────────────────────────────────────
    op.create_table(
        "daily_metrics",
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column("metrics_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("trimp", sa.Float()),
        sa.Column("trimp_epoc", sa.Float()),
        sa.Column("ctl", sa.Float()),
        sa.Column("atl", sa.Float()),
        sa.Column("tsb", sa.Float()),
        sa.Column("acwr", sa.Float()),
        sa.Column("ctl_ramp_rate", sa.Float()),
        sa.Column("ctl_ramp_warning", sa.Boolean()),
        sa.Column("monotony", sa.Float()),
        sa.Column("strain", sa.Float()),
        sa.Column("whoop_strain", sa.Float()),
        sa.Column("daily_efficiency", sa.Float()),
        sa.Column("ef_trend", sa.Float()),
        sa.Column("fatigue_index", sa.Float()),
        sa.Column("polarization_low_pct", sa.Float()),
        sa.Column("polarization_high_pct", sa.Float()),
        sa.Column("z3_junk_pct", sa.Float()),
        sa.Column("polarization_efficiency", sa.Float()),
        sa.Column("readiness_score", sa.Float()),
        sa.Column("pure_recovery_score", sa.Float()),
        sa.Column("hrv_last_night", sa.Float()),
        sa.Column("hrv_weekly_avg", sa.Float()),
        sa.Column("hrv_cv_pct", sa.Float()),
        sa.Column("rhr_day", sa.Float()),
        sa.Column("avg_stress_day", sa.Float()),
        sa.Column("sleep_score_day", sa.Float()),
        sa.Column("sleep_duration_min", sa.Float()),
        sa.Column("sleep_need_min", sa.Float()),
        sa.Column("sleep_performance_pct", sa.Float()),
        sa.Column("max_hrr_60s_avg", sa.Float()),
        sa.Column("stress_flag_count", sa.Integer()),
        sa.Column("illness_warning", sa.Boolean()),
        sa.Column("stress_flags", sa.Text()),
        sa.Column("coach_advice", sa.Text()),
        sa.Column("epoc_score_daily", sa.Float()),
        sa.Column("recovery_tax_hours_daily", sa.Float()),
        sa.Column("fat_kcal_daily", sa.Float()),
        sa.Column("carb_kcal_daily", sa.Float()),
        sa.Column("fat_g_daily", sa.Float()),
        sa.Column("carb_g_daily", sa.Float()),
        sa.Column("fluid_loss_l_daily", sa.Float()),
    )

    # ── sync_state ─────────────────────────────────────────────────────────
    op.create_table(
        "sync_state",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("sync_state")
    op.drop_table("daily_metrics")
    op.drop_table("daily_biometrics")
    op.drop_table("records")
    op.drop_table("activity_metrics")
    op.drop_table("activities")
