"""Klidový tep jako časová řada, zdroj biometrie, LTHR z terénních dat

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── daily_biometrics: složený klíč (date, source) ─────────────────────
    # Apple Watch a Garmin měří klidový tep měřitelně jinak – slévat je
    # do jedné hodnoty by vyrobilo umělý skok na přechodu 8/2025.
    op.add_column(
        "daily_biometrics",
        sa.Column("source", sa.String(16), nullable=False, server_default="garmin"),
    )
    op.drop_constraint("daily_biometrics_pkey", "daily_biometrics", type_="primary")
    op.create_primary_key("daily_biometrics_pkey", "daily_biometrics", ["date", "source"])
    op.create_index("ix_daily_biometrics_source", "daily_biometrics", ["source"])

    # ── activity_metrics: TRIMP z dobového RHR + podklady pro LTHR ────────
    for col in (
        sa.Column("trimp_adjusted", sa.Float()),
        sa.Column("rhr_used", sa.Float()),
        sa.Column("best_20min_hr", sa.Float()),
        sa.Column("best_30min_hr", sa.Float()),
        sa.Column("best_60min_hr", sa.Float()),
        sa.Column("dfa_alpha1_min", sa.Float()),
        sa.Column("dfa_alpha1_median", sa.Float()),
        sa.Column("dfa_window_count", sa.Integer()),
    ):
        op.add_column("activity_metrics", col)

    # ── daily_metrics: dlouhý baseline a odhad prahu ──────────────────────
    op.add_column("daily_metrics", sa.Column("rhr_baseline_90d", sa.Float()))
    op.add_column("daily_metrics", sa.Column("rhr_source", sa.String(16)))
    op.add_column("daily_metrics", sa.Column("lthr_estimate", sa.Float()))


def downgrade() -> None:
    op.drop_column("daily_metrics", "lthr_estimate")
    op.drop_column("daily_metrics", "rhr_source")
    op.drop_column("daily_metrics", "rhr_baseline_90d")

    for name in (
        "dfa_window_count", "dfa_alpha1_median", "dfa_alpha1_min",
        "best_60min_hr", "best_30min_hr", "best_20min_hr",
        "rhr_used", "trimp_adjusted",
    ):
        op.drop_column("activity_metrics", name)

    op.drop_index("ix_daily_biometrics_source", table_name="daily_biometrics")
    op.drop_constraint("daily_biometrics_pkey", "daily_biometrics", type_="primary")
    op.create_primary_key("daily_biometrics_pkey", "daily_biometrics", ["date"])
    op.drop_column("daily_biometrics", "source")
