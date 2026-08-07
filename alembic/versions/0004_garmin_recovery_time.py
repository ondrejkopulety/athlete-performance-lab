"""Zrušit vymyšlenou recovery_tax, importovat Garmin Training Readiness

Revision ID: 0004
Revises: 0003

recovery_tax_hours = min(96, 0.08 × TRIMP^1.2) byl vymyšlený vzorec:
Spearman s total_trimp 0.9955 (tedy žádná informace navíc) a proti
Garminovu naměřenému recovery time RMSE 32.0 h – k nerozeznání od nejlepší
možné konstanty (32.1 h). Nahrazuje ho recovery_time_h – hodnota, kterou spočítal
Firstbeat v hodinkách a která se rok stahovala do training_readiness.csv,
aniž by se kdy naimportovala.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# Sloupce, které přibývají shodně do zdrojové i výstupní tabulky
NEW_COLUMNS = (
    ("recovery_time_h", sa.Float()),
    ("garmin_readiness_score", sa.Float()),
    ("garmin_hrv_factor_pct", sa.Float()),
)


def upgrade() -> None:
    for table in ("daily_biometrics", "daily_metrics"):
        for name, type_ in NEW_COLUMNS:
            op.add_column(table, sa.Column(name, type_))

    op.drop_column("activity_metrics", "recovery_tax_hours")
    op.drop_column("daily_metrics", "recovery_tax_hours_daily")


def downgrade() -> None:
    # Sloupce se vrátí prázdné. Není to ztráta: recovery_tax byla plně
    # odvozená z activities.total_trimp, takže ji lze kdykoli dopočítat.
    op.add_column("activity_metrics", sa.Column("recovery_tax_hours", sa.Float()))
    op.add_column("daily_metrics", sa.Column("recovery_tax_hours_daily", sa.Float()))

    for table in ("daily_biometrics", "daily_metrics"):
        for name, _type in NEW_COLUMNS:
            op.drop_column(table, name)
