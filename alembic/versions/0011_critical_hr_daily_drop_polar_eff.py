"""critical_hr do daily_metrics, zrušení polarization_efficiency

Revision ID: 0011
Revises: 0010

Dvě úpravy z revize výpočtů (8/2026):

1. ``critical_hr`` se přesouvá z ``activity_metrics`` do ``daily_metrics``.
   Byla to jediná atletova hodnota (85. percentil průměrného tepu přes celou
   historii kardio aktivit), takže v master exportu stála 884× stejná. TATI
   ji nepotřebuje jako sloupec – počítá se z lokální proměnné ve stejné
   funkci. Do ``daily_metrics`` jde konstantou; ``tati_score`` zůstává
   per-activity.

2. ``daily_metrics.polarization_efficiency`` se ruší. Algebraicky vycházela
   ≈ ``105 − 2·z3_junk_pct`` (podíly low+high ≈ 100 − z3_pct, penalta
   z3_pct − 5), tedy jen převrácený ``z3_junk_pct`` bez vlastní informace.

Hodnoty se dopočítají samy při nejbližším běhu ANALYZE (bump
DAILY_METRICS_VERSION → full rebuild ``daily_metrics``; per-activity sloupec
jen mizí).
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("daily_metrics", sa.Column("critical_hr", sa.Float()))
    op.drop_column("activity_metrics", "critical_hr")
    op.drop_column("daily_metrics", "polarization_efficiency")


def downgrade() -> None:
    op.add_column("daily_metrics", sa.Column("polarization_efficiency", sa.Float()))
    op.add_column("activity_metrics", sa.Column("critical_hr", sa.Float()))
    op.drop_column("daily_metrics", "critical_hr")
