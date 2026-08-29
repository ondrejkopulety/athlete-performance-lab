"""Percentilové pořadí zátěže aktivity

Revision ID: 0008
Revises: 0007

Detail aktivity (nový web) potřebuje verdikt zátěže a strop gauge TRIMP.
Mockup měl pevné vymyšlené prahy (≥300/≥150 TRIMP) – místo nich se ukládá
percentilové pořadí téhle aktivity mezi kardio aktivitami (běh + kolo)
v celé dosavadní historii atleta (viz
``src/analytics/activity.py:compute_trimp_load_percentile``).

Sloupec v ``activity_metrics``, ne vlastní tabulka: je to jedno číslo na
aktivitu ve stejném cyklu jako ``critical_hr``/``tati_score``, se kterými
sdílí i způsob přepočtu – global metrika, přepisuje se pro všechny řádky
při každém běhu ANALYZE, protože nová aktivita posune percentil i starým
záznamům.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "activity_metrics",
        sa.Column("trimp_load_percentile", sa.Float()),
    )


def downgrade() -> None:
    op.drop_column("activity_metrics", "trimp_load_percentile")
