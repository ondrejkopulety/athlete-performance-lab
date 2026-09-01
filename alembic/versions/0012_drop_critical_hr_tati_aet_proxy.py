"""odstranění critical_hr, tati_score a aet_hr_proxy

Revision ID: 0012
Revises: 0011

Audit doménové logiky (8/2026) – tři metriky bez fyziologické opory se ruší:

1. ``daily_metrics.critical_hr`` a ``activity_metrics.tati_score``.
   „Critical HR" (85. percentil průměrného tepu, strop 0,85·MaxHR) a TATI
   („bpm·min nad Critical HR") přebíraly Monod–Scherrerův model kritického
   VÝKONU a aplikovaly ho na tep. Tep je omezená, neaditivní veličina;
   „bpm·minuty nad prahem" nejsou práce a nemají interpretaci. TATI navíc
   vážila každou minutu v zóně STŘEDEM zóny, ne skutečným tepem.

2. ``activity_metrics.aet_hr_proxy``. Náhradní odhad aerobního prahu ze
   „zlomu linearity tep↔rychlost" (argmax rozdílu sklonů binovaného
   HR-vs-speed) není uznávaná metoda – žádný fyziologický základ pro
   „největší nárůst sklonu = AeT".

Navazuje migrace 0013, která ruší i DFA-alpha1 prahy a ``trimp_epoc``.

Downgrade sloupce vrátí jako prázdné; hodnoty už se nikde nepočítají.
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("daily_metrics", "critical_hr")
    op.drop_column("activity_metrics", "tati_score")
    op.drop_column("activity_metrics", "aet_hr_proxy")


def downgrade() -> None:
    op.add_column("activity_metrics", sa.Column("aet_hr_proxy", sa.Integer()))
    op.add_column("activity_metrics", sa.Column("tati_score", sa.Float()))
    op.add_column("daily_metrics", sa.Column("critical_hr", sa.Float()))
