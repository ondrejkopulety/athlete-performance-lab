"""odstranění DFA-alpha1 prahů a trimp_epoc

Revision ID: 0013
Revises: 0012

Druhá část auditu doménové logiky (8/2026):

1. ``activity_metrics.aet_hr_dfa`` / ``ant_hr_dfa`` a diagnostiky
   ``dfa_alpha1_min`` / ``dfa_alpha1_median`` / ``dfa_window_count``.
   DFA-alpha1 prahy z R-R intervalů: žádný FIT tohoto atleta neobsahuje
   pravé beat-to-beat R-R (všech 79 souborů s ``hrv`` zprávami nese
   kvantizovanou tepovou křivku – viz src/physio/quality.py), takže metoda
   nikdy neměla vstup a sloupce byly trvale NULL. Časové okno 120 s na tom
   nic nezměnilo – proto se celá větev ruší.
   Sloupec ``dfa_quality`` (posudek pravosti R-R řady: rr_ok / synthetic_rr
   / no_rr / …) ZŮSTÁVÁ – plní ho krok ``rr`` v src/physio nezávisle a nese
   samostatnou informaci „má tahle aktivita reálnou HRV".

2. ``daily_metrics.trimp_epoc`` (TRIMP + 0,1·EPOC skóre). Po přechodu ACWR
   na čistou TRIMP (uncoupled model – aby platily literární prahy 0,8–1,3)
   už tuhle EPOC-váženou variantu nic nečetlo.

Downgrade sloupce vrátí jako prázdné; hodnoty už se nikde nepočítají.
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

_ACTIVITY_DROP = [
    "aet_hr_dfa",
    "ant_hr_dfa",
    "dfa_alpha1_min",
    "dfa_alpha1_median",
    "dfa_window_count",
]


def upgrade() -> None:
    op.drop_column("daily_metrics", "trimp_epoc")
    for col in _ACTIVITY_DROP:
        op.drop_column("activity_metrics", col)


def downgrade() -> None:
    op.add_column("activity_metrics", sa.Column("dfa_window_count", sa.Integer()))
    op.add_column("activity_metrics", sa.Column("dfa_alpha1_median", sa.Float()))
    op.add_column("activity_metrics", sa.Column("dfa_alpha1_min", sa.Float()))
    op.add_column("activity_metrics", sa.Column("ant_hr_dfa", sa.Integer()))
    op.add_column("activity_metrics", sa.Column("aet_hr_dfa", sa.Integer()))
    op.add_column("daily_metrics", sa.Column("trimp_epoc", sa.Float()))
