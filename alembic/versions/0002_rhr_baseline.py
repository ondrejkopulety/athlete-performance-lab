"""Klidový tep relativně k vlastnímu baseline místo pevného prahu

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Baseline i odchylka se ukládají, aby je viděl dashboard i chatbot:
    # „48 bpm" samo o sobě nic neříká, „48 při běžných 43" ano.
    op.add_column("daily_metrics", sa.Column("rhr_baseline_14d", sa.Float()))
    op.add_column("daily_metrics", sa.Column("rhr_elevation_bpm", sa.Float()))


def downgrade() -> None:
    op.drop_column("daily_metrics", "rhr_elevation_bpm")
    op.drop_column("daily_metrics", "rhr_baseline_14d")
