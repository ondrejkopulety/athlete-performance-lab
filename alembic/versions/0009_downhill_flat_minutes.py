"""Čas z kopce a po rovině

Revision ID: 0009
Revises: 0008

Stránka Stats na webu potřebuje terénní rozklad jízdy (Do kopce / Z kopce /
Po rovině), ale dřív se z FIT parseru ukládal jen ``uphill_minutes``.
``downhill_minutes`` a ``flat_minutes`` jsou symetrický doplněk – stejná
podmínka pohybu (``is_active`` & ``spd > 0``), jen podle znaménka změny
nadmořské výšky (viz ``src/ingestion/fit_parser.py``).

Sloupce na ``activities``, ne v ``activity_metrics``: je to surové odvození
přímo z FIT záznamů při parsování, ne z odvozených denních/aktivitních
metrik – stejně jako ``uphill_minutes``. Existující aktivity mají tyhle
sloupce ``NULL``, dokud neproběhne ``scripts/main.py load --force``
(reparse FIT, hash souboru se nemění, takže inkrementální LOAD by je jinak
přeskočil).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("activities", sa.Column("downhill_minutes", sa.Float()))
    op.add_column("activities", sa.Column("flat_minutes", sa.Float()))


def downgrade() -> None:
    op.drop_column("activities", "flat_minutes")
    op.drop_column("activities", "downhill_minutes")
