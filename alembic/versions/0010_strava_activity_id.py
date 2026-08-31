"""Odkaz na aktivitu na Stravě

Revision ID: 0010
Revises: 0009

Ke každé jízdě chceme odkaz ``strava.com/activities/{id}``. Problém: to ID
není ``activities.activity_id`` – tím je u většiny jízd Garmin ID (Garmin
soubor s hrudním pásem vyhrává deduplikaci) a Strava protějšek se i s jeho
ID dnes zahazuje.

``strava_id`` si pamatuje ID spárovaného Strava souboru **bez ohledu na to,
kdo vyhrál deduplikaci** (viz ``src/ingestion/dedup.py``). ``NULL`` znamená,
že aktivita na Stravě není (typicky jen z hodinek, nikdy nenahraná).

Existující řádky mají sloupec ``NULL``, dokud neproběhne
``scripts/main.py load --force`` – hash FIT souboru se nemění, takže
inkrementální LOAD by párování jinak nedoplnil.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("activities", sa.Column("strava_id", sa.String(length=64)))


def downgrade() -> None:
    op.drop_column("activities", "strava_id")
