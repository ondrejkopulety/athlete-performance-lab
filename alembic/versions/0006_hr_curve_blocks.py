"""Tepová křivka a souvislé bloky nad prahem

Revision ID: 0006
Revises: 0005

Dvě metriky, které se z vteřinových dat nedají počítat za běhu, takže se
předpočítávají per aktivita:

  activity_hr_curve   maximální průměrný tep za 5 s až 60 min (obdoba
                      výkonové křivky)
  activity_hr_blocks  jak dlouho vydrží tep nad prahem V KUSE – to, co ze
                      "času v zónách" nevyčteš: 131 minut nad prahem může
                      být 272 úseků s mediánem 6 sekund

Prahy, ne zóny
--------------
activity_hr_blocks nese absolutní práh v bpm (mřížka 135–185 po 5), ne
označení zóny. Zóny se odvozují z LTHR, které se mění (172 → 177 → po
terénním testu znovu), a uložené zóny by při každé změně znamenaly přepočet
celé historie. Takhle je zóna lookup: "Z4 při LTHR 177" = práh 168 →
nejbližší řádek. Změna prahu mění dotaz, ne data.

Tepová křivka je na LTHR nezávislá úplně.

Obě varianty přemostění vedle sebe
----------------------------------
bridge_tolerance_s je součástí primárního klíče, ne parametrem jednoho
běhu: 0 ukazuje surovou fragmentaci, 15 s realistickou souvislost úsilí.
Rozdíl mezi nimi je sám o sobě informace o charakteru jízdy.

calc_version
------------
Obě tabulky ho nesou proto, aby po změně pravidel přemostění nebo vyhlazení
šlo poznat, která čísla vznikla jakou logikou. Bez něj vzniká druhý
dfa_quality = 'proxy' – hodnota, u které se po roce nedá zjistit, co
vlastně znamená.

activity_id je varchar(64), ne bigint: activities.activity_id je v celé
databázi řetězec (Strava ID nejsou čísla), takže na bigint by nešel navázat
cizí klíč.

Tabulky jsou malé (~8 tis. a ~18 tis. řádků při 800 aktivitách), takže
kromě primárního klíče a indexu na calc_version tu nic dalšího není –
index na calc_version obsluhuje cache ("co už je spočítané aktuální
logikou") a filtr při přepočtu.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "activity_hr_curve",
        sa.Column("activity_id", sa.String(64), nullable=False),
        sa.Column("duration_s", sa.Integer(), nullable=False),
        sa.Column("max_mean_hr", sa.Numeric(4, 1), nullable=False),
        sa.Column("calc_version", sa.SmallInteger(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["activity_id"], ["activities.activity_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("activity_id", "duration_s"),
    )
    op.create_index(
        "ix_activity_hr_curve_calc_version", "activity_hr_curve", ["calc_version"]
    )

    op.create_table(
        "activity_hr_blocks",
        sa.Column("activity_id", sa.String(64), nullable=False),
        sa.Column("threshold_bpm", sa.Integer(), nullable=False),
        sa.Column("bridge_tolerance_s", sa.Integer(), nullable=False),
        sa.Column("longest_block_s", sa.Integer(), nullable=False),
        sa.Column("total_time_s", sa.Integer(), nullable=False),
        sa.Column("time_in_long_blocks_s", sa.Integer(), nullable=False),
        sa.Column("segment_count", sa.Integer(), nullable=False),
        sa.Column("median_segment_s", sa.Numeric(6, 1)),
        sa.Column("calc_version", sa.SmallInteger(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["activity_id"], ["activities.activity_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("activity_id", "threshold_bpm", "bridge_tolerance_s"),
    )
    op.create_index(
        "ix_activity_hr_blocks_calc_version", "activity_hr_blocks", ["calc_version"]
    )


def downgrade() -> None:
    op.drop_index("ix_activity_hr_blocks_calc_version", table_name="activity_hr_blocks")
    op.drop_table("activity_hr_blocks")
    op.drop_index("ix_activity_hr_curve_calc_version", table_name="activity_hr_curve")
    op.drop_table("activity_hr_curve")
