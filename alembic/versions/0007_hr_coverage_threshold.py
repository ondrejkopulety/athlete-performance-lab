"""Pokrytí jízdy, rozdělení délek úseků a editovatelný práh

Revision ID: 0007
Revises: 0006

Tři věci, které dashboard potřebuje a v databázi nebyly.

activity_hr_coverage
--------------------
Pokrytí dosud vznikalo v ``physio/hr_batch.py`` a po výpisu do CLI se
zahodilo. Jenže "můžu tomuhle číslu věřit" není vlastnost běhu skriptu, ale
vlastnost jízdy – patří tedy do databáze vedle metrik, které na něm stojí.

Vlastní tabulka, ne sloupce v ``activity_metrics``: platnost pokrytí visí na
``HR_CURVE_VERSION`` a limitu ffillu, ne na ``ACTIVITY_METRICS_VERSION``.
V jedné tabulce by se dvě nezávislé verze pletly.

Ukládají se SEKUNDY, ne procenta. Procenta jsou podíl dvou uložených čísel a
dopočítat se dají kdykoli; obráceně to nejde, a rozdíl mezi "80 % z hodiny" a
"80 % ze čtyř hodin" je pro důvěru v číslo podstatný.

max_curve_duration_s je nejdelší okno, které téhle jízdě v tepové křivce
vyšlo. Bez něj by věta "tepová křivka nemá okna delší než 45 min" musela
vzniknout druhým dotazem do activity_hr_curve při každém vykreslení karty.

segment_hist_counts / segment_hist_seconds
------------------------------------------
Rozdělení délek úseků do košů. Ze ``segment_count`` a ``median_segment_s`` ho
sestavit nelze a počítat ho při zobrazení by znamenalo znovu číst sekundová
data. Vzniká ve stejném průchodu segmentací, který už teď dělá zbytek řádku.

Pole, ne řádky: koše jsou tři sta tisíc řádků navíc (798 aktivit × 11 prahů ×
2 tolerance × 6 košů), zatímco takhle je to šest čísel v existujícím řádku.
Pořadí odpovídá ``settings.HR_SEGMENT_BUCKETS_S``.

Obojí je NOT NULL s prázdným polem jako default, aby se "nula úseků" neslilo
s "nespočítáno" – ta samá úvaha, proč řádek vzniká i pro práh, nad kterým
jezdec nebyl ani sekundu.

athlete_threshold
-----------------
Historie nastavení prahu, ne jeden přepisovaný řádek. Dashboard má ukazovat
"LTHR 178 · nastaveno před N dny" a to N musí být z něčeho měřitelného –
z ``mtime`` souboru settings.py se odvodit nedá a při každém deploy by se
vynulovalo.

Zóny se z tohohle prahu NEODVOZUJÍ. ZONES v settings.py zůstávají měřené z
laktátového testu; LTHR řídí jen lookup prahu v panelu bloků a odznak stáří.
Změna prahu proto nikdy neznamená přepočet uložených dat – mění se dotaz,
ne data.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "activity_hr_coverage",
        sa.Column("activity_id", sa.String(64), nullable=False),
        # Rozsah aktivity od prvního do posledního záznamu.
        sa.Column("span_s", sa.Integer(), nullable=False),
        # Sekundy se skutečně naměřeným vzorkem (bez ffillu) – hustota zápisu.
        sa.Column("measured_s", sa.Integer(), nullable=False),
        # Sekundy s použitelnou hodnotou PO ffillu – na tomhle stojí varování.
        sa.Column("usable_s", sa.Integer(), nullable=False),
        sa.Column("longest_gap_s", sa.Integer(), nullable=False),
        # Nejdelší okno, které jízdě v tepové křivce vyšlo; NULL = žádné.
        sa.Column("max_curve_duration_s", sa.Integer()),
        sa.Column("calc_version", sa.SmallInteger(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["activity_id"], ["activities.activity_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("activity_id"),
    )
    op.create_index(
        "ix_activity_hr_coverage_calc_version", "activity_hr_coverage", ["calc_version"]
    )

    op.add_column(
        "activity_hr_blocks",
        sa.Column(
            "segment_hist_counts",
            sa.ARRAY(sa.Integer()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "activity_hr_blocks",
        sa.Column(
            "segment_hist_seconds",
            sa.ARRAY(sa.Integer()),
            nullable=False,
            server_default="{}",
        ),
    )

    op.create_table(
        "athlete_threshold",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("lthr_bpm", sa.Integer(), nullable=False),
        sa.Column("hr_max_bpm", sa.Integer(), nullable=False),
        # Odkdy hodnota platí – zadává uživatel, typicky datum testu.
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("note", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_athlete_threshold_valid_from", "athlete_threshold", ["valid_from"]
    )


def downgrade() -> None:
    op.drop_index("ix_athlete_threshold_valid_from", table_name="athlete_threshold")
    op.drop_table("athlete_threshold")
    op.drop_column("activity_hr_blocks", "segment_hist_seconds")
    op.drop_column("activity_hr_blocks", "segment_hist_counts")
    op.drop_index(
        "ix_activity_hr_coverage_calc_version", table_name="activity_hr_coverage"
    )
    op.drop_table("activity_hr_coverage")
