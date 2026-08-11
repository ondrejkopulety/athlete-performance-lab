"""Diagnostika R-R intervalů; proxy odhad přestává vystupovat jako DFA

Revision ID: 0005
Revises: 0004

Sloupec aet_hr_dfa nesl u 161 aktivit hodnotu z dfa_alpha1_proxy – odhadu
odvozeného ze zlomu linearity tep↔rychlost, ne z R-R intervalů. Ten odhad
koreluje s průměrným tepem aktivity na 0,74 a leží systematicky ~3 tepy pod
ním, takže jako práh nic neříká. Migrace ho z aet_hr_dfa vyprazdňuje;
zůstává v aet_hr_proxy, kde byl paralelně uložený, takže se nic neztrácí a
jde porovnat, jak moc byl vedle.

Nové sloupce nesou to, co o R-R skutečně víme:

  rr_beat_count      počet validních R-R intervalů po vyčištění
  rr_artifact_pct    podíl tepů vyhozených filtrem artefaktů
  rr_zero_diff_pct   podíl sousedních rozdílů rovných přesně nule
  rr_unique_values   počet různých hodnot v řadě
  rr_lattice_coverage obsazenost mřížky 1 ms mezi minimem a maximem řady
  rr_authenticity    beat_to_beat | synthetic | unknown

Poslední čtyři existují proto, že přítomnost hrv zpráv ve FIT souboru
neznamená, že soubor nese variabilitu mezi tepy – u všech 79 měřených
souborů ji nenese (55–77 % nulových rozdílů, obsazenost mřížky 6–37 %).
Bez těchhle čísel v datech by se ten závěr musel pokaždé znovu dokazovat.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

NEW_COLUMNS = (
    ("rr_beat_count", sa.Integer()),
    ("rr_artifact_pct", sa.Float()),
    ("rr_zero_diff_pct", sa.Float()),
    ("rr_unique_values", sa.Integer()),
    ("rr_lattice_coverage", sa.Float()),
    ("rr_authenticity", sa.String(16)),
)


def upgrade() -> None:
    for name, type_ in NEW_COLUMNS:
        op.add_column("activity_metrics", sa.Column(name, type_))

    # Proxy odhad zůstává v aet_hr_proxy; z aet_hr_dfa mizí, protože tam
    # vystupoval jako hodnota spočítaná z R-R, kterou nikdy nebyl.
    op.execute(
        """
        UPDATE activity_metrics
           SET aet_hr_proxy = COALESCE(aet_hr_proxy, aet_hr_dfa),
               aet_hr_dfa   = NULL
         WHERE dfa_quality = 'proxy'
        """
    )

    # Hodnota 'real' u 64 aktivit znamenala jen "neurokit2 doběhl" – prahy
    # z toho stejně nevyšly (α1 medián 1,56, aet_hr_dfa NULL u všech).
    # Přeznačit na 'synthetic_rr' by předjímalo posudek, který umí spočítat
    # jen src/physio; necháváme prázdné, ať se dopočítá z dat.
    op.execute("UPDATE activity_metrics SET dfa_quality = NULL WHERE dfa_quality IN ('real', 'none')")


def downgrade() -> None:
    # aet_hr_dfa se vrací z aet_hr_proxy tam, kde šlo o proxy odhad.
    op.execute(
        """
        UPDATE activity_metrics
           SET aet_hr_dfa = aet_hr_proxy
         WHERE dfa_quality = 'proxy' AND aet_hr_dfa IS NULL
        """
    )
    for name, _type in NEW_COLUMNS:
        op.drop_column("activity_metrics", name)
