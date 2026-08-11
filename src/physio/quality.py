"""
quality.py – je to opravdu R-R, nebo dopočítaná tepová křivka?
===============================================================

Přítomnost ``hrv`` zpráv ve FIT souboru **neznamená**, že soubor obsahuje
variabilitu mezi tepy. Diagnostika nad 79 soubory s ``hrv`` zprávami
(všechny 2026, všechny s připojeným ANT+ pásem) ukázala u všech stejný
podpis:

  • 63–74 % sousedních R-R rozdílů je **přesně nula**
  • 64–131 unikátních hodnot na 622 až 61 083 tepů
  • hodnoty leží na reciproké mřížce ``RR ≈ K/n`` pro celá ``n``
    s průměrnou odchylkou 0,34 ms

Ukázka surových hodnot (activity_22985785304, ms)::

    623 630 630 637 644 650 650 650 658 658 658 658 665 665 665 665
    673 673 673 673 673 673 673 673 673 673 673 665 665 665 658 658

To je kvantizovaná, vyhlazená tepová křivka převedená na intervaly, ne
naměřené časy tepů. Veličina, která se v 70 % kroků nezmění vůbec a jinak
skočí o jeden dílek mřížky, žádnou variabilitu mezi tepy nenese.

Praktický dopad: DFA-alpha1 měří korelační strukturu fluktuací mezi tepy.
Nad touto řadou vychází α1 ≈ 1,8, což je hodnota pro Brownovský
(integrovaný) signál – nezávislá implementace i neurokit2 dávají totéž
(1,77–1,84 vs 1,56). Prahy z toho nevypadnou žádnou implementací, protože
α1 nikdy neklesne k 0,75.

Tenhle modul proto stojí **před** jakýmkoli výpočtem: rozliší data, na
kterých má smysl počítat, od dat, kde by výsledek byl jen dobře vypadající
číslo bez opory v měření.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import (
    RR_AUTHENTICITY_MIN_BEATS,
    RR_MAX_ZERO_DIFF_PCT,
    RR_MIN_LATTICE_COVERAGE,
)

log = logging.getLogger("physio.quality")

# Rozlišení, ve kterém FIT ukládá R-R: uint16 se scale 1000 → 1 ms.
_FIT_RESOLUTION_MS = 1.0


@dataclass(frozen=True)
class RrAuthenticity:
    """
    Posudek, jestli řada nese skutečnou variabilitu mezi tepy.

    Attributes:
        verdict: ``"beat_to_beat"`` (použitelné), ``"synthetic"`` (dopočítaná
            křivka) nebo ``"unknown"`` (příliš málo tepů na rozhodnutí).
        zero_diff_pct: Podíl sousedních rozdílů rovných přesně nule, 0.0–1.0.
        unique_values: Počet různých hodnot v řadě.
        lattice_coverage: Podíl obsazených hodnot mřížky 1 ms mezi minimem a
            maximem řady. U skutečného měření se blíží 1, u dopočítané
            křivky zůstává nízký. ``None``, když je řada na tenhle test krátká.
        rmssd_ms: RMSSD v milisekundách – doplňkový údaj do reportu.
        reason: Slovní zdůvodnění verdiktu.
    """

    verdict: str
    zero_diff_pct: float
    unique_values: int
    lattice_coverage: float | None
    rmssd_ms: float | None
    reason: str

    @property
    def usable(self) -> bool:
        """True, jen když jde o skutečné beat-to-beat R-R."""
        return self.verdict == "beat_to_beat"


def assess_rr_authenticity(
    rr_seconds: np.ndarray | list[float],
    min_beats: int = RR_AUTHENTICITY_MIN_BEATS,
    max_zero_diff_pct: float = RR_MAX_ZERO_DIFF_PCT,
    min_lattice_coverage: float = RR_MIN_LATTICE_COVERAGE,
) -> RrAuthenticity:
    """
    Posoudí, jestli R-R řada nese variabilitu mezi tepy.

    Dva nezávislé testy:

    1. **Podíl nulových rozdílů.** U měřených časů tepů s rozlišením 1 ms se
       dvě po sobě jdoucí hodnoty shodnou jen občas. Vysoký podíl přesných
       shod znamená, že se hodnota mezi tepy nemění, protože se nemění
       veličina, ze které je dopočítaná.

    2. **Obsazenost mřížky.** Kolik různých hodnot řada využije z těch, které
       jsou mezi jejím minimem a maximem po 1 ms dostupné. Test se pouští jen
       tehdy, když je tepů výrazně víc než dostupných hodnot – jinak je nízká
       obsazenost jen důsledek krátkého záznamu.

    Args:
        rr_seconds: R-R intervaly v sekundách.
        min_beats: Minimální počet tepů pro rozhodnutí.
        max_zero_diff_pct: Podíl nulových rozdílů, nad kterým je řada
            označená za dopočítanou.
        min_lattice_coverage: Obsazenost mřížky, pod kterou je řada označená
            za dopočítanou.

    Returns:
        RrAuthenticity s verdiktem a měřenými hodnotami, ze kterých vyšel.
    """
    arr = np.asarray(rr_seconds, dtype=float)

    if arr.size < min_beats:
        return RrAuthenticity(
            verdict="unknown",
            zero_diff_pct=0.0,
            unique_values=int(len(np.unique(arr))),
            lattice_coverage=None,
            rmssd_ms=None,
            reason=f"jen {arr.size} tepů, na posouzení je potřeba aspoň {min_beats}",
        )

    rr_ms = arr * 1000.0
    diffs = np.diff(rr_ms)
    zero_diff_pct = float(np.mean(np.isclose(diffs, 0.0, atol=1e-6)))
    rmssd_ms = float(np.sqrt(np.mean(diffs**2)))

    unique_values = int(len(np.unique(np.round(rr_ms, 3))))
    span_slots = int(np.round(rr_ms.max() - rr_ms.min()) / _FIT_RESOLUTION_MS) + 1

    # Obsazenost má vypovídací hodnotu, jen když je tepů dost na to, aby se
    # mřížka vůbec dala zaplnit. Faktor 5 je konzervativní: při 5 tepech na
    # dostupnou hodnotu by náhodné rozdělení obsadilo přes 99 % mřížky.
    if arr.size >= 5 * span_slots and span_slots > 1:
        lattice_coverage: float | None = unique_values / span_slots
    else:
        lattice_coverage = None

    if zero_diff_pct > max_zero_diff_pct:
        return RrAuthenticity(
            verdict="synthetic",
            zero_diff_pct=zero_diff_pct,
            unique_values=unique_values,
            lattice_coverage=lattice_coverage,
            rmssd_ms=rmssd_ms,
            reason=(
                f"{zero_diff_pct * 100:.0f} % sousedních rozdílů je přesně nula "
                f"(limit {max_zero_diff_pct * 100:.0f} %) – hodnota se mezi tepy nemění, "
                f"jde o dopočítanou tepovou křivku, ne o naměřené R-R"
            ),
        )

    if lattice_coverage is not None and lattice_coverage < min_lattice_coverage:
        return RrAuthenticity(
            verdict="synthetic",
            zero_diff_pct=zero_diff_pct,
            unique_values=unique_values,
            lattice_coverage=lattice_coverage,
            rmssd_ms=rmssd_ms,
            reason=(
                f"jen {unique_values} různých hodnot z {span_slots} dostupných "
                f"({lattice_coverage * 100:.0f} %, limit {min_lattice_coverage * 100:.0f} %) "
                f"na {arr.size} tepů – řada leží na hrubé mřížce"
            ),
        )

    return RrAuthenticity(
        verdict="beat_to_beat",
        zero_diff_pct=zero_diff_pct,
        unique_values=unique_values,
        lattice_coverage=lattice_coverage,
        rmssd_ms=rmssd_ms,
        reason=(
            f"{zero_diff_pct * 100:.1f} % nulových rozdílů, RMSSD {rmssd_ms:.1f} ms – "
            f"řada nese variabilitu mezi tepy"
        ),
    )
