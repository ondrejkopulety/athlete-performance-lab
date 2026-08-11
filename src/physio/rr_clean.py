"""
rr_clean.py – filtr artefaktů v R-R řadě
=========================================

Artefakt = interval, který se liší o víc než ``tolerance`` od klouzavého
mediánu okolních tepů. Median je použitý záměrně místo průměru: jeden
vynechaný nebo zdvojený tep průměr okna vychýlí natolik, že by kolem sebe
vyrobil další falešné artefakty.

Vyhozené tepy se **nenahrazují** interpolací. Chybějící tep by do řady
vnesl umělou hodnotu s nulovou variabilitou vůči sousedům, což je přesně
ta vlastnost, kterou pak analýza variability měří.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import (
    RR_ARTIFACT_MEDIAN_WINDOW,
    RR_ARTIFACT_TOLERANCE,
    RR_UNRELIABLE_ARTIFACT_PCT,
)

log = logging.getLogger("physio.rr_clean")


@dataclass(frozen=True)
class CleanedRr:
    """
    Vyčištěná R-R řada.

    Attributes:
        rr_seconds: R-R intervaly po odstranění artefaktů.
        t_seconds: Kumulativní čas **původní** řady u ponechaných tepů.
            Časová osa se schválně nepřepočítává ze zkrácené řady – jinak by
            se po vyhození artefaktu posunul zbytek aktivity dopředu.
        keep_mask: Boolean maska nad vstupem (True = tep ponechán).
        artifact_pct: Podíl vyhozených tepů, 0.0–1.0.
        reliable: False, když artefaktů je víc než práh ze settings.
    """

    rr_seconds: np.ndarray
    t_seconds: np.ndarray
    keep_mask: np.ndarray
    artifact_pct: float
    reliable: bool

    @property
    def beat_count(self) -> int:
        """Počet ponechaných tepů."""
        return int(len(self.rr_seconds))


def clean_rr(
    rr_seconds: np.ndarray | list[float],
    tolerance: float = RR_ARTIFACT_TOLERANCE,
    median_window: int = RR_ARTIFACT_MEDIAN_WINDOW,
    unreliable_above: float = RR_UNRELIABLE_ARTIFACT_PCT,
) -> CleanedRr:
    """
    Odstraní z R-R řady artefakty podle odchylky od klouzavého mediánu.

    Args:
        rr_seconds: R-R intervaly v sekundách.
        tolerance: Relativní odchylka od mediánu, nad kterou je tep artefakt
            (0.20 = 20 %).
        median_window: Šířka okna klouzavého mediánu v tepech. Liché číslo,
            aby okno bylo symetrické kolem posuzovaného tepu.
        unreliable_above: Podíl artefaktů, nad kterým je celá řada označená
            jako nespolehlivá.

    Returns:
        CleanedRr s vyčištěnou řadou a podílem vyhozených tepů.

    Raises:
        ValueError: Když ``tolerance`` není kladná nebo ``median_window`` < 3.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance musí být kladná, dostal jsem {tolerance}")
    if median_window < 3:
        raise ValueError(f"median_window musí být aspoň 3, dostal jsem {median_window}")

    arr = np.asarray(rr_seconds, dtype=float)

    if arr.size == 0:
        empty = np.empty(0, dtype=float)
        return CleanedRr(empty, empty, np.empty(0, dtype=bool), 0.0, False)

    t_all = np.cumsum(arr)

    # Kratší řada než okno nemá z čeho medián počítat; vrátíme ji beze změny
    # a jako nespolehlivou, ať se s ní dál nepočítá jako s vyčištěnou.
    if arr.size < median_window:
        log.debug("Řada má %d tepů, okno mediánu je %d – filtr se nepoužil.",
                  arr.size, median_window)
        return CleanedRr(arr, t_all, np.ones(arr.size, dtype=bool), 0.0, False)

    median = (
        pd.Series(arr)
        .rolling(median_window, center=True, min_periods=3)
        .median()
        .to_numpy()
    )

    # min_periods=3 pokrývá okraje, ale u velmi krátkých řad může medián
    # vyjít NaN – takový tep raději ponecháme, než abychom ho zahodili naslepo.
    with np.errstate(invalid="ignore"):
        deviation = np.abs(arr - median)
        keep = np.isnan(median) | (deviation <= tolerance * median)

    artifact_pct = float(1.0 - keep.mean())
    reliable = artifact_pct <= unreliable_above

    if not reliable:
        log.warning("Artefakty %.1f %% (limit %.1f %%) – řada je nespolehlivá.",
                    artifact_pct * 100, unreliable_above * 100)

    return CleanedRr(
        rr_seconds=arr[keep],
        t_seconds=t_all[keep],
        keep_mask=keep,
        artifact_pct=artifact_pct,
        reliable=reliable,
    )
