"""
hr_batch.py – dávkový předvýpočet tepové křivky a bloků
========================================================

Spojuje přípravu streamu (``hr_stream``), obě jádra (``hr_curve``,
``hr_blocks``) a zápis do databáze. Jádra samotná o databázi nevědí – tady
je to jediné místo, kde se potkají s tabulkou ``records``.

Proč to neběží paralelně jako extrakce R-R
-------------------------------------------
U R-R je drahé parsování FIT souborů, proto tam ProcessPoolExecutor je.
Tady je vstupem tabulka ``records`` a rozložení nákladů je jiné: celý
dataset (799 aktivit, 5,6 milionu sekund mřížky) se přečte jedním dotazem
za ~5 s a spočítá za ~2 s. Procesní pool by na dvousekundový numpy výpočet
přidal spawn workerů a vlastní engine na proces, tedy víc režie než práce.

Cache a progress bar tu naopak smysl dávají a fungují stejně jako u R-R.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import (
    HR_BLOCK_BRIDGE_MAX_DROP_BPM,
    HR_BLOCK_BRIDGE_TOLERANCES_S,
    HR_BLOCK_LONG_S,
    HR_BLOCK_SMOOTH_S,
    HR_BLOCK_THRESHOLDS_BPM,
    HR_BLOCKS_VERSION,
    HR_CURVE_DURATIONS_S,
    HR_CURVE_VERSION,
    HR_GRID_FFILL_LIMIT_S,
)
from src.physio.hr_blocks import block_rows
from src.physio.hr_curve import curve_rows
from src.physio.hr_stream import longest_gap, sample_coverage, to_second_grid

log = logging.getLogger("physio.hr_batch")


@dataclass(frozen=True)
class ActivityCoverage:
    """
    Kolik z rozsahu aktivity stojí na datech – ve dvou různých smyslech.

    Rozlišit je potřeba, protože Smart Recording zapisuje vzorek jednou za
    5 sekund a 305 z 799 aktivit v databázi ho má. Taková jízda má hustotu
    vzorků kolem 20 %, ale po doplnění mezer (ffill do 5 s) je její mřížka
    plná a všechna okna z ní vycházejí normálně. Kdyby varování viselo na
    hustotě vzorků, křičelo by u 588 aktivit a skutečné výpadky by se v tom
    ztratily.

      sample_density_pct  podíl skutečně naměřených vzorků – nízká hodnota
                          při plném pokrytí znamená Smart Recording, ne
                          ztrátu dat
      coverage_pct        podíl sekund s použitelnou hodnotou po ffillu –
                          tohle rozhoduje, jestli okno křivky vznikne a kde
                          se blok přeruší, takže na tom visí varování
    """

    activity_id: str
    measured_s: int
    usable_s: int
    span_s: int
    longest_gap_s: int
    curve_windows: int

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.usable_s / self.span_s if self.span_s else 0.0

    @property
    def sample_density_pct(self) -> float:
        return 100.0 * self.measured_s / self.span_s if self.span_s else 0.0


@dataclass
class HrBatchResult:
    """Co dávka spočítala – vstup pro zápis i pro souhrn na stdout."""

    curve_rows: list[dict] = field(default_factory=list)
    block_rows: list[dict] = field(default_factory=list)
    processed: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    skipped_no_hr: list[str] = field(default_factory=list)
    coverage: list[ActivityCoverage] = field(default_factory=list)
    grid_seconds: int = 0
    gap_seconds: int = 0

    @property
    def gap_pct(self) -> float:
        """Kolik procent sekundové mřížky jsou díry po autopauze."""
        return 100.0 * self.gap_seconds / self.grid_seconds if self.grid_seconds else 0.0

    def low_coverage(self, threshold_pct: float) -> list[ActivityCoverage]:
        """Aktivity pod zadaným pokrytím, od nejhorší."""
        return sorted(
            (c for c in self.coverage if c.coverage_pct < threshold_pct),
            key=lambda c: c.coverage_pct,
        )


def compute_activity(
    activity_id: str, records: pd.DataFrame
) -> tuple[list[dict], list[dict], ActivityCoverage]:
    """
    Spočítá křivku i bloky jedné aktivity z jejích vteřinových záznamů.

    Args:
        activity_id: ID aktivity.
        records: Sloupce ``timestamp`` a ``heart_rate`` jedné aktivity.

    Returns:
        ``(řádky_křivky, řádky_bloků, pokrytí)``.
    """
    timestamps = records["timestamp"].to_numpy()
    heart_rate = records["heart_rate"].to_numpy(dtype=float)

    measured_s, span_s = sample_coverage(timestamps, heart_rate)
    grid = to_second_grid(timestamps, heart_rate, ffill_limit_s=HR_GRID_FFILL_LIMIT_S)
    if grid.size == 0:
        return [], [], ActivityCoverage(activity_id, 0, 0, 0, 0, 0)

    curve = curve_rows(activity_id, grid, HR_CURVE_DURATIONS_S, HR_CURVE_VERSION)
    blocks = block_rows(
        activity_id,
        grid,
        thresholds_bpm=HR_BLOCK_THRESHOLDS_BPM,
        bridge_tolerances_s=HR_BLOCK_BRIDGE_TOLERANCES_S,
        bridge_max_drop_bpm=HR_BLOCK_BRIDGE_MAX_DROP_BPM,
        long_block_s=HR_BLOCK_LONG_S,
        smooth_s=HR_BLOCK_SMOOTH_S,
        calc_version=HR_BLOCKS_VERSION,
    )
    return curve, blocks, ActivityCoverage(
        activity_id=activity_id,
        measured_s=measured_s,
        usable_s=int(grid.size - np.isnan(grid).sum()),
        span_s=span_s,
        longest_gap_s=longest_gap(grid),
        curve_windows=len(curve),
    )


def run_batch(
    series: pd.DataFrame,
    todo: list[str],
    cached: list[str] | None = None,
    progress=None,
) -> HrBatchResult:
    """
    Spočítá křivku a bloky pro zadané aktivity.

    Args:
        series: Sloupce ``activity_id``, ``timestamp``, ``heart_rate`` pro
            všechny aktivity najednou (``repo.read_hr_series``).
        todo: ID aktivit ke zpracování.
        cached: ID přeskočená kvůli cache – jen pro souhrn.
        progress: Volitelný callback ``(hotovo, celkem)``.

    Returns:
        HrBatchResult se řádky pro obě tabulky.
    """
    result = HrBatchResult(cached=list(cached or []))
    if series.empty:
        result.skipped_no_hr = list(todo)
        return result

    by_activity = {aid: g for aid, g in series.groupby("activity_id", sort=False)}
    total = len(todo)
    if progress:
        progress(0, total)

    for done, activity_id in enumerate(todo, start=1):
        group = by_activity.get(activity_id)
        if group is None or group.empty:
            # Aktivita bez tepu (posilovna bez pásu, poškozený záznam).
            # Není to chyba – jen z ní křivka ani bloky nevzniknou.
            result.skipped_no_hr.append(activity_id)
        else:
            curve, blocks, cov = compute_activity(activity_id, group)
            result.curve_rows.extend(curve)
            result.block_rows.extend(blocks)
            result.grid_seconds += cov.span_s
            result.gap_seconds += cov.span_s - cov.usable_s
            if curve or blocks:
                result.processed.append(activity_id)
                result.coverage.append(cov)
            else:
                result.skipped_no_hr.append(activity_id)
        if progress:
            progress(done, total)

    return result
