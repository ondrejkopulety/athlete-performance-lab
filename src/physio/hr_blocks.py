"""
hr_blocks.py – souvislé bloky nad prahem
=========================================

Jak dlouho vydrží tep nad daným prahem **v kuse**. Ze "času v zónách" se to
vyčíst nedá: 131 minut nad prahem napříč sedmi jízdami může být 272 úseků
s mediánem 6 sekund, a to je úplně jiný trénink než sedm dvacetiminutových
bloků se stejným součtem.

Ukládá se na mřížce absolutních prahů v bpm, ne v zónách – zóny se odvozují
z LTHR, které se mění, a uložené zóny by při každé změně znamenaly přepočet
historie. Zóna je až lookup nad uloženými prahy.

Dvě věci, které dělá tenhle modul jinak než hr_curve.py:

  vyhlazení   Tep se před segmentací vyhladí klouzavým průměrem (10 s).
              Bez toho šum na hranici prahu vyrábí umělé úseky: jízda
              26. 7. 2026 dává nad prahem 172 syrově 54 úseků s mediánem
              10 s, po vyhlazení 30 úseků – přičemž celkový čas (37,9 vs
              37,0 min) ani nejdelší úsek (4,4 vs 4,4 min) se prakticky
              nemění. Vyhlazení tedy nemaže práci, maže šum senzoru.

  pauzy       Pauza úsek **přerušuje**, i kdyby tep zůstal nad prahem.
              Zastavení na křižovatce dělí úsilí na dvě. Přemostění se
              přes díru v datech nikdy nedělá, ani kdyby byla kratší než
              tolerance. (hr_curve.py naopak pauzy v datech nechává – tam
              je stojící jezdec pořád součástí zátěže.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockSummary:
    """
    Souhrn souvislých úseků nad jedním prahem při jedné toleranci.

    ``hist_counts`` a ``hist_seconds`` nesou rozdělení délek úseků po koších
    (``settings.HR_SEGMENT_BUCKETS_S``). Obojí, protože každé říká něco
    jiného: 199 úseků pod 30 sekund vypadá jinak než 18 minut, které
    dohromady dají. Ze ``segment_count`` a ``median_segment_s`` se rozdělení
    zpětně sestavit nedá, a dopočítávat ho při zobrazení by znamenalo znovu
    číst sekundová data – proto se ukládá tady.
    """

    longest_block_s: int
    total_time_s: int
    time_in_long_blocks_s: int
    segment_count: int
    median_segment_s: float | None
    hist_counts: list[int]
    hist_seconds: list[int]


def smooth_within_runs(heart_rate: np.ndarray, window: int) -> np.ndarray:
    """
    Vyhladí tep uvnitř souvislých úseků dat, přes díry nikdy.

    Každý souvislý blok ne-``NaN`` hodnot se vyhladí zvlášť, centrovaným
    klouzavým průměrem s částečnými okny u okrajů. Díry (pauzy) zůstanou
    ``NaN``.

    Proč ne prostě klouzavý průměr přes celou řadu: okno, které by sáhlo
    přes díru, by vrátilo ``NaN``, a kolem každé pauzy by tak zmizelo
    dalších ±5 sekund platných dat. Ta sekundy patří sousedním úsekům –
    pauza je má rozdělit, ne ukrojit.

    Centrované okno (ne zarovnané doprava jako ``pandas.rolling``) proto,
    že vyhlazení nemá posouvat hranice úseků o půl okna dozadu.

    Args:
        heart_rate: Tep na sekundové mřížce; ``NaN`` = díra.
        window: Délka okna v sekundách. ``<= 1`` vrací kopii vstupu.

    Returns:
        Vyhlazený tep, stejná délka, ``NaN`` na stejných místech.
    """
    hr = np.asarray(heart_rate, dtype=float)
    if window <= 1 or hr.size == 0:
        return hr.astype(float, copy=True)

    out = np.full(hr.size, np.nan, dtype=float)
    # Rozdělení (window-1)//2 dozadu a window//2 dopředu dává přesně `window`
    # vzorků i pro sudé okno – "10 s vyhlazení" opravdu průměruje 10 sekund.
    back, fwd = (window - 1) // 2, window // 2
    for start, end in _runs(~np.isnan(hr)):
        seg = hr[start : end + 1]
        csum = np.concatenate(([0.0], np.cumsum(seg)))
        idx = np.arange(seg.size)
        lo = np.maximum(idx - back, 0)
        hi = np.minimum(idx + fwd + 1, seg.size)
        out[start : end + 1] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def find_segments(
    heart_rate: np.ndarray,
    threshold_bpm: float,
    bridge_tolerance_s: int = 0,
    bridge_max_drop_bpm: float = 5.0,
    smooth_s: int = 1,
) -> np.ndarray:
    """
    Souvislé úseky, ve kterých je tep nad prahem.

    Přemostění spojí dva úseky přes krátký propad pod práh, ale jen když
    jsou splněné **obě** podmínky:

      1. propad je kratší nebo roven ``bridge_tolerance_s``
      2. tep v něm neklesne víc než ``bridge_max_drop_bpm`` pod práh

    Druhá podmínka je podstatná. Bez ní se přes sjezd spojí dvě opravdu
    oddělená úsilí a metrika ztratí smysl – "nejdelší blok" by pak měřil,
    jak dlouho jsi byl na kole, ne jak dlouho jsi šlapal.

    Přes díru v datech (pauza) se nepřemosťuje nikdy, bez ohledu na délku.

    Args:
        heart_rate: Tep na spojité sekundové mřížce; ``NaN`` = díra.
        threshold_bpm: Práh v bpm; úsek je "nad prahem" při ``tep >= práh``.
        bridge_tolerance_s: Nejdelší propad, který se ještě přemostí.
            0 = striktní segmentace.
        bridge_max_drop_bpm: O kolik bpm smí tep v přemosťovaném propadu
            klesnout pod práh.
        smooth_s: Okno vyhlazení před segmentací; 1 = bez vyhlazení.

    Returns:
        Pole tvaru ``(n, 2)`` s dvojicemi ``[start, end]`` – indexy sekund,
        oba včetně. Prázdné pole tvaru ``(0, 2)``, když nad prahem nic není.
    """
    hr = np.asarray(heart_rate, dtype=float)
    if hr.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    smoothed = smooth_within_runs(hr, smooth_s) if smooth_s > 1 else hr
    above = np.nan_to_num(smoothed, nan=-np.inf) >= threshold_bpm

    segments = _runs(above)
    if bridge_tolerance_s <= 0 or len(segments) < 2:
        return np.array(segments, dtype=np.int64).reshape(-1, 2)

    is_gap = np.isnan(hr)
    merged: list[list[int]] = [list(segments[0])]
    for start, end in segments[1:]:
        gap_from = merged[-1][1] + 1
        gap_len = start - gap_from
        if gap_len <= bridge_tolerance_s and _bridgeable(
            smoothed[gap_from:start],
            is_gap[gap_from:start],
            threshold_bpm,
            bridge_max_drop_bpm,
        ):
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return np.array(merged, dtype=np.int64).reshape(-1, 2)


def segment_histogram(
    lengths: np.ndarray, buckets_s: Sequence[int]
) -> tuple[list[int], list[int]]:
    """
    Rozdělí délky úseků do košů – počet úseků a jejich součet času.

    ``buckets_s`` jsou horní hranice (včetně); poslední koš je všechno nad
    poslední hranicí, takže výsledek má vždy ``len(buckets_s) + 1`` položek.

    Args:
        lengths: Délky úseků v sekundách.
        buckets_s: Horní hranice košů, vzestupně
            (``settings.HR_SEGMENT_BUCKETS_S``).

    Returns:
        ``(počty_úseků, součty_sekund)``, obojí délky ``len(buckets_s) + 1``.
    """
    edges = [int(b) for b in buckets_s]
    counts = [0] * (len(edges) + 1)
    seconds = [0] * (len(edges) + 1)
    if lengths.size == 0:
        return counts, seconds

    # searchsorted se stranou "left": délka rovná hranici patří do koše pod
    # ní, takže úsek dlouhý přesně 30 s je "<30 s", ne "30–60 s".
    idx = np.searchsorted(np.asarray(edges), lengths, side="left")
    for bucket, length in zip(idx.tolist(), lengths.tolist()):
        counts[bucket] += 1
        seconds[bucket] += int(length)
    return counts, seconds


def summarize_segments(
    segments: np.ndarray, long_block_s: int, buckets_s: Sequence[int] = ()
) -> BlockSummary:
    """
    Převede úseky na čísla, která jdou do ``activity_hr_blocks``.

    ``total_time_s`` je součet délek úseků – u přemostěné varianty tedy
    včetně přemostěných propadů. Je to záměr: přemostěný propad je součástí
    bloku, jinak by "souvislý blok" a "čas v blocích" měřily každý něco
    jiného. Surový čas nad prahem je vždy k dispozici v řádku s
    ``bridge_tolerance_s = 0``.

    Args:
        segments: Pole ``(n, 2)`` z ``find_segments``.
        long_block_s: Délka, nad kterou se úsek počítá jako "dlouhý"
            (``settings.HR_BLOCK_LONG_S``). Porovnává se ostře.
        buckets_s: Horní hranice košů histogramu délek
            (``settings.HR_SEGMENT_BUCKETS_S``).

    Returns:
        BlockSummary; při prázdném vstupu samé nuly, ``median_segment_s=None``
        a histogram plný nul – nula úseků je platná odpověď, ne "nespočítáno".
    """
    segments = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
    empty_counts, empty_seconds = segment_histogram(np.empty(0, dtype=np.int64), buckets_s)
    if segments.size == 0:
        return BlockSummary(0, 0, 0, 0, None, empty_counts, empty_seconds)

    lengths = segments[:, 1] - segments[:, 0] + 1
    long = lengths[lengths > long_block_s]
    counts, seconds = segment_histogram(lengths, buckets_s)
    return BlockSummary(
        longest_block_s=int(lengths.max()),
        total_time_s=int(lengths.sum()),
        time_in_long_blocks_s=int(long.sum()),
        segment_count=int(lengths.size),
        median_segment_s=round(float(np.median(lengths)), 1),
        hist_counts=counts,
        hist_seconds=seconds,
    )


def block_rows(
    activity_id: str,
    heart_rate: np.ndarray,
    thresholds_bpm: Sequence[int],
    bridge_tolerances_s: Sequence[int],
    bridge_max_drop_bpm: float,
    long_block_s: int,
    smooth_s: int,
    calc_version: int,
    segment_buckets_s: Sequence[int] = (),
) -> list[dict]:
    """
    Bloky pro celou mřížku prahů a tolerancí jako řádky pro ``activity_hr_blocks``.

    Řádek vzniká pro **každou** kombinaci prahu a tolerance, i když nad
    prahem není ani sekunda. Nula je platná odpověď ("nad 185 jsem nebyl")
    a odlišuje se tak od "nespočítáno" – jinak by aktivita bez tvrdého
    úsilí neměla řádek žádný a cache by ji počítala pořád dokola.

    Args:
        activity_id: ID aktivity.
        heart_rate: Tep na sekundové mřížce.
        thresholds_bpm: Mřížka prahů.
        bridge_tolerances_s: Varianty přemostění (obě se ukládají).
        bridge_max_drop_bpm: Maximální hloubka přemosťovaného propadu.
        long_block_s: Práh pro "dlouhý" úsek.
        smooth_s: Okno vyhlazení.
        calc_version: Verze výpočtu (``settings.HR_BLOCKS_VERSION``).
        segment_buckets_s: Horní hranice košů histogramu délek úseků.

    Returns:
        Řádky připravené k upsertu.
    """
    hr = np.asarray(heart_rate, dtype=float)
    if hr.size == 0:
        return []

    # Vyhladit stačí jednou pro všechny prahy – práh se aplikuje až na
    # vyhlazenou řadu, takže se výsledek neliší.
    smoothed = smooth_within_runs(hr, smooth_s) if smooth_s > 1 else hr

    rows: list[dict] = []
    for threshold in sorted(set(int(t) for t in thresholds_bpm)):
        for tolerance in sorted(set(int(t) for t in bridge_tolerances_s)):
            segments = find_segments(
                smoothed if smooth_s > 1 else hr,
                threshold_bpm=threshold,
                bridge_tolerance_s=tolerance,
                bridge_max_drop_bpm=bridge_max_drop_bpm,
                # Vyhlazeno už je; opakovaným vyhlazením by se okno rozšířilo.
                smooth_s=1,
            )
            summary = summarize_segments(segments, long_block_s, segment_buckets_s)
            rows.append(
                {
                    "activity_id": activity_id,
                    "threshold_bpm": threshold,
                    "bridge_tolerance_s": tolerance,
                    "longest_block_s": summary.longest_block_s,
                    "total_time_s": summary.total_time_s,
                    "time_in_long_blocks_s": summary.time_in_long_blocks_s,
                    "segment_count": summary.segment_count,
                    "median_segment_s": summary.median_segment_s,
                    "segment_hist_counts": summary.hist_counts,
                    "segment_hist_seconds": summary.hist_seconds,
                    "calc_version": calc_version,
                }
            )
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Pomocné
# ═══════════════════════════════════════════════════════════════════════════

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Souvislé úseky ``True`` jako dvojice indexů (oba včetně)."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _bridgeable(
    gap_values: np.ndarray,
    gap_is_hole: np.ndarray,
    threshold_bpm: float,
    max_drop_bpm: float,
) -> bool:
    """Smí se přes tenhle propad přemostit?"""
    if gap_is_hole.any():
        return False          # pauza dělí úsilí bez ohledu na délku
    if gap_values.size == 0:
        return True
    return bool(np.nanmin(gap_values) >= threshold_bpm - max_drop_bpm)
