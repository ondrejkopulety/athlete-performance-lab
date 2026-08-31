"""
hr_stream.py – příprava vteřinového tepu pro křivku a bloky
============================================================

Tohle musí proběhnout dřív než jakýkoli výpočet, jinak jsou obě metriky
nesmysl. Autopauza dělá díry v časové ose: jízda 26. 7. 2026 má 10 693
záznamů rozprostřených přes 17 239 sekund, největší souvislá díra je 3 474 s
(58 minut). Klouzavé okno "20 minut" nad neupravenou řadou by tam pokrylo
46 minut reálného času.

Pauzy se ale pro každou metriku řeší jinak, a vědomě:

  tepová křivka  Pauzy zůstávají v datech. Tep při zastavení neklesá skokově
                 a je součástí zátěže – vteřiny, kdy hodinky měřily stojícího
                 jezdce, se do průměru počítají normálně. Okno, které díru
                 nepřekryje daty, ale nevznikne vůbec (viz hr_curve.py).

  souvislé bloky Pauza je přerušení úseku, i kdyby tep zůstal nad prahem.
                 Zastavení na křižovatce dělí úsilí na dvě, ať už tep spadl
                 nebo ne. Přemostění přes díru se nikdy nedělá, i kdyby byla
                 kratší než tolerance (viz hr_blocks.py).

Rozdíl je v tom, co se metrikou tvrdí: křivka je "jaký nejvyšší průměr jsem
za tu dobu udržel", blok je "jak dlouho jsem šlapal bez přerušení".

Pauza = mezera v timestampech, ne is_active
--------------------------------------------
Autopauza se v datech projevuje jako **chybějící řádky**, ne jako
``is_active = False``. Na jízdě 12. 7. 2026 je 11 187 sekund bez záznamu
(46 % rozsahu) a přitom ani jeden řádek s ``is_active = False``; na jízdě
11. 4. 2026 je to 5 916 sekund děr proti pěti řádkům. Medián přes všechny
aktivity je 12 řádků ``is_active = False`` na aktivitu, tedy o dva řády
míň, než kolik pauz v datech opravdu je. Detekce pauz z toho sloupce by
nedetekovala nic, takže se na něj tenhle modul vůbec nedívá – pauzy poznává
výhradně z mezer v časové ose.

Chybějící tep v existujícím řádku
----------------------------------
Jiná porucha než autopauza, ale zachází se s ní stejně: ``NaN`` v tepu je
nepokrytá sekunda, ať už řádek existuje nebo ne. Aktivita 11. 4. 2026 má
1 580 řádků (16 %) s časem, ale bez tepu, nejdelší souvislý výpadek 26
minut. Bez limitu na ffill by z toho vzniklo 26 minut konstantního tepu, a
z něj falešný souvislý blok (25,8 místo 8,0 min) i falešný hodinový řádek
křivky. Limit ``ffill_limit_s`` proto platí na obě poruchy stejně.

Modul nesahá na databázi ani na FIT – bere pole a vrací pole.
"""

from __future__ import annotations

import numpy as np

from config.settings import HR_PLAUSIBLE_MAX_BPM, HR_PLAUSIBLE_MIN_BPM


def to_second_grid(
    timestamps: np.ndarray,
    heart_rate: np.ndarray,
    ffill_limit_s: int = 5,
) -> np.ndarray:
    """
    Přeindexuje tep na spojitou sekundovou mřížku.

    Mřížka jde od prvního do posledního záznamu po jedné sekundě. Sekundy,
    ke kterým žádný záznam není (autopauza, výpadek senzoru), zůstávají
    ``NaN`` – s výjimkou krátkých děr, které se doplní poslední známou
    hodnotou.

    Args:
        timestamps: Časy záznamů; ``datetime64`` nebo sekundy jako čísla.
            Nemusí být seřazené a smí obsahovat duplicity – sloučené
            fragmenty Strava exportů je běžně mají.
        heart_rate: Tep v bpm, stejná délka jako ``timestamps``. ``NaN``
            znamená "záznam je, ale tep v něm chybí".
        ffill_limit_s: Nejdelší díra, kterou ještě doplní poslední známá
            hodnota. Delší zůstane ``NaN``.

    Returns:
        1D pole tepů délky ``(poslední − první) + 1`` sekund. Prázdný vstup
        vrací prázdné pole.
    """
    ts = np.asarray(timestamps)
    hr = np.asarray(heart_rate, dtype=float)
    if ts.size == 0 or hr.size == 0:
        return np.empty(0, dtype=float)

    # Implausibilní vzorek (glitch senzoru) = nepokrytá sekunda, ne tep.
    # Musí padnout dřív, než ffill roztáhne poslední známou hodnotu –
    # jinak by se 3bpm výpadek nebo 240bpm špička propsaly do okolí.
    hr = np.where((hr < HR_PLAUSIBLE_MIN_BPM) | (hr > HR_PLAUSIBLE_MAX_BPM), np.nan, hr)
    if ts.size != hr.size:
        raise ValueError(f"timestamps a heart_rate mají různou délku: {ts.size} vs {hr.size}")

    seconds = _to_seconds(ts)

    order = np.argsort(seconds, kind="stable")
    seconds = seconds[order]
    hr = hr[order]

    # Duplicitní sekunda: bere se první záznam, který nese tep. Průměrovat
    # by znamenalo míchat dva různé zdroje téže vteřiny.
    keep = np.ones(seconds.size, dtype=bool)
    keep[1:] = seconds[1:] != seconds[:-1]
    if not keep.all():
        for idx in np.flatnonzero(~keep):
            first = idx
            while first > 0 and seconds[first - 1] == seconds[idx]:
                first -= 1
            if np.isnan(hr[first]) and not np.isnan(hr[idx]):
                hr[first] = hr[idx]
        seconds = seconds[keep]
        hr = hr[keep]

    span = int(seconds[-1] - seconds[0]) + 1
    grid = np.full(span, np.nan, dtype=float)
    grid[(seconds - seconds[0]).astype(np.int64)] = hr

    return _ffill(grid, ffill_limit_s)


def sample_coverage(timestamps: np.ndarray, heart_rate: np.ndarray) -> tuple[int, int]:
    """
    Kolik sekund aktivity je pokryto **skutečně naměřeným** tepem.

    Počítají se jen vzorky, které přišly ze záznamu – ne hodnoty doplněné
    ffillem. Pokrytí je diagnostika vstupních dat, takže se nesmí zlepšovat
    tím, jak s nimi zacházíme.

    Args:
        timestamps: Časy záznamů.
        heart_rate: Tep; ``NaN`` = záznam bez tepu, počítá se jako nepokrytá
            sekunda stejně jako chybějící řádek.

    Returns:
        ``(naměřených_sekund, rozsah_aktivity_v_sekundách)``.
    """
    raw = to_second_grid(timestamps, heart_rate, ffill_limit_s=0)
    return int((~np.isnan(raw)).sum()), int(raw.size)


def longest_gap(grid: np.ndarray) -> int:
    """Nejdelší souvislá díra v sekundách (``NaN`` po přeindexování a ffillu)."""
    holes = np.isnan(grid)
    if not holes.any():
        return 0
    padded = np.concatenate(([False], holes, [False]))
    edges = np.diff(padded.astype(np.int8))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def _to_seconds(ts: np.ndarray) -> np.ndarray:
    """Časy → celé sekundy jako int64, ať už přišly jako datetime nebo čísla."""
    if np.issubdtype(ts.dtype, np.datetime64):
        return ts.astype("datetime64[s]").astype(np.int64)
    if ts.dtype == object:
        return np.array(ts, dtype="datetime64[s]").astype(np.int64)
    return np.rint(np.asarray(ts, dtype=float)).astype(np.int64)


def _ffill(values: np.ndarray, limit: int) -> np.ndarray:
    """
    Doplní ``NaN`` poslední známou hodnotou, ale nejvýš ``limit`` kroků.

    Delší díra zůstává ``NaN`` schválně: po půl minutě bez záznamu už není
    poslední naměřený tep tvrzení o tom, co se dělo.
    """
    if limit <= 0 or values.size == 0:
        return values

    known = ~np.isnan(values)
    if not known.any():
        return values

    # Index posledního známého vzorku pro každou pozici; -1 = ještě žádný.
    idx = np.where(known, np.arange(values.size), -1)
    idx = np.maximum.accumulate(idx)

    out = values.copy()
    gap = np.arange(values.size) - idx
    fillable = (~known) & (idx >= 0) & (gap <= limit)
    out[fillable] = values[idx[fillable]]
    return out


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Klouzavý průměr, který ``NaN`` nešíří přes celou řadu, ale ani nepřehlíží.

    Okno je zarovnané doprava (hodnota na pozici *i* je průměr z posledních
    ``window`` vzorků včetně *i*), jako u ``pandas.rolling``. Okno, ve kterém
    je jediný ``NaN``, vrací ``NaN`` – u tepu je díra v datech chybějící
    informace, ne nula, a průměr z poloviny okna tvrdí něco jiného, než co
    se opravdu naměřilo.

    Args:
        values: Vstupní řada.
        window: Délka okna ve vzorcích. ``window <= 1`` vrací vstup beze změny.

    Returns:
        Pole stejné délky; prvních ``window − 1`` hodnot je ``NaN``.
    """
    if window <= 1:
        return values.astype(float, copy=True)
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out

    filled = np.nan_to_num(values, nan=0.0)
    # cumsum s vloženou nulou → součet okna je rozdíl dvou prvků
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    sums = csum[window:] - csum[:-window]

    nan_csum = np.concatenate(([0], np.cumsum(np.isnan(values).astype(np.int64))))
    nan_counts = nan_csum[window:] - nan_csum[:-window]

    means = sums / window
    means[nan_counts > 0] = np.nan
    out[window - 1 :] = means
    return out
