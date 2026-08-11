"""
hr_curve.py – tepová křivka
============================

Maximální průměrný tep za 5 s až 60 min. Obdoba výkonové křivky, jen z tepu:
jeden řádek na délku okna, hodnota je nejlepší klouzavý průměr přes celou
aktivitu.

Na LTHR nezávislá úplně – spočítá se jednou a platí, dokud se nezmění
samotná data. Proto tu není žádný práh ani zóna.

**Bez vyhlazení.** Průměrování je už v definici metriky; vyhladit tep před
tím, než se z něj počítá průměr, znamená průměrovat dvakrát.

**Pauzy zůstávají v datech** (na rozdíl od hr_blocks.py). Tep při zastavení
neklesá skokově a je součástí zátěže. Okno, které nemá plné pokrytí daty,
ale nevznikne – viz ``max_mean_curve``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.physio.hr_stream import moving_average


def max_mean_curve(
    heart_rate: np.ndarray,
    durations_s: Sequence[int],
) -> dict[int, float]:
    """
    Nejlepší klouzavý průměr tepu pro každou délku okna.

    Okno musí být **plně pokryté daty**. Díra po autopauze uvnitř okna
    znamená, že se hodnota nespočítá – ne že se spočítá z toho, co zbylo.
    Průměr z 15 minut prezentovaný jako dvacetiminutový je horší než chybějící
    řádek, protože se tváří srovnatelně s ostatními.

    Když je aktivita kratší než okno, klíč v návratové hodnotě **nevznikne**
    – ne nula, ne ``None``. "Nemám dost dlouhou jízdu" a "jel jsem hodinu na
    nule" jsou různá tvrzení a v tabulce se nesmí slít.

    Args:
        heart_rate: Tep na spojité sekundové mřížce (``hr_stream.to_second_grid``);
            ``NaN`` = díra.
        durations_s: Délky oken v sekundách.

    Returns:
        ``{délka_okna_s: max_průměrný_tep}``, jen pro okna, která vyšla.
    """
    hr = np.asarray(heart_rate, dtype=float)
    out: dict[int, float] = {}
    if hr.size == 0:
        return out

    for duration in sorted(set(int(d) for d in durations_s)):
        if duration <= 0 or hr.size < duration:
            continue
        means = moving_average(hr, duration)
        if np.all(np.isnan(means)):
            continue
        out[duration] = round(float(np.nanmax(means)), 1)
    return out


def curve_rows(
    activity_id: str,
    heart_rate: np.ndarray,
    durations_s: Sequence[int],
    calc_version: int,
) -> list[dict]:
    """
    Tepová křivka jako řádky pro ``activity_hr_curve``.

    Args:
        activity_id: ID aktivity.
        heart_rate: Tep na sekundové mřížce.
        durations_s: Délky oken v sekundách.
        calc_version: Verze výpočtu (``settings.HR_CURVE_VERSION``).

    Returns:
        Řádky připravené k upsertu; prázdný seznam, když nevyšlo žádné okno.
    """
    curve = max_mean_curve(heart_rate, durations_s)
    return [
        {
            "activity_id": activity_id,
            "duration_s": duration,
            "max_mean_hr": value,
            "calc_version": calc_version,
        }
        for duration, value in sorted(curve.items())
    ]
