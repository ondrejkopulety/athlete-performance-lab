"""
hr_panels.py – podklad pro panely tepové křivky a souvislých bloků
===================================================================

Čte předpočítané řádky (``activity_hr_curve``, ``activity_hr_blocks``,
``activity_hr_coverage``) a skládá z nich čísla, která jde vykreslit. **Nic
se tu nepočítá ze sekundových dat.** Když na něco data nejsou, vrací se
``None`` a UI ukazuje pomlčku – chybějící okno není nula.

Tři pravidla, na kterých modul stojí:

  chybějící ≠ nula   Aktivita, která nemá řádek pro ``duration_s = 3600``,
                     nedostane nulu ani se nepřeskočí potichu. Vrací se bod
                     s ``max_mean_hr = None`` a agregace přes období, ve
                     kterém takové okno nemá žádná jízda, vrací ``None``.
                     "Nikdo nejel hodinu naplno" a "jel jsem hodinu na nule"
                     jsou různá tvrzení.

  zóna je lookup     Bloky jsou uložené na mřížce absolutních prahů. Zóna se
                     spočítá z LTHR a zaokrouhlí na nejbližší práh mřížky –
                     mění se tím dotaz, ne data. Změna LTHR proto nikdy
                     nespustí přepočet.

  varuje se na       Nízká hustota vzorků (Smart Recording) není ztráta dat:
  pokrytí            po doplnění mezer je mřížka plná a všechna okna vyjdou.
                     Varování visí výhradně na pokrytí po ffillu.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from config.settings import (
    HR_BLOCK_THRESHOLDS_BPM,
    HR_COVERAGE_WARN_PCT,
    HR_SEGMENT_BUCKET_LABELS,
    HR_ZONE_LTHR_RATIO,
)


# ═══════════════════════════════════════════════════════════════════════════
# Zóna jako lookup nad mřížkou prahů
# ═══════════════════════════════════════════════════════════════════════════

def zone_threshold_bpm(
    lthr_bpm: float,
    zone: str,
    grid: Sequence[int] = tuple(HR_BLOCK_THRESHOLDS_BPM),
) -> int:
    """
    Práh z mřížky, který nejlíp odpovídá spodní hranici zóny při daném LTHR.

    Args:
        lthr_bpm: Prahový tep.
        zone: ``"Z2"`` … ``"Z5"``.
        grid: Mřížka uložených prahů (``settings.HR_BLOCK_THRESHOLDS_BPM``).

    Returns:
        Nejbližší práh z mřížky. Mimo rozsah mřížky se ořízne na její kraj –
        pro LTHR nad 185 by jinak vznikl dotaz na neexistující řádek.

    Raises:
        KeyError: Když zóna není v ``settings.HR_ZONE_LTHR_RATIO``.
    """
    target = lthr_bpm * HR_ZONE_LTHR_RATIO[zone]
    return min(grid, key=lambda t: (abs(t - target), t))


# ═══════════════════════════════════════════════════════════════════════════
# Pokrytí → verdikt, ne tři procenta
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Coverage:
    """
    Pokrytí jedné jízdy i s větou, která říká, co z toho plyne.

    Tři čísla zůstávají k dispozici, ale ven jde hlavně ``ok`` a ``note``:
    uživatel potřebuje odpověď na "můžu tomuhle číslu věřit", ne tři
    procenta k interpretaci.
    """

    span_s: int
    measured_s: int
    usable_s: int
    longest_gap_s: int
    max_curve_duration_s: int | None

    @property
    def coverage_pct(self) -> float | None:
        return 100.0 * self.usable_s / self.span_s if self.span_s else None

    @property
    def sample_density_pct(self) -> float | None:
        return 100.0 * self.measured_s / self.span_s if self.span_s else None

    @property
    def ok(self) -> bool:
        """Stojí čísla téhle jízdy na dostatečně úplných datech?"""
        pct = self.coverage_pct
        return pct is not None and pct >= HR_COVERAGE_WARN_PCT

    @property
    def note(self) -> str | None:
        """
        Vysvětlení pro detail jízdy; ``None``, když je pokrytí v pořádku.

        Věta se skládá tady, ne ve frontendu: "křivka nemá okna delší než
        45 min" je ``max_curve_duration_s``, které by si jinak musel klient
        dohledávat druhým dotazem.
        """
        pct = self.coverage_pct
        if self.ok:
            return None
        if pct is None:
            # Nulový rozsah – odznak visí na `ok`, takže i tenhle stav musí
            # mít vysvětlení, jinak by karta varovala beze slova.
            return "Tepová data k téhle jízdě nejsou."

        parts = [
            f"Naměřeno {pct:.0f} % času jízdy, nejdelší výpadek "
            f"{_human_duration(self.longest_gap_s)}."
        ]
        consequences = ["metriky souvislých bloků jsou podhodnocené"]
        if self.max_curve_duration_s:
            consequences.append(
                "tepová křivka nemá okna delší než "
                f"{_human_duration(self.max_curve_duration_s)}"
            )
        else:
            consequences.append("tepová křivka z ní nevyšla vůbec")
        parts.append(_capitalize(", ".join(consequences)) + ".")

        if self.sample_density_pct is not None and self.sample_density_pct < 50:
            # Aby se řídký zápis nespletl s výpadkem: sám o sobě problém není.
            parts.append(
                f"Vzorek se zapisuje řídce ({self.sample_density_pct:.0f} % sekund), "
                "to ale samo o sobě data neztrácí – rozhoduje pokrytí po doplnění mezer."
            )
        return " ".join(parts)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "pct": _round(self.coverage_pct),
            "density": _round(self.sample_density_pct),
            "gap": self.longest_gap_s,
            "note": self.note,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Tepová křivka
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CurvePoint:
    """Jeden bod křivky za období i s tím, ze které jízdy pochází."""

    duration_s: int
    max_mean_hr: float | None
    activity_id: str | None
    date: date | None
    label: str | None

    def as_dict(self) -> dict:
        return {
            "d": self.duration_s,
            "hr": self.max_mean_hr,
            "activity_id": self.activity_id,
            "date": self.date.isoformat() if self.date else None,
            "label": self.label,
        }


def curve_points(
    rows: Iterable[dict],
    durations_s: Sequence[int],
) -> list[CurvePoint]:
    """
    Nejlepší hodnota pro každé okno za období + jízda, ze které je.

    Args:
        rows: Řádky křivky s klíči ``duration_s``, ``max_mean_hr``,
            ``activity_id``, ``date``, ``label``.
        durations_s: Všechna okna, která má křivka mít.

    Returns:
        Bod pro **každé** zadané okno, seřazeno vzestupně. Okno, které v
        období nemá žádnou jízdu, dostane ``max_mean_hr=None`` – ne nulu a
        ne vynechaný bod, aby čára v grafu mohla skončit a tabulka ukázat
        pomlčku.
    """
    best: dict[int, dict] = {}
    for row in rows:
        value = row.get("max_mean_hr")
        if value is None:
            continue
        duration = int(row["duration_s"])
        current = best.get(duration)
        if current is None or float(value) > float(current["max_mean_hr"]):
            best[duration] = row

    points = []
    for duration in sorted(durations_s):
        row = best.get(duration)
        points.append(
            CurvePoint(
                duration_s=duration,
                max_mean_hr=round(float(row["max_mean_hr"]), 1) if row else None,
                activity_id=row["activity_id"] if row else None,
                date=row.get("date") if row else None,
                label=row.get("label") if row else None,
            )
        )
    return points


def last_max_effort(
    points: Sequence[CurvePoint],
    reference_duration_s: int,
    today: date,
    stale_days: int = 60,
) -> dict | None:
    """
    Kdy naposledy padlo maximum – datum jízdy, která drží referenční bod.

    Ne "poslední tvrdá jízda", ale poslední jízda, která do křivky opravdu
    přispěla. To je informace, kvůli které panel hlavně je: dvacetiminutový
    bod z jízdy staré čtyři měsíce znamená, že se od té doby naplno nešlo.

    Args:
        points: Body křivky z ``curve_points``.
        reference_duration_s: Okno, ze kterého se čte (typicky 1200 s).
        today: Dnešek – předává se, ať je funkce testovatelná.
        stale_days: Po kolika dnech se hodnota zvýrazní.

    Returns:
        ``None``, když referenční okno v období nevyšlo.
    """
    point = next(
        (p for p in points if p.duration_s == reference_duration_s and p.date), None
    )
    if point is None or point.date is None:
        return None
    days = (today - point.date).days
    return {
        "date": point.date.isoformat(),
        "activity_id": point.activity_id,
        "label": point.label,
        "duration_s": reference_duration_s,
        "days_ago": days,
        "stale": days > stale_days,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Souvislé bloky
# ═══════════════════════════════════════════════════════════════════════════

def block_period_summary(rows: Sequence[dict]) -> dict:
    """
    Souhrn bloků za období – nejdelší blok, histogram délek, součty.

    Args:
        rows: Řádky ``activity_hr_blocks`` jednoho prahu a jedné tolerance,
            doplněné o ``date`` a ``label`` aktivity.

    Returns:
        ``longest_block_s`` je ``None``, když v období není ani jedna jízda –
        nikoli nula. Nula znamená "jel jsem, ale nad práh se nedostal", a to
        je jiné tvrzení.
    """
    if not rows:
        return {
            "longest_block_s": None,
            "longest_block": None,
            "hist": _empty_hist(),
            "totals": {"segment_count": 0, "total_time_s": 0, "time_in_long_blocks_s": 0},
        }

    top = max(rows, key=lambda r: int(r["longest_block_s"] or 0))
    counts = [0] * len(HR_SEGMENT_BUCKET_LABELS)
    seconds = [0] * len(HR_SEGMENT_BUCKET_LABELS)
    for row in rows:
        for i, value in enumerate(row.get("segment_hist_counts") or []):
            if i < len(counts):
                counts[i] += int(value)
        for i, value in enumerate(row.get("segment_hist_seconds") or []):
            if i < len(seconds):
                seconds[i] += int(value)

    return {
        "longest_block_s": int(top["longest_block_s"] or 0),
        "longest_block": {
            "activity_id": top["activity_id"],
            "date": top["date"].isoformat() if top.get("date") else None,
            "label": top.get("label"),
        },
        "hist": [
            {"bucket": label, "count": c, "seconds": s}
            for label, c, s in zip(HR_SEGMENT_BUCKET_LABELS, counts, seconds)
        ],
        "totals": {
            "segment_count": sum(int(r["segment_count"] or 0) for r in rows),
            "total_time_s": sum(int(r["total_time_s"] or 0) for r in rows),
            "time_in_long_blocks_s": sum(
                int(r["time_in_long_blocks_s"] or 0) for r in rows
            ),
        },
    }


def unintentional_z3_s(
    total_time_s: dict[int, int],
    long_blocks_s: dict[int, int],
    z3_lo_bpm: int,
    z3_hi_bpm: int,
) -> int:
    """
    Nezáměrná Z3: čas v Z3 strávený v úsecích kratších než "dlouhý blok".

    Z3 v souvislém bloku je sweet spot trénink, ne odpad. Odpad je Z3, do
    které se spadne kvůli kopci. Proto se od času v Z3 odečítá čas strávený
    v dlouhých souvislých úsecích.

    Rozdíl **dvou** prahů, ne jen odečtení dlouhých bloků nad spodní hranicí:
    dvacetiminutový Z4 interval leží celý nad spodní hranicí Z3, takže by se
    započítal jako "záměrná Z3" v objemu, který v Z3 nikdy nebyl, a metrika
    by šla do záporu.

    Args:
        total_time_s: Čas nad prahem podle prahu (``threshold → sekundy``).
        long_blocks_s: Čas v dlouhých úsecích podle prahu.
        z3_lo_bpm: Práh mřížky odpovídající spodní hranici Z3.
        z3_hi_bpm: Práh mřížky odpovídající spodní hranici Z4.

    Returns:
        Sekundy nezáměrné Z3, nikdy záporné. Nula znamená, že veškerá Z3
        byla součástí dlouhého úsilí – tedy záměrná.
    """
    def short(threshold: int) -> int:
        return max(0, int(total_time_s.get(threshold, 0)) - int(long_blocks_s.get(threshold, 0)))

    return max(0, short(z3_lo_bpm) - short(z3_hi_bpm))


# ═══════════════════════════════════════════════════════════════════════════
# Pomocné
# ═══════════════════════════════════════════════════════════════════════════

def _empty_hist() -> list[dict]:
    return [
        {"bucket": label, "count": 0, "seconds": 0}
        for label in HR_SEGMENT_BUCKET_LABELS
    ]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _human_duration(seconds: int) -> str:
    """Sekundy na text do věty: ``45 s``, ``26 min``, ``1 h 12 min``."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h" if rest == 0 else f"{hours} h {rest} min"
