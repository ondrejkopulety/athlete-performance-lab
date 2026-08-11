"""
rr_extract.py – R-R intervaly z FIT souboru
============================================

R-R intervaly neleží v ``record`` zprávách, ale ve zprávě ``hrv``
(global message number 78). Ta má proměnnou frekvenci – jeden záznam na
tep – takže se nedá zarovnat do vteřinové mřížky a parser, který čte jen
``record``, ji přeskočí.

Pole ``time`` zprávy ``hrv`` je pole až pěti hodnot, každá je jeden R-R
interval v sekundách. Neplatné jsou ``None`` a 65.535 (uint16 maximum =
FIT "invalid" hodnota).

Preferovaný parser je ``fitdecode``; když není nainstalovaný, použije se
``fitparse``. Obě knihovny vracejí stejnou strukturu, liší se jen API.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import RR_MAX_SECONDS, RR_MIN_SECONDS

log = logging.getLogger("physio.rr_extract")

# uint16 maximum se scale 1000 – FIT takhle značí "hodnota chybí".
_FIT_INVALID_RR = 65.535

_ID_PATTERNS = [
    re.compile(r"activity_(\d+)\.fit$", re.IGNORECASE),
    re.compile(r"^(\d+)_ACTIVITY\.fit$", re.IGNORECASE),
    re.compile(r"(\d+)\.fit$", re.IGNORECASE),
]


def _backend() -> str:
    """Vrátí ``"fitdecode"`` nebo ``"fitparse"`` podle toho, co je k dispozici."""
    try:
        import fitdecode  # noqa: F401

        return "fitdecode"
    except ImportError:
        return "fitparse"


def extract_activity_id(file_path: str | Path) -> str:
    """ID aktivity z názvu souboru – stejná pravidla jako ve fit_parser."""
    base = os.path.basename(str(file_path))
    for pat in _ID_PATTERNS:
        m = pat.search(base)
        if m:
            return m.group(1)
    return re.sub(r"\.fit$", "", base, flags=re.IGNORECASE)


@dataclass(frozen=True)
class RrExtraction:
    """
    Výsledek extrakce z jednoho FIT souboru.

    Attributes:
        activity_id: ID odvozené z názvu souboru.
        rr_seconds: Validní R-R intervaly v sekundách, v pořadí záznamu.
        t_seconds: Kumulativní čas tepu od začátku R-R řady (konec intervalu).
        message_counts: Počty všech typů zpráv v souboru – diagnostika.
        n_raw: Kolik hodnot pole ``time`` se celkem přečetlo.
        n_invalid: Kolik z nich bylo mimo platný rozsah nebo None.
        error: Text chyby, pokud se soubor nepodařilo přečíst.
    """

    activity_id: str
    rr_seconds: np.ndarray
    t_seconds: np.ndarray
    message_counts: dict[str, int] = field(default_factory=dict)
    n_raw: int = 0
    n_invalid: int = 0
    error: str | None = None

    @property
    def has_hrv_messages(self) -> bool:
        """True, když soubor obsahuje aspoň jednu ``hrv`` zprávu."""
        return self.message_counts.get("hrv", 0) > 0

    @property
    def beat_count(self) -> int:
        """Počet validních R-R intervalů."""
        return int(len(self.rr_seconds))

    @property
    def duration_seconds(self) -> float:
        """Délka R-R řady v sekundách (součet intervalů)."""
        return float(self.t_seconds[-1]) if len(self.t_seconds) else 0.0


def _valid_rr(values) -> tuple[list[float], int, int]:
    """
    Vytáhne platné R-R hodnoty z pole ``time`` jedné ``hrv`` zprávy.

    Returns:
        (platné hodnoty, počet přečtených, počet neplatných)
    """
    if values is None:
        return [], 0, 0
    if not isinstance(values, (list, tuple)):
        values = [values]

    good: list[float] = []
    n_raw = 0
    n_invalid = 0
    for v in values:
        if v is None:
            # None je výplň pole na pevnou délku 5, ne chyba měření –
            # do statistiky neplatných se nepočítá.
            continue
        n_raw += 1
        try:
            fv = float(v)
        except (TypeError, ValueError):
            n_invalid += 1
            continue
        if fv == _FIT_INVALID_RR or not (RR_MIN_SECONDS <= fv <= RR_MAX_SECONDS):
            n_invalid += 1
            continue
        good.append(fv)
    return good, n_raw, n_invalid


def scan_fit_messages(file_path: str | Path) -> dict[str, int]:
    """
    Spočítá typy zpráv ve FIT souboru.

    Diagnostika, která se pouští jako první – ať je hned vidět, jestli
    ``hrv`` v souboru vůbec je, místo hádání z prázdného výstupu.

    Args:
        file_path: Cesta k .fit souboru.

    Returns:
        Slovník {název zprávy: počet}, seřazený sestupně podle počtu.

    Raises:
        OSError: Když soubor nejde otevřít nebo přečíst.
    """
    counts: Counter[str] = Counter()

    if _backend() == "fitdecode":
        import fitdecode

        with fitdecode.FitReader(str(file_path)) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    counts[frame.name] += 1
    else:
        from fitparse import FitFile

        for msg in FitFile(str(file_path)).get_messages():
            counts[msg.name] += 1

    return dict(counts.most_common())


def _iter_hrv_time_fields(file_path: str | Path):
    """Yielduje hodnotu pole ``time`` z každé ``hrv`` zprávy."""
    if _backend() == "fitdecode":
        import fitdecode

        with fitdecode.FitReader(str(file_path)) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "hrv":
                    if frame.has_field("time"):
                        yield frame.get_value("time")
    else:
        from fitparse import FitFile

        for msg in FitFile(str(file_path)).get_messages("hrv"):
            yield msg.get_values().get("time")


def extract_rr(file_path: str | Path) -> RrExtraction:
    """
    Vytáhne R-R intervaly ze ``hrv`` zpráv FIT souboru.

    Když soubor ``hrv`` zprávy neobsahuje, vrátí prázdné pole s vyplněným
    ``message_counts`` – volající pozná chybějící data podle
    ``has_hrv_messages`` a nemá důvod si domýšlet náhradní odhad.

    Args:
        file_path: Cesta k .fit souboru.

    Returns:
        RrExtraction; při chybě čtení má vyplněný ``error`` a prázdná pole.
    """
    activity_id = extract_activity_id(file_path)
    empty = np.empty(0, dtype=float)

    try:
        counts = scan_fit_messages(file_path)
    except Exception as exc:  # noqa: BLE001 – poškozený FIT nesmí shodit dávku
        log.error("[%s] FIT nelze přečíst: %s", activity_id, exc)
        return RrExtraction(activity_id, empty, empty, error=str(exc))

    if counts.get("hrv", 0) == 0:
        log.info("[%s] Soubor neobsahuje 'hrv' zprávy – R-R intervaly nejsou k dispozici.",
                 activity_id)
        return RrExtraction(activity_id, empty, empty, message_counts=counts)

    rr: list[float] = []
    n_raw = 0
    n_invalid = 0
    try:
        for value in _iter_hrv_time_fields(file_path):
            good, raw, invalid = _valid_rr(value)
            rr.extend(good)
            n_raw += raw
            n_invalid += invalid
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] Chyba při čtení 'hrv' zpráv: %s", activity_id, exc)
        return RrExtraction(activity_id, empty, empty, message_counts=counts, error=str(exc))

    rr_arr = np.asarray(rr, dtype=float)
    t_arr = np.cumsum(rr_arr) if len(rr_arr) else empty

    if n_invalid:
        log.debug("[%s] %d z %d hodnot pole 'time' bylo mimo rozsah %.1f–%.1f s.",
                  activity_id, n_invalid, n_raw, RR_MIN_SECONDS, RR_MAX_SECONDS)

    return RrExtraction(
        activity_id=activity_id,
        rr_seconds=rr_arr,
        t_seconds=t_arr,
        message_counts=counts,
        n_raw=n_raw,
        n_invalid=n_invalid,
    )


def write_rr_csv(extraction: RrExtraction, out_dir: str | Path) -> Path:
    """
    Uloží R-R řadu jako ``{activity_id}_rr.csv``.

    Samostatný soubor, ne sloupec ve vteřinovém CSV – R-R má jeden řádek
    na tep, takže se do vteřinové mřížky nedá zarovnat.

    Args:
        extraction: Výsledek ``extract_rr``.
        out_dir: Adresář, kam soubor uložit (vytvoří se, když neexistuje).

    Returns:
        Cesta k zapsanému souboru.
    """
    out_path = Path(out_dir) / f"{extraction.activity_id}_rr.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["beat_index", "t_seconds", "rr_seconds"])
        for i, (t, rr) in enumerate(zip(extraction.t_seconds, extraction.rr_seconds, strict=True)):
            writer.writerow([i, f"{t:.3f}", f"{rr:.3f}"])

    return out_path


def read_rr_csv(path: str | Path) -> np.ndarray:
    """Načte sloupec ``rr_seconds`` z dřív uloženého ``_rr.csv`` (cache)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return np.asarray([float(row["rr_seconds"]) for row in reader], dtype=float)
