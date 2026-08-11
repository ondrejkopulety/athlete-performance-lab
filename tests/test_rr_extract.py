"""
Testy extrakce R-R z FIT souborů.

Testy nad reálnými soubory se přeskočí, když data/fit chybí – ať repo
projde i na stroji bez stažených aktivit.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from config.settings import FIT_DIR
from src.physio.rr_extract import (
    RrExtraction,
    _valid_rr,
    extract_activity_id,
    read_rr_csv,
    write_rr_csv,
)


def test_valid_rr_odfiltruje_neplatne_hodnoty():
    """None je výplň pole, 65.535 je FIT 'invalid', mimo rozsah je chyba přenosu."""
    good, n_raw, n_invalid = _valid_rr((0.623, 0.630, 65.535, None, None))

    assert good == [0.623, 0.630]
    assert n_raw == 3          # None se nepočítá jako přečtená hodnota
    assert n_invalid == 1


def test_valid_rr_meze_rozsahu():
    good, _, n_invalid = _valid_rr([0.19, 0.2, 3.0, 3.01])

    assert good == [0.2, 3.0]
    assert n_invalid == 2


def test_valid_rr_skalarni_hodnota():
    """Některé soubory mají v poli 'time' jednu hodnotu, ne pole."""
    good, n_raw, _ = _valid_rr(0.85)

    assert good == [0.85]
    assert n_raw == 1


def test_valid_rr_prazdny_vstup():
    assert _valid_rr(None) == ([], 0, 0)
    assert _valid_rr([]) == ([], 0, 0)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("activity_19652688163.fit", "19652688163"),
        ("21896943450_ACTIVITY.fit", "21896943450"),
        ("12345.FIT", "12345"),
        ("nejaky_export.fit", "nejaky_export"),
    ],
)
def test_extract_activity_id(filename, expected):
    assert extract_activity_id(f"/data/fit/{filename}") == expected


def test_write_a_read_rr_csv(tmp_path):
    """Zápis a zpětné načtení musí dát tutéž řadu."""
    rr = np.array([0.623, 0.630, 0.637])
    extraction = RrExtraction(
        activity_id="42",
        rr_seconds=rr,
        t_seconds=np.cumsum(rr),
        message_counts={"hrv": 2, "record": 10},
        n_raw=3,
    )

    path = write_rr_csv(extraction, tmp_path)

    assert path.name == "42_rr.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["beat_index"] for r in rows] == ["0", "1", "2"]
    assert float(rows[2]["t_seconds"]) == pytest.approx(1.890, abs=1e-3)
    assert read_rr_csv(path) == pytest.approx(rr, abs=1e-3)


def test_extrakce_bez_hrv_zprav_nic_nefabrikuje():
    """
    Soubor bez 'hrv' zpráv musí vrátit prázdno a říct to – ne se pokusit
    o náhradní odhad z 'record' zpráv.
    """
    candidates = sorted(FIT_DIR.glob("*.fit"))
    if not candidates:
        pytest.skip("data/fit je prázdná")

    from src.physio.rr_extract import extract_rr, scan_fit_messages

    without_hrv = next(
        (p for p in candidates[:20] if "hrv" not in scan_fit_messages(p)), None
    )
    if without_hrv is None:
        pytest.skip("mezi prvními soubory není žádný bez hrv zpráv")

    result = extract_rr(without_hrv)

    assert result.has_hrv_messages is False
    assert result.beat_count == 0
    assert result.error is None
    assert result.message_counts, "diagnostika typů zpráv musí být vyplněná"
