"""
Testy filtru artefaktů a rozlišení skutečného R-R od dopočítané křivky.

Běží bez FIT souborů i bez databáze – vstupem jsou syntetické řady, u
kterých je správná odpověď známá předem.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physio.quality import assess_rr_authenticity
from src.physio.rr_clean import clean_rr


def _beat_to_beat_series(n: int = 20_000, seed: int = 0) -> np.ndarray:
    """
    Syntetická řada, která se chová jako skutečné R-R z hrudního pásu.

    Pomalý drift tepu (rozjezd → zátěž) plus variabilita mezi tepy o
    rozptylu odpovídajícím RMSSD ~40 ms, kvantizovaná na 1 ms jako ve FIT.
    """
    rng = np.random.default_rng(seed)
    drift = 0.60 + np.cumsum(rng.normal(0, 0.002, n)) * 0.1
    rr = np.clip(drift + rng.normal(0, 0.030, n), 0.35, 1.2)
    return np.round(rr, 3)


def _synthetic_series(n: int = 20_000, seed: int = 1) -> np.ndarray:
    """
    Řada, jaká je v měřených FIT souborech: hladká tepová křivka převedená
    na intervaly a kvantizovaná na hrubou mřížku. Mezi tepy se skoro nemění.
    """
    rng = np.random.default_rng(seed)
    hr = 95 + 45 * (1 - np.exp(-np.arange(n) / (n / 3))) + rng.normal(0, 1.5, n).cumsum() * 0.01
    rr = 60.0 / np.round(hr)          # reciproká mřížka celočíselné frekvence
    return np.round(rr, 3)


# ── filtr artefaktů ────────────────────────────────────────────────────────


def test_clean_rr_zachova_cistou_radu():
    """Řada bez výpadků nesmí přijít o víc než zlomek tepů."""
    rr = _beat_to_beat_series(5_000)
    result = clean_rr(rr)

    assert result.artifact_pct < 0.02
    assert result.reliable is True
    assert result.beat_count > 4_900


def test_clean_rr_vyhodi_vlozene_vypadky():
    """Vynechaný tep (dvojnásobný interval) i zdvojený tep musí zmizet."""
    rr = _beat_to_beat_series(2_000).copy()
    dropouts = [100, 500, 900, 1_300, 1_700]
    for i in dropouts:
        rr[i] = rr[i] * 2.0      # vynechaný tep
        rr[i + 1] = rr[i + 1] * 0.45   # zdvojený tep

    result = clean_rr(rr)

    for i in dropouts:
        assert not result.keep_mask[i], f"výpadek na indexu {i} prošel filtrem"
        assert not result.keep_mask[i + 1], f"zdvojený tep na indexu {i + 1} prošel filtrem"
    assert result.beat_count == len(rr) - result.keep_mask.size + result.keep_mask.sum()


def test_clean_rr_oznaci_nespolehlivou_radu():
    """Nad 10 % artefaktů je celá aktivita nespolehlivá."""
    rng = np.random.default_rng(7)
    rr = _beat_to_beat_series(1_000).copy()
    idx = rng.choice(len(rr), size=150, replace=False)
    rr[idx] *= 2.0

    result = clean_rr(rr)

    assert result.artifact_pct > 0.10
    assert result.reliable is False


def test_clean_rr_tolerance_je_konfigurovatelna():
    """Přísnější tolerance musí vyhodit aspoň tolik tepů co volnější."""
    rr = _beat_to_beat_series(2_000)

    prisna = clean_rr(rr, tolerance=0.05)
    volna = clean_rr(rr, tolerance=0.40)

    assert prisna.artifact_pct >= volna.artifact_pct


def test_clean_rr_prazdny_vstup():
    """Prázdný vstup nesmí spadnout ani se tvářit jako platný výsledek."""
    result = clean_rr([])

    assert result.beat_count == 0
    assert result.artifact_pct == 0.0
    assert result.reliable is False
    assert result.keep_mask.size == 0


def test_clean_rr_kratsi_nez_okno():
    """Řada kratší než okno mediánu projde beze změny, ale jako nespolehlivá."""
    result = clean_rr([0.8, 0.81, 0.79, 0.8])

    assert result.beat_count == 4
    assert result.reliable is False


def test_clean_rr_odmitne_nesmyslne_parametry():
    with pytest.raises(ValueError):
        clean_rr([0.8] * 100, tolerance=0.0)
    with pytest.raises(ValueError):
        clean_rr([0.8] * 100, median_window=2)


# ── pravost R-R ────────────────────────────────────────────────────────────


def test_skutecne_rr_projde_jako_beat_to_beat():
    result = assess_rr_authenticity(_beat_to_beat_series())

    assert result.verdict == "beat_to_beat"
    assert result.usable is True
    assert result.zero_diff_pct < 0.05
    assert result.rmssd_ms > 20


def test_dopocitana_krivka_je_rozpoznana():
    """Řada odvozená z celočíselné tepové frekvence nesmí projít."""
    result = assess_rr_authenticity(_synthetic_series())

    assert result.verdict == "synthetic"
    assert result.usable is False
    assert result.zero_diff_pct > 0.30
    assert "nulov" in result.reason or "hodnot" in result.reason


def test_kratka_rada_nedostane_verdikt():
    """Bez dostatku tepů se nerozhoduje – 'unknown', ne falešné 'ok'."""
    result = assess_rr_authenticity(_beat_to_beat_series(50))

    assert result.verdict == "unknown"
    assert result.usable is False
    assert "50 tepů" in result.reason


def test_prazdny_vstup_pravosti():
    result = assess_rr_authenticity([])

    assert result.verdict == "unknown"
    assert result.usable is False


def test_konstantni_rada_je_synteticka():
    """Naprosto neměnný interval je krajní případ dopočítané křivky."""
    result = assess_rr_authenticity(np.full(1_000, 0.75))

    assert result.verdict == "synthetic"
    assert result.zero_diff_pct == pytest.approx(1.0)
