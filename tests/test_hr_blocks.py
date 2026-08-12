"""
Testy segmentace souvislých bloků nad prahem a tepové křivky.

Běží bez databáze i bez FIT souborů – vstupem jsou syntetické signály,
u kterých je správná odpověď známá předem. Obrácené znaménko nebo posun
o jednu sekundu je na nich vidět okamžitě, na reálné jízdě ne.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physio.hr_blocks import find_segments, segment_histogram, summarize_segments
from src.physio.hr_curve import max_mean_curve
from src.physio.hr_stream import to_second_grid

THRESHOLD = 170.0


def _lengths(segments: np.ndarray) -> list[int]:
    """Délky úseků v sekundách."""
    return [int(e - s + 1) for s, e in segments]


# ═══════════════════════════════════════════════════════════════════════════
# Segmentace – obdélníkové signály
# ═══════════════════════════════════════════════════════════════════════════

def test_obdelnik_jeden_usek():
    """300 s nad prahem, pak 300 s pod → jeden úsek dlouhý 300 s."""
    hr = np.concatenate([np.full(300, 175.0), np.full(300, 150.0)])

    segments = find_segments(hr, THRESHOLD)

    assert _lengths(segments) == [300]
    assert segments[0][0] == 0


def test_prah_je_inkluzivni():
    """Tep přesně na prahu je nad prahem, ne pod ním."""
    hr = np.full(100, THRESHOLD)

    assert _lengths(find_segments(hr, THRESHOLD)) == [100]
    assert _lengths(find_segments(hr, THRESHOLD + 0.5)) == []


def test_jednosekundovy_propad_tolerance_rozhoduje():
    """
    Jedna sekunda mírně pod prahem uprostřed dlouhého úsilí.

    Při toleranci 0 rozdělí úsilí na dva úseky, při 15 s se přemostí do
    jednoho – právě kvůli tomuhle přemostění existuje.
    """
    hr = np.full(601, 175.0)
    hr[300] = 168.0            # 2 bpm pod prahem

    strict = find_segments(hr, THRESHOLD, bridge_tolerance_s=0)
    bridged = find_segments(hr, THRESHOLD, bridge_tolerance_s=15)

    assert _lengths(strict) == [300, 300]
    assert _lengths(bridged) == [601]


def test_propad_30s_nepremostuje_ani_pri_toleranci():
    """Propad delší než tolerance zůstává předělem u obou variant."""
    hr = np.full(630, 175.0)
    hr[300:330] = 168.0        # 30 s pod prahem, ale jen mělce

    for tolerance in (0, 15):
        segments = find_segments(hr, THRESHOLD, bridge_tolerance_s=tolerance)
        assert _lengths(segments) == [300, 300], f"tolerance={tolerance}"


def test_hluboky_propad_se_nepremostuje_ani_kdyz_je_kratky():
    """
    10 s hluboko pod prahem (−20 bpm) → dva úseky i při toleranci 15.

    Tohle je kontrola podmínky na hloubku. Bez ní by se přes sjezd spojila
    dvě opravdu oddělená úsilí a "nejdelší blok" by měřil dobu na kole.
    """
    hr = np.full(610, 175.0)
    hr[300:310] = THRESHOLD - 20.0

    segments = find_segments(hr, THRESHOLD, bridge_tolerance_s=15)

    assert _lengths(segments) == [300, 300]


def test_melky_propad_se_premosti_hloubka_na_hranici():
    """Přesně na povolené hloubce se ještě přemosťuje, o bpm hloub už ne."""
    hr = np.full(610, 175.0)
    hr[300:310] = THRESHOLD - 5.0

    assert _lengths(find_segments(hr, THRESHOLD, 15, bridge_max_drop_bpm=5.0)) == [610]
    assert _lengths(find_segments(hr, THRESHOLD, 15, bridge_max_drop_bpm=4.0)) == [300, 300]


def test_pauza_deli_usek_i_kdyz_tep_zustal_nad_prahem():
    """
    Díra v datech je přerušení úsilí, ne jeho součást.

    Rozdíl proti hr_curve.py, kde pauzy v datech zůstávají – v docstringu
    obou modulů je to popsané, tady je to zafixované testem.
    """
    hr = np.full(610, 175.0)
    hr[300:305] = np.nan       # 5 s pauza, kratší než tolerance

    segments = find_segments(hr, THRESHOLD, bridge_tolerance_s=15)

    assert _lengths(segments) == [300, 305]


# ═══════════════════════════════════════════════════════════════════════════
# Vyhlazení
# ═══════════════════════════════════════════════════════════════════════════

def test_vyhlazeni_polyka_sum_na_hranici_prahu():
    """
    Tep kmitající o ±2 bpm kolem prahu je jedno úsilí, ne 150 úseků.

    Bez vyhlazení vyrobí každý překmit vlastní úsek – přesně ten artefakt,
    kvůli kterému má jízda 26. 7. syrově medián úseku 10 sekund.
    """
    rng = np.random.default_rng(0)
    hr = THRESHOLD + 3.0 + rng.normal(0, 2.0, 600)

    raw = find_segments(hr, THRESHOLD, smooth_s=1)
    smoothed = find_segments(hr, THRESHOLD, smooth_s=10)

    assert len(raw) > 20
    assert _lengths(smoothed) == [600]


def test_vyhlazeni_nemeni_delku_dlouheho_bloku():
    """
    Vyhlazení maže šum, ne práci.

    Na ostré hraně (30 bpm skok za sekundu, v reálném tepu neexistuje) se
    hranice bloku posune o pár sekund dovnitř – to je vlastnost průměrování.
    Podstatné je, že blok zůstane jeden a zkrátí se řádově o jednotky sekund,
    ne o desítky.
    """
    hr = np.concatenate([np.full(200, 150.0), np.full(300, 180.0), np.full(200, 150.0)])

    segments = find_segments(hr, THRESHOLD, smooth_s=10)

    assert len(segments) == 1
    assert _lengths(segments)[0] == pytest.approx(300, abs=5)


# ═══════════════════════════════════════════════════════════════════════════
# Souhrn
# ═══════════════════════════════════════════════════════════════════════════

def test_souhrn_pocita_dlouhe_useky_ostre():
    """Úsek přesně 180 s se ještě nepočítá jako dlouhý, 181 s ano."""
    segments = np.array([[0, 179], [1000, 1180], [2000, 2999]])

    summary = summarize_segments(segments, long_block_s=180)

    assert summary.segment_count == 3
    assert summary.longest_block_s == 1000
    assert summary.total_time_s == 180 + 181 + 1000
    assert summary.time_in_long_blocks_s == 181 + 1000
    assert summary.median_segment_s == 181.0


def test_souhrn_prazdneho_vstupu():
    """Nic nad prahem → nuly a medián NULL, ne pád."""
    summary = summarize_segments(np.empty((0, 2)), long_block_s=180)

    assert summary.segment_count == 0
    assert summary.total_time_s == 0
    assert summary.longest_block_s == 0
    assert summary.median_segment_s is None


def test_premosteny_propad_se_pocita_do_celkoveho_casu():
    """
    Přemostěný propad je součástí bloku, takže i jeho času.

    Surový čas nad prahem zůstává dostupný v řádku s tolerancí 0.
    """
    hr = np.full(610, 175.0)
    hr[300:310] = 168.0

    strict = summarize_segments(find_segments(hr, THRESHOLD, 0), 180)
    bridged = summarize_segments(find_segments(hr, THRESHOLD, 15), 180)

    assert strict.total_time_s == 600
    assert bridged.total_time_s == 610


# ═══════════════════════════════════════════════════════════════════════════
# Okrajové vstupy
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("hr", [np.empty(0), np.array([175.0]), np.array([np.nan])])
def test_okrajove_vstupy_nepadaji(hr):
    """Prázdný, jednoprvkový a celý chybějící vstup."""
    segments = find_segments(hr, THRESHOLD, bridge_tolerance_s=15, smooth_s=10)
    summary = summarize_segments(segments, long_block_s=180)

    assert segments.shape[1] == 2
    assert summary.total_time_s == int(sum(_lengths(segments)))


def test_jednoprvkovy_vstup_nad_prahem_je_usek_1s():
    assert _lengths(find_segments(np.array([175.0]), THRESHOLD)) == [1]


# ═══════════════════════════════════════════════════════════════════════════
# Tepová křivka
# ═══════════════════════════════════════════════════════════════════════════

def test_krivka_na_konstantnim_signalu():
    """Konstantní tep → všechny délky stejná hodnota."""
    hr = np.full(3600, 165.0)

    curve = max_mean_curve(hr, [5, 10, 30, 60, 300, 1200, 3600])

    assert set(curve.values()) == {165.0}
    assert len(curve) == 7


def test_krivka_kratsi_aktivita_radek_nevznikne():
    """Aktivita kratší než okno → klíč chybí. Ne nula, ne None."""
    hr = np.full(600, 170.0)

    curve = max_mean_curve(hr, [300, 600, 1200, 3600])

    assert sorted(curve) == [300, 600]
    assert 1200 not in curve


def test_krivka_najde_nejlepsi_okno_ne_prumer_aktivity():
    """Pětiminutové úsilí uprostřed klidné jízdy musí křivka najít."""
    hr = np.full(3600, 120.0)
    hr[1000:1300] = 180.0      # přesně 300 s

    curve = max_mean_curve(hr, [300, 600])

    assert curve[300] == 180.0
    assert curve[600] < 180.0


def test_krivka_neprumeruje_pres_diru():
    """
    Okno musí být plně pokryté daty.

    Průměr z 15 minut prezentovaný jako dvacetiminutový je horší než
    chybějící řádek – tváří se srovnatelně s ostatními jízdami.
    """
    hr = np.full(1200, 175.0)
    hr[600:700] = np.nan

    curve = max_mean_curve(hr, [300, 1200])

    assert 1200 not in curve
    assert curve[300] == 175.0


def test_krivka_pauzy_nechava_v_datech():
    """
    Zastavení s klesajícím tepem průměr snižuje – a má.

    Tep při zastavení neklesá skokově a je součástí zátěže; hr_blocks.py
    to řeší opačně a je to vědomý rozdíl.
    """
    hr = np.concatenate([np.full(300, 180.0), np.full(300, 120.0)])

    curve = max_mean_curve(hr, [600])

    assert curve[600] == 150.0


@pytest.mark.parametrize("hr", [np.empty(0), np.array([175.0]), np.array([np.nan] * 100)])
def test_krivka_okrajove_vstupy(hr):
    curve = max_mean_curve(hr, [5, 60, 1200])
    assert all(isinstance(v, float) for v in curve.values())


# ═══════════════════════════════════════════════════════════════════════════
# Příprava streamu
# ═══════════════════════════════════════════════════════════════════════════

def test_prazdna_mrizka_z_prazdneho_vstupu():
    assert to_second_grid(np.empty(0), np.empty(0)).size == 0


def test_mrizka_doplni_diru_po_autopauze():
    """
    Chybějící sekundy se doplní na mřížku, krátká díra ffillem.

    Bez toho by klouzavé okno "20 minut" pokrylo reálně víc času – na jízdě
    26. 7. 2026 je 6 546 sekund děr v 17 239 sekundách záznamu.
    """
    ts = np.array([0, 1, 2, 10, 11], dtype=np.int64)
    hr = np.array([170.0, 171.0, 172.0, 160.0, 161.0])

    grid = to_second_grid(ts, hr, ffill_limit_s=5)

    assert grid.size == 12
    assert grid[2] == 172.0
    assert grid[3:8].tolist() == [172.0] * 5      # ffill jen do 5 s
    assert np.isnan(grid[8])
    assert np.isnan(grid[9])
    assert grid[10] == 160.0


def test_mrizka_zvlada_neserazene_a_duplicitni_casy():
    """Sloučené fragmenty Strava exportů nosí obojí."""
    ts = np.array([2, 0, 1, 1], dtype=np.int64)
    hr = np.array([172.0, 170.0, np.nan, 171.0])

    grid = to_second_grid(ts, hr)

    assert grid.tolist() == [170.0, 171.0, 172.0]


def test_mrizka_z_datetime64():
    ts = np.array(
        ["2026-05-08T10:00:00", "2026-05-08T10:00:01", "2026-05-08T10:00:03"],
        dtype="datetime64[s]",
    )
    hr = np.array([170.0, 171.0, 173.0])

    grid = to_second_grid(ts, hr)

    assert grid.tolist() == [170.0, 171.0, 171.0, 173.0]


# ═══════════════════════════════════════════════════════════════════════════
# Histogram délek úseků
# ═══════════════════════════════════════════════════════════════════════════
# Ukládá se předpočítaný, protože z uložených souhrnů (počet úseků, medián)
# se rozdělení sestavit nedá a dopočítat by znamenalo znovu číst records.

BUCKETS = [30, 60, 120, 180, 300]


def test_histogram_radi_delky_do_kosu():
    lengths = np.array([10, 29, 45, 90, 150, 240, 600])

    counts, seconds = segment_histogram(lengths, BUCKETS)

    assert counts == [2, 1, 1, 1, 1, 1]
    assert seconds == [39, 45, 90, 150, 240, 600]
    assert sum(counts) == lengths.size
    assert sum(seconds) == int(lengths.sum())


def test_hranice_kose_patri_do_nizsiho():
    """Úsek dlouhý přesně 30 s je "<30 s", ne "30–60 s"."""
    counts, _ = segment_histogram(np.array([30, 60, 180]), BUCKETS)
    assert counts == [1, 1, 0, 1, 0, 0]


def test_histogram_bez_useku_je_samá_nula():
    """Nula úseků je platná odpověď, ne chybějící údaj."""
    counts, seconds = segment_histogram(np.empty(0, dtype=np.int64), BUCKETS)
    assert counts == [0] * 6
    assert seconds == [0] * 6


def test_souhrn_nese_histogram_ktery_sedi_na_useky():
    hr = np.concatenate([
        np.full(20, 175.0), np.full(60, 150.0),      # 20 s
        np.full(250, 175.0), np.full(60, 150.0),     # 250 s
        np.full(400, 175.0),                          # 400 s
    ])

    summary = summarize_segments(find_segments(hr, THRESHOLD), 180, BUCKETS)

    assert summary.segment_count == 3
    assert sum(summary.hist_counts) == summary.segment_count
    assert sum(summary.hist_seconds) == summary.total_time_s
    assert summary.hist_counts[0] == 1       # 20 s → "<30 s"
    assert summary.hist_counts[4] == 1       # 250 s → "3–5 min"
    assert summary.hist_counts[5] == 1       # 400 s → ">5 min"
