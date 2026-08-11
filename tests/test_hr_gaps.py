"""
Testy zacházení s dírami ve vteřinových datech.

Oddělené od test_hr_blocks.py schválně: tohle nejsou testy segmentace, ale
testy toho, co se stane, když data chybí. Referenční hodnoty spočítané mimo
repo tuhle vrstvu neověří – vznikaly nad zkomprimovaným streamem, kde byly
díry vypuštěné a řádky slepené, takže by shoda mohla znamenat stejnou chybu
na obou stranách.

Rozlišují se tři různé poruchy, které vypadají podobně a nejsou totéž:

  autopauza        chybějící řádky v časové ose (46 % rozsahu jízdy 12. 7.)
  výpadek tepu     řádek existuje, heart_rate je NULL (16 % řádků 11. 4.,
                   nejdelší souvislý výpadek 26 minut)
  krátká mezera    2–5 s bez záznamu, tedy hikup zápisu, ne pauza

První dvě se musí chovat stejně (nepokrytá sekunda), třetí se doplní.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physio.hr_blocks import find_segments, summarize_segments
from src.physio.hr_curve import max_mean_curve
from src.physio.hr_stream import longest_gap, sample_coverage, to_second_grid

THRESHOLD = 170.0
FFILL_LIMIT = 5


def _with_missing_rows(gap_s: int, hr_value: float = 175.0) -> np.ndarray:
    """Dva úseky po 300 s nad prahem, mezi nimi ``gap_s`` chybějících řádků."""
    timestamps = np.concatenate([np.arange(0, 300), np.arange(300 + gap_s, 600 + gap_s)])
    heart_rate = np.full(timestamps.size, hr_value)
    return to_second_grid(timestamps, heart_rate, ffill_limit_s=FFILL_LIMIT)


def _lengths(segments: np.ndarray) -> list[int]:
    return [int(e - s + 1) for s, e in segments]


# ═══════════════════════════════════════════════════════════════════════════
# 1) Okno tepové křivky přes díru
# ═══════════════════════════════════════════════════════════════════════════

def test_krivka_vyzaduje_plne_pokryti_okna():
    """
    Okno bez plného pokrytí skutečnými vzorky řádek nevytvoří.

    Ne průměr z toho, co v okně zbylo – takové číslo se tváří srovnatelně
    s okny, která plná jsou, a přitom měří něco jiného.
    """
    hr = np.full(3600, 175.0)
    hr[1000:1060] = np.nan          # minuta chybí

    curve = max_mean_curve(hr, [300, 3600])

    assert 3600 not in curve
    assert curve[300] == 175.0      # kratší okno se vejde mimo díru


def test_nejdelsi_okno_krivky_je_dano_nejdelsim_souvislym_usekem():
    """
    Aktivita 12. 7. 2026: 403 min rozsahu, 54 % pokrytí, nejdelší úsek dat
    2 710 s.

    Rozhoduje nejdelší souvislý úsek, ne délka aktivity. Hodinové okno nemá
    šanci vyjít, i když je aktivita skoro sedmihodinová – a přesně to v DB
    u téhle jízdy vidíme (uložená okna končí u 1 800 s).
    """
    hr = np.full(24223, np.nan)     # rozsah jako 12. 7.
    for start in range(0, 24223, 4000):
        hr[start : start + 2710] = 175.0    # úseky dat po 2 710 s

    curve = max_mean_curve(hr, [1800, 3600])

    assert 1800 in curve            # vejde se do souvislého úseku 2 710 s
    assert 3600 not in curve


def test_prumer_pres_diru_by_dal_jine_cislo():
    """
    Kontrola, že plné pokrytí opravdu chrání před zkreslením.

    Kdyby se okno počítalo z dostupných vzorků, vyšlo by 180 – průměr jen
    té tvrdší poloviny. Správná odpověď je, že takové okno nevznikne.
    """
    hr = np.concatenate([np.full(1800, 180.0), np.full(1800, np.nan)])

    curve = max_mean_curve(hr, [3600])
    dostupne_vzorky = float(np.nanmean(hr))

    assert curve == {}
    assert dostupne_vzorky == 180.0


# ═══════════════════════════════════════════════════════════════════════════
# 2) Přemostění nesmí překlenout díru
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("gap_s", [6, 8, 10, 15])
def test_dira_kratsi_nez_tolerance_stejne_deli(gap_s):
    """
    Díra do 15 s je kratší než tolerance přemostění, a přesto úsek ukončí.

    Přemostit se smí jen skutečně naměřený tep pod prahem. Chybějící data
    nejsou "tep, který mírně klesl" – nevíme o nich nic.
    """
    grid = _with_missing_rows(gap_s)

    segments = find_segments(grid, THRESHOLD, bridge_tolerance_s=15)

    assert len(segments) == 2, f"mezera {gap_s} s se nesmí přemostit"


@pytest.mark.parametrize("gap_s", [1, 2, 3, 5])
def test_kratka_mezera_se_doplni_a_uzavre_se(gap_s):
    """
    Mezera do limitu ffillu je hikup zápisu, ne pauza – doplní se.

    Po doplnění už to je naměřený tep a úsek pokračuje. Hranice je ostrá:
    5 s se doplní, 6 s ne.
    """
    grid = _with_missing_rows(gap_s)

    assert not np.isnan(grid).any()
    assert _lengths(find_segments(grid, THRESHOLD, bridge_tolerance_s=0)) == [600 + gap_s]


def test_hranice_ffillu_je_ostra():
    """Přesně na limitu se doplní, o sekundu dál už zůstane díra."""
    assert not np.isnan(_with_missing_rows(FFILL_LIMIT)).any()
    assert np.isnan(_with_missing_rows(FFILL_LIMIT + 1)).sum() == 1


def test_podminka_na_hloubku_se_vyhodnocuje_jen_na_namerenych_vzorcich():
    """
    Stejně dlouhý propad: mělký naměřený se přemostí, díra nikdy.

    Bez tohohle rozlišení by chybějící data fungovala jako "tep těsně pod
    prahem" a spojovala by úsilí, o kterých nic nevíme.
    """
    melky = np.full(610, 175.0)
    melky[300:310] = THRESHOLD - 3.0

    hluboky = np.full(610, 175.0)
    hluboky[300:310] = THRESHOLD - 15.0

    chybejici = np.full(610, 175.0)
    chybejici[300:310] = np.nan

    assert _lengths(find_segments(melky, THRESHOLD, 15)) == [610]
    assert _lengths(find_segments(hluboky, THRESHOLD, 15)) == [300, 300]
    assert _lengths(find_segments(chybejici, THRESHOLD, 15)) == [300, 300]


def test_dira_uvnitr_propadu_zabrani_premosteni():
    """Propad, ve kterém je i jediná chybějící sekunda, se nepřemosťuje."""
    hr = np.full(610, 175.0)
    hr[300:310] = THRESHOLD - 1.0      # mělký propad, sám o sobě přemostitelný
    hr[305] = np.nan                   # ale chybí v něm sekunda

    assert _lengths(find_segments(hr, THRESHOLD, 15)) == [300, 300]


# ═══════════════════════════════════════════════════════════════════════════
# 3) Výpadek tepu v existujících řádcích
# ═══════════════════════════════════════════════════════════════════════════

def test_chybejici_tep_v_existujicim_radku_je_take_dira():
    """
    Řádek existuje, heart_rate je NULL → nepokrytá sekunda, stejně jako
    kdyby řádek chyběl. Jiná porucha, stejné zacházení.
    """
    timestamps = np.arange(0, 600)
    heart_rate = np.full(600, 175.0)
    heart_rate[300:330] = np.nan       # 30 s bez tepu, ale s časem

    grid = to_second_grid(timestamps, heart_rate, ffill_limit_s=FFILL_LIMIT)

    assert np.isnan(grid).sum() == 30 - FFILL_LIMIT
    assert len(find_segments(grid, THRESHOLD, bridge_tolerance_s=15)) == 2


def test_dlouhy_vypadek_tepu_nevyrobi_falesny_blok():
    """
    26minutový výpadek tepu uprostřed jízdy (aktivita 11. 4. 2026).

    Bez limitu na ffill by z něj vzniklo 26 minut konstantního tepu a z toho
    jeden falešný blok přes celou jízdu. S limitem výpadek blok ukončí.
    """
    timestamps = np.arange(0, 3600)
    heart_rate = np.full(3600, 175.0)
    heart_rate[600:2160] = np.nan      # 26 minut bez tepu

    grid = to_second_grid(timestamps, heart_rate, ffill_limit_s=FFILL_LIMIT)
    segments = find_segments(grid, THRESHOLD, bridge_tolerance_s=15)
    summary = summarize_segments(segments, long_block_s=180)

    assert summary.segment_count == 2
    assert summary.longest_block_s == 1440          # zbytek po výpadku
    assert summary.longest_block_s < 1560           # ne přes výpadek

    # Kontrast: ffill bez limitu by dal jeden blok přes celou hodinu
    bez_limitu = to_second_grid(timestamps, heart_rate, ffill_limit_s=10_000)
    assert _lengths(find_segments(bez_limitu, THRESHOLD, 0)) == [3600]


def test_dlouhy_vypadek_tepu_nevyrobi_falesny_radek_krivky():
    """Tentýž výpadek nesmí vyrobit ani hodinový řádek tepové křivky."""
    timestamps = np.arange(0, 3600)
    heart_rate = np.full(3600, 175.0)
    heart_rate[600:2160] = np.nan

    grid = to_second_grid(timestamps, heart_rate, ffill_limit_s=FFILL_LIMIT)
    bez_limitu = to_second_grid(timestamps, heart_rate, ffill_limit_s=10_000)

    assert 3600 not in max_mean_curve(grid, [1200, 3600])
    assert 3600 in max_mean_curve(bez_limitu, [1200, 3600])


# ═══════════════════════════════════════════════════════════════════════════
# 4) Pauza se pozná z časové osy, ne z is_active
# ═══════════════════════════════════════════════════════════════════════════

def test_pauza_se_pozna_z_mezery_v_timestampech():
    """
    Vstupem je jen čas a tep – žádný příznak aktivity.

    is_active na to použít nejde: na jízdě 12. 7. 2026 je 11 187 sekund děr
    a ani jeden řádek s is_active = False, medián přes všechny aktivity je
    12 takových řádků. Detekce pauz z toho sloupce by nedetekovala nic.
    """
    bez_pauzy = to_second_grid(np.arange(0, 600), np.full(600, 175.0))
    s_pauzou = _with_missing_rows(120)

    assert not np.isnan(bez_pauzy).any()
    assert np.isnan(s_pauzou).sum() == 120 - FFILL_LIMIT
    assert len(find_segments(s_pauzou, THRESHOLD, 15)) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Pokrytí
# ═══════════════════════════════════════════════════════════════════════════

def test_pokryti_pocita_jen_namerene_vzorky():
    """
    Pokrytí nesmí vylepšit ffill – je to diagnostika vstupu, ne výstupu.

    Deset minut záznamu ve dvacetiminutovém rozsahu je 50 % pokrytí bez
    ohledu na to, že se prvních 5 s díry doplní.
    """
    timestamps = np.concatenate([np.arange(0, 600), np.arange(1200, 1200)])
    heart_rate = np.full(timestamps.size, 175.0)

    measured, span = sample_coverage(timestamps, heart_rate)

    assert measured == 600
    assert span == 600


def test_pokryti_zapocitava_radky_bez_tepu_jako_nepokryte():
    timestamps = np.arange(0, 1000)
    heart_rate = np.full(1000, 175.0)
    heart_rate[400:900] = np.nan

    measured, span = sample_coverage(timestamps, heart_rate)

    assert span == 1000
    assert measured == 500
    assert 100.0 * measured / span == 50.0


def test_nejdelsi_dira_se_meri_po_ffillu():
    """
    Díra, kterou ffill doplnil, už dírou není.

    Report tak neukazuje mezery, které na výsledek nemají vliv.
    """
    assert longest_gap(_with_missing_rows(3)) == 0
    assert longest_gap(_with_missing_rows(20)) == 20 - FFILL_LIMIT


def test_smart_recording_neni_ztrata_dat():
    """
    Vzorek po 5 s (Smart Recording, 305 z 799 aktivit) dá hustotu 20 %,
    ale po doplnění mezer plnou mřížku.

    Varování proto visí na pokrytí po ffillu, ne na hustotě vzorků – jinak
    by křičelo u 588 aktivit a skutečné výpadky by v tom zapadly.
    """
    timestamps = np.arange(0, 3000, 5)
    heart_rate = np.full(timestamps.size, 175.0)

    measured, span = sample_coverage(timestamps, heart_rate)
    grid = to_second_grid(timestamps, heart_rate, ffill_limit_s=FFILL_LIMIT)

    assert 100.0 * measured / span == pytest.approx(20.0, abs=0.1)   # hustota vzorků
    assert not np.isnan(grid).any()                                  # pokrytí 100 %
    assert 1800 in max_mean_curve(grid, [1800])                      # okna vycházejí
    assert _lengths(find_segments(grid, THRESHOLD, 0)) == [grid.size]


@pytest.mark.parametrize(
    "interval_s, expect_full_grid",
    [(5, True), (6, True), (7, False), (10, False)],
)
def test_hranice_rozestupu_vzorku(interval_s, expect_full_grid):
    """
    Rozestup vzorků N sekund znamená N−1 chybějících sekund mezi nimi.

    Limit ffillu je 5, takže vzorek po 6 s se ještě celý doplní a teprve od
    7 s zůstává mřížka děravá. Aktivita s takovým zápisem se pak trhá na
    úseky po jednom vzorku – a přesně to má report ukázat, ne schovat.
    """
    timestamps = np.arange(0, 3000, interval_s)
    heart_rate = np.full(timestamps.size, 175.0)

    grid = to_second_grid(timestamps, heart_rate, ffill_limit_s=FFILL_LIMIT)

    assert np.isnan(grid).any() is not expect_full_grid
    if expect_full_grid:
        assert _lengths(find_segments(grid, THRESHOLD, 0)) == [grid.size]
    else:
        assert len(find_segments(grid, THRESHOLD, bridge_tolerance_s=15)) > 100
