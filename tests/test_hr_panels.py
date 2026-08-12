"""
Podklad pro panely tepové křivky a souvislých bloků.

Těžiště testů je jedno pravidlo, které se nejsnáz poruší a nejhůř se pozná:
**chybějící okno není nula.** Aktivita bez hodinového úseku dat nemá v
křivce hodinový bod, agregace přes období bez takové jízdy vrací ``None`` a
UI z toho musí udělat pomlčku. Nula by tvrdila, že se hodina odjela na nule.

Zbytek jsou lookup zóny (změna LTHR nesmí sáhnout na data), verdikt pokrytí
a nezáměrná Z3.
"""

from __future__ import annotations

from datetime import date

import pytest

from config.settings import (
    HR_BLOCK_THRESHOLDS_BPM,
    HR_COVERAGE_WARN_PCT,
    HR_SEGMENT_BUCKET_LABELS,
)
from src.analytics.hr_panels import (
    Coverage,
    block_period_summary,
    curve_points,
    last_max_effort,
    unintentional_z3_s,
    zone_threshold_bpm,
)

DURATIONS = [5, 60, 1200, 3600]


def _curve_row(duration: int, hr: float, day: str = "2026-07-01", aid: str = "a1") -> dict:
    return {
        "duration_s": duration,
        "max_mean_hr": hr,
        "activity_id": aid,
        "date": date.fromisoformat(day),
        "label": "cycling · 60 km",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Chybějící okno není nula
# ═══════════════════════════════════════════════════════════════════════════

def test_chybejici_okno_je_none_ne_nula():
    """
    Jízda 12. 7. 2026 nemá řádek pro 3600 s – nejdelší souvislý úsek dat je
    2 710 s. Bod musí vzniknout s ``None``, ne s nulou a ne se vynechat:
    graf na něm končí, tabulka ukáže pomlčku.
    """
    points = curve_points([_curve_row(1200, 172.4)], DURATIONS)
    by_duration = {p.duration_s: p for p in points}

    assert set(by_duration) == set(DURATIONS)      # bod vznikne pro každé okno
    assert by_duration[1200].max_mean_hr == 172.4
    assert by_duration[3600].max_mean_hr is None
    assert by_duration[3600].max_mean_hr != 0


def test_agregace_pres_obdobi_bez_okna_vraci_none():
    """MAX(max_mean_hr) přes období, kde hodinové okno nemá žádná jízda."""
    rows = [
        _curve_row(1200, 168.0, "2026-06-01", "a1"),
        _curve_row(1200, 171.5, "2026-06-08", "a2"),
    ]
    points = {p.duration_s: p for p in curve_points(rows, DURATIONS)}

    assert points[1200].max_mean_hr == 171.5       # maximum, ne poslední
    assert points[1200].activity_id == "a2"        # i s jízdou, ze které je
    assert points[3600].max_mean_hr is None
    assert points[3600].activity_id is None


def test_prazdne_obdobi_neda_nuly():
    """Období bez jediné jízdy: samé None, žádná nula."""
    points = curve_points([], DURATIONS)
    assert all(p.max_mean_hr is None for p in points)
    assert all(p.date is None for p in points)


def test_nejdelsi_blok_bez_jizd_je_none():
    """
    Prázdné období vrací ``None``, ne nulu. Nula znamená "jel jsem, ale nad
    práh se nedostal" – to je jiné tvrzení než "nejel jsem".
    """
    assert block_period_summary([])["longest_block_s"] is None


def test_nula_nad_prahem_zustava_nulou():
    """Jízda, která se nad práh nedostala, ale proběhla, má nulu – ne None."""
    row = {
        "activity_id": "a1",
        "longest_block_s": 0,
        "total_time_s": 0,
        "time_in_long_blocks_s": 0,
        "segment_count": 0,
        "segment_hist_counts": [0] * len(HR_SEGMENT_BUCKET_LABELS),
        "segment_hist_seconds": [0] * len(HR_SEGMENT_BUCKET_LABELS),
        "date": date(2026, 7, 1),
        "label": "cycling",
    }
    assert block_period_summary([row])["longest_block_s"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Poslední maximální výkon
# ═══════════════════════════════════════════════════════════════════════════

def test_posledni_maximum_pocita_dny_od_prispivajici_jizdy():
    """Ne poslední tvrdá jízda, ale ta, která do 20min bodu opravdu přispěla."""
    rows = [
        _curve_row(1200, 175.0, "2026-05-01", "stary_rekord"),
        _curve_row(1200, 160.0, "2026-08-01", "novejsi_ale_slabsi"),
    ]
    points = curve_points(rows, DURATIONS)

    effort = last_max_effort(points, 1200, date(2026, 8, 11), stale_days=60)

    assert effort["activity_id"] == "stary_rekord"
    assert effort["days_ago"] == 102
    assert effort["stale"] is True


def test_posledni_maximum_bez_okna_je_none():
    effort = last_max_effort(curve_points([], DURATIONS), 1200, date(2026, 8, 11))
    assert effort is None


# ═══════════════════════════════════════════════════════════════════════════
# Zóna je lookup, ne přepočet
# ═══════════════════════════════════════════════════════════════════════════

def test_prah_zony_je_z_ulozene_mrizky():
    """Výsledek musí být vždy existující řádek, jinak dotaz nic nevrátí."""
    for lthr in range(150, 200):
        for zone in ("Z2", "Z3", "Z4", "Z5"):
            assert zone_threshold_bpm(lthr, zone) in HR_BLOCK_THRESHOLDS_BPM


def test_zmena_lthr_posouva_jen_prah():
    """Vyšší LTHR = vyšší (nebo stejný) práh, nikdy nižší."""
    prev = 0
    for lthr in range(150, 200):
        current = zone_threshold_bpm(lthr, "Z4")
        assert current >= prev
        prev = current


def test_prah_odpovida_merenym_zonam():
    """Při LTHR z měřených zón musí Z4 vyjít na nejbližší práh k 172."""
    assert zone_threshold_bpm(172, "Z4") == 170
    assert zone_threshold_bpm(172, "Z3") == 155


# ═══════════════════════════════════════════════════════════════════════════
# Pokrytí: verdikt, ne tři procenta
# ═══════════════════════════════════════════════════════════════════════════

def _coverage(usable: int, span: int, measured: int, gap: int, max_curve: int | None):
    return Coverage(
        span_s=span,
        measured_s=measured,
        usable_s=usable,
        longest_gap_s=gap,
        max_curve_duration_s=max_curve,
    )


def test_ridky_zapis_neni_varovani():
    """
    Smart Recording: hustota vzorků 20 %, ale po doplnění mezer plná mřížka.
    Tohle je nejdůležitější rozlišení celého pokrytí – varovat na hustotě by
    znamenalo označit stovky aktivit, kterým nic není.
    """
    cov = _coverage(usable=3600, span=3600, measured=720, gap=0, max_curve=3600)

    assert cov.sample_density_pct == pytest.approx(20.0)
    assert cov.ok is True
    assert cov.note is None


def test_vypadek_tepu_je_varovani_i_s_duvodem():
    """Jízda 12. 7. 2026: pokrytí 55 %, díra 56 min, křivka končí na 30 min."""
    cov = _coverage(usable=13397, span=24223, measured=13031, gap=3330, max_curve=1800)

    assert cov.ok is False
    assert "55 %" in cov.note
    assert "56 min" in cov.note
    assert "30 min" in cov.note          # kam až křivka sahá
    assert "podhodnocené" in cov.note    # co to znamená pro bloky


def test_hranice_varovani_sedi_na_nastaveni():
    tesne_nad = _coverage(
        usable=int(HR_COVERAGE_WARN_PCT * 100), span=10000, measured=9000, gap=10, max_curve=3600
    )
    tesne_pod = _coverage(
        usable=int(HR_COVERAGE_WARN_PCT * 100) - 1, span=10000, measured=9000, gap=10,
        max_curve=3600,
    )
    assert tesne_nad.ok is True
    assert tesne_pod.ok is False


def test_bez_rozsahu_neni_pokryti():
    """Nulový rozsah nesmí dát 0 % – to by vypadalo jako naprostý výpadek."""
    cov = _coverage(usable=0, span=0, measured=0, gap=0, max_curve=None)
    assert cov.coverage_pct is None
    assert cov.as_dict()["pct"] is None
    # Odznak visí na `ok`, takže i tenhle stav musí mít vysvětlení.
    assert cov.ok is False
    assert cov.note is not None


# ═══════════════════════════════════════════════════════════════════════════
# Nezáměrná Z3
# ═══════════════════════════════════════════════════════════════════════════

def test_nezamerna_z3_odecte_dlouhe_bloky():
    """Ze 40 minut nad 155 bylo 25 v souvislých blocích → 15 nezáměrných."""
    total = {155: 2400, 170: 0}
    long = {155: 1500, 170: 0}

    assert unintentional_z3_s(total, long, 155, 170) == 900


def test_souvisly_z4_interval_nedela_zapornou_z3():
    """
    Dvacetiminutový práh leží celý nad spodní hranicí Z3. Kdyby se
    odečítaly jen dlouhé bloky nad 155, metrika by šla do záporu.
    """
    total = {155: 1200, 170: 1200}
    long = {155: 1200, 170: 1200}

    assert unintentional_z3_s(total, long, 155, 170) == 0


def test_kratke_vyjezdy_do_z4_se_nepocitaji_do_z3():
    """Sekundy nad 170 patří Z4, ne Z3 – i když jsou v krátkých úsecích."""
    total = {155: 600, 170: 200}
    long = {155: 0, 170: 0}

    assert unintentional_z3_s(total, long, 155, 170) == 400


def test_chybejici_prah_nezpusobi_pad():
    assert unintentional_z3_s({}, {}, 155, 170) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Histogram délek úseků
# ═══════════════════════════════════════════════════════════════════════════

def test_histogram_secte_pocty_i_cas_pres_jizdy():
    rows = [
        {
            "activity_id": "a1",
            "longest_block_s": 400,
            "total_time_s": 1000,
            "time_in_long_blocks_s": 400,
            "segment_count": 5,
            "segment_hist_counts": [3, 1, 0, 0, 0, 1],
            "segment_hist_seconds": [45, 40, 0, 0, 0, 400],
            "date": date(2026, 7, 1),
            "label": "cycling",
        },
        {
            "activity_id": "a2",
            "longest_block_s": 900,
            "total_time_s": 1500,
            "time_in_long_blocks_s": 900,
            "segment_count": 2,
            "segment_hist_counts": [1, 0, 0, 0, 0, 1],
            "segment_hist_seconds": [20, 0, 0, 0, 0, 900],
            "date": date(2026, 7, 8),
            "label": "cycling",
        },
    ]

    summary = block_period_summary(rows)

    assert summary["longest_block_s"] == 900
    assert summary["longest_block"]["activity_id"] == "a2"
    assert summary["hist"][0] == {"bucket": HR_SEGMENT_BUCKET_LABELS[0], "count": 4, "seconds": 65}
    assert summary["hist"][-1]["count"] == 2
    assert summary["totals"]["segment_count"] == 7
