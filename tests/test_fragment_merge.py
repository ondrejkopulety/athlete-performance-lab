"""
Sloučení fragmentů v parseru
=============================

Strava exporty dělí jednu vteřinu do několika 'record' zpráv, každou s jinou
podmnožinou polí. Parser počítá délku segmentu jako rozdíl proti předchozímu
záznamu, takže fragment s tepem dostával nulový časový krok a nepřinesl do
zón ani do TRIMP nic – jeden fotbalový zápas tak vyšel na TRIMP 0,9 místo 154.

Tenhle test hlídá tutéž semantiku, kterou o vrstvu níž hlídá
tests/test_records_merge.py (tam GROUP BY v SQL, tady v parseru). Obojí musí
dávat stejný výsledek, jinak by přepočet z databáze nesouhlasil s parserem.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.ingestion.fit_parser import _merge_record_fragments

T0 = datetime(2023, 7, 12, 16, 34, 56)


def test_fragments_of_one_second_collapse_into_one_row():
    """Tři fragmenty téže vteřiny = jeden úplný záznam."""
    merged = _merge_record_fragments([
        {"timestamp": T0, "distance": 1.12},
        {"timestamp": T0, "speed": 0.855},
        {"timestamp": T0, "heart_rate": 73},
    ])

    assert len(merged) == 1
    assert merged[0]["distance"] == 1.12
    assert merged[0]["speed"] == 0.855
    assert merged[0]["heart_rate"] == 73


def test_later_value_wins_but_none_never_overwrites():
    """Novější neprázdná hodnota přepíše starší; None nepřepíše nic."""
    merged = _merge_record_fragments([
        {"timestamp": T0, "heart_rate": 73, "speed": 1.0},
        {"timestamp": T0, "heart_rate": 75},
        {"timestamp": T0, "speed": None},
    ])

    assert len(merged) == 1
    assert merged[0]["heart_rate"] == 75
    assert merged[0]["speed"] == 1.0


def test_one_hz_input_passes_through_unchanged():
    """Garmin má jednu zprávu na vteřinu – sloučení musí být no-op."""
    raw = [
        {"timestamp": T0 + timedelta(seconds=i), "heart_rate": 100 + i}
        for i in range(5)
    ]

    merged = _merge_record_fragments(raw)

    assert len(merged) == len(raw)
    assert [r["heart_rate"] for r in merged] == [100, 101, 102, 103, 104]
    assert [r["timestamp"] for r in merged] == [r["timestamp"] for r in raw]


def test_records_without_timestamp_survive_as_separate_rows():
    """Bez času není podle čeho slučovat – řádek se nesmí zahodit."""
    merged = _merge_record_fragments([
        {"timestamp": None, "heart_rate": 60},
        {"timestamp": T0, "heart_rate": 73},
        {"timestamp": "nonsense", "heart_rate": 61},
    ])

    assert len(merged) == 3
    assert [r["timestamp"] for r in merged] == [None, T0, None]


def test_tz_aware_and_naive_timestamps_of_same_instant_merge():
    """
    Strava vrací čas s časovou zónou, Garmin bez ní. Po převodu na naivní
    UTC jde o tentýž okamžik, takže se fragmenty musí potkat.
    """
    aware = T0.replace(tzinfo=timezone.utc)

    merged = _merge_record_fragments([
        {"timestamp": aware, "distance": 1.12},
        {"timestamp": T0, "heart_rate": 73},
    ])

    assert len(merged) == 1
    assert merged[0]["timestamp"] == T0
    assert merged[0]["timestamp"].tzinfo is None
    assert merged[0]["distance"] == 1.12
    assert merged[0]["heart_rate"] == 73


def test_iso_string_timestamps_are_parsed_and_merged():
    """Fragmenty s časem jako řetězcem patří ke stejné vteřině."""
    merged = _merge_record_fragments([
        {"timestamp": T0.isoformat(), "distance": 1.12},
        {"timestamp": T0, "heart_rate": 73},
    ])

    assert len(merged) == 1
    assert merged[0]["timestamp"] == T0
