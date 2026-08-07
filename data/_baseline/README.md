# Zmrazený baseline pro regresní testy

Tyhle dva soubory vygeneroval **původní** `athlete_analytics.py` před
přechodem na databázi (6. 8. 2026). Slouží výhradně jako referenční bod
pro `tests/test_parity.py`.

**Needituj je a nepřepisuj.** Čerstvé exporty z databáze jdou do
`data/summaries/` (`python scripts/main.py export`) – tyhle musí zůstat
zamrzlé, jinak by test porovnával výstup sám se sebou a nic by nehlídal.

| soubor | co obsahuje |
|---|---|
| `athlete_readiness.csv` | denní metriky, 1652 dní |
| `master_high_res_summary.csv` | 864 aktivit se souhrnnými metrikami |
