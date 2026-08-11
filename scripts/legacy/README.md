# Zastaralé skripty

Tyhle skripty čtou CSV soubory, které pipeline **už neaktualizuje**, nebo
řeší úlohu, kterou dnes zvládne něco jiného. Nechávám je tu, protože můžou
obsahovat logiku, kterou budeš chtít, ale spouštět je znamená pracovat
se zamrzlými daty.

| skript | stav | náhrada |
|---|---|---|
| `enrich_master_with_readiness.py` | čte `master_high_res_summary.csv` a `athlete_readiness.csv` | data jsou spojená v databázi, `GET /api/activities/{id}` |
| `test_api.py` | ruční sonda proti Garmin API | `python scripts/main.py sync` |
| `weed.py` | jednorázový úklid dat | – |

Než některý z nich použiješ, přepiš ho na databázi – v `src/db/repository.py`
je na to všechno potřebné.

`split_cycling_activities.py` tuhle cestu už prošel: skládal
`data/cycling_splits/` z 582MB CSV, které pipeline neaktualizuje, a nahradil
ho `export_cycling_splits()` v `src/analytics/exports.py`
(`scripts/main.py splits`). Historie zbytku skriptu zůstává v gitu.
