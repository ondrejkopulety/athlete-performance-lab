# Datová složka

Zdrojem pravdy je **databáze** (PostgreSQL + TimescaleDB, viz
`docker-compose.yml`). Soubory tady jsou buď vstupy, které do databáze
teprve vedou, nebo výstupy z ní.

```
data/
├── fit/                  ← VSTUP: FIT soubory (zdroj pravdy pro aktivity)
│   └── strava_originals/
├── Apple/                ← VSTUP: exporty z Apple Health (2017–2025)
├── raw/                  ← VSTUP: surové odpovědi Garmin API
├── summaries/            ← VSTUP i VÝSTUP, viz níže
├── processed/            ← VÝSTUP: botanická analýza
├── _baseline/            ← zmrazený referenční bod pro testy (needitovat)
└── _archive/             ← osiřelé soubory, které nikdo nečte
```

## fit/ — zdroj pravdy

FIT soubory jsou jediné, co nejde znovu vyrobit. Všechno ostatní se z nich
dá dopočítat: `python scripts/main.py load --force` naplní databázi znovu
od nuly. **Tohle zálohuj.**

Deduplikace rozhoduje, který soubor je kanonický, když tatáž aktivita leží
na disku dvakrát (Garmin i Strava) — viz `src/ingestion/dedup.py`.

## summaries/ — dvě různé role v jedné složce

**Vstupy** — píše je `garmin_sync.py`, čte `biometrics_import.py`:

`activities.csv` · `hrv.csv` · `sleep.csv` · `daily_health.csv` ·
`movement.csv` · `intensity.csv` · `vo2_max.csv` ·
`training_readiness.csv` · `heart_rate_*.csv`

**Výstupy z databáze** — generuje `python scripts/main.py export`:

| soubor | obsah |
|---|---|
| `master_high_res_summary.csv` | všechny aktivity včetně odvozených metrik |
| `athlete_readiness.csv` | denní metriky (PMC, regenerace, kvalita) |
| `cycling_summary.csv` | jen cyklistika |

Export běží automaticky na konci každé pipeline, takže soubory
nezastarávají. Dřív je psala sama pipeline a po přechodu na databázi se
přestaly aktualizovat — vypadaly aktuálně, ale byly zamrzlé.

**Zbytek** — `apple_*.csv` píše `apple_health.py`, `metadata_cache.json`
zrychluje deduplikaci, `master_high_res_training_data.csv` (582 MB) ještě
čte botanický modul, než se převede na databázi.

## _baseline/ — needitovat

Zmrazený výstup **původní** implementace z 6. 8. 2026. Slouží jen jako
referenční bod pro `tests/test_parity.py`. Kdyby se přepsal živým exportem,
test by porovnával výstup sám se sebou a přestal by cokoli hlídat.

## _archive/ — nikdo nečte, nic se neztratí

Osiřelé soubory, které vznikly během vývoje. Ověřeno, že **každá aktivita
v nich je i v databázi**:

| soubor | proč tady je |
|---|---|
| `high_res_training_data.csv` (168 MB) | výstup zrušeného kroku `parse`, nikdo ho nečetl |
| `high_res_summary.csv` | totéž |
| `AHOJ.csv`, `DNESKA.csv`, `rekon.csv`, `kolo.csv` | ad-hoc exporty z ladění |
| ` .csv` | soubor pojmenovaný mezerou |
| `master_high_res_summary (old).csv` | starší verze, 735 aktivit |

Můžeš to smazat — dohromady ~210 MB. Nechávám to na tobě, jsou to tvoje data.

## Co zálohovat a co ne

| složka | zálohovat? |
|---|---|
| `fit/`, `Apple/` | **ano** — nejde vyrobit znovu |
| databáze (Docker volume `garmin_pgdata`) | volitelně — dá se obnovit z FIT |
| `summaries/`, `processed/`, `_archive/` | ne — všechno je odvozené |
