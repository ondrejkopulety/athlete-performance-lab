# Garmin Training Analytics

Automatizovaná datová pipeline pro analýzu cyklistického tréninku a denní
biometrie z ekosystému Garmin. Stahuje data z Garmin Connect, dekóduje FIT
soubory, počítá sportovní metriky a servíruje je přes REST API — včetně
strukturovaného kontextu pro AI trenéra.

---

## Rychlý start

```bash
# 1. Závislosti
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Konfigurace
cp .env.example .env        # doplň Garmin credentials

# 3. Databáze (PostgreSQL 16 + TimescaleDB)
docker compose up -d db
.venv/bin/alembic upgrade head

# 4. Jednorázový import historických CSV (pokud je máš)
.venv/bin/python scripts/migrate_csv_to_db.py

# 5. Běh pipeline
.venv/bin/python scripts/main.py

# 6. API
.venv/bin/uvicorn src.api.app:app --reload   # http://localhost:8000/docs
```

---

## Pipeline

```
SYNC ──▶ IMPORT ──▶ LOAD ──▶ ANALYZE
```

| Krok | Co dělá |
|---|---|
| **SYNC** | Stáhne novinky z Garmin Connect (CSV souhrny + FIT soubory na disk) |
| **IMPORT** | Přetaví denní CSV (HRV, spánek, RHR, stres) do `daily_biometrics` |
| **LOAD** | Deduplikuje Garmin vs. Strava FIT soubory a zapíše je do `activities` + `records` |
| **ANALYZE** | Per-activity metriky (inkrementálně) + denní metriky (PMC, regenerace, kvalita) |

```bash
python scripts/main.py                  # celá pipeline
python scripts/main.py analyze          # jen přepočet metrik
python scripts/main.py load --force     # přeparsovat všechny FIT soubory
python scripts/main.py --skip-download  # bez sítě, jen nad lokálními daty
python scripts/main.py status           # co je v databázi
```

### Inkrementalita

Denní běh trvá jednotky sekund, protože se přepočítává jen to, co se změnilo:

- **FIT soubory** se identifikují SHA-256 obsahu. Soubor se stejným hashem
  se přeskočí bez otevření.
- **Per-activity metriky** (cardiac drift, DFA-alpha1, RSA, durabilita) nesou
  `metrics_version`. Přepočítají se jen aktivity s chybějící nebo zastaralou
  verzí — typicky jedna denně místo 864.
- **Denní metriky** se počítají vždy celé. Je to levné (~1 s na 1650 dní) a
  jediné, co zaručí identická čísla: `ef_trend` používá `ffill` s neomezenou
  pamětí a 42denní EMA nese vliv řádově rok zpátky, takže žádné konečné
  lookback okno by nedalo přesný výsledek.

Změnil jsi vzorec? Zvyš `ACTIVITY_METRICS_VERSION` v `config/settings.py` a
pipeline sama přepočítá vše dotčené.

---

## Metriky

**Tréninková zátěž (PMC)** — denní TRIMP (Banister), CTL (42 d), ATL (7 d),
TSB s jednodenním posunem (ranní forma *před* dnešním tréninkem).

**Prevence zranění** — ACWR (7 d / 28 d z EPOC-vážené TRIMP, bez clipování),
CTL ramp rate.

**Kvalita tréninku** — Fosterova monotonie a strain, Whoop logaritmický
strain (0–21), Seilerova polarizace 80/20 s penalizací za Z3 „junk miles".

**Regenerace a biometrie** — Pure Recovery Score (HRV 40 % + RHR 30 % +
spánek 30 %), dual-era Bio-Readiness, sleep performance (Whoop-style),
7denní HRV CV, celodenní Garmin stres, multi-indikátorový illness warning.

**Per-activity fyziologie** — cardiac drift (Pa:HR decoupling), maximální
pokles tepu za 60 s, durabilita, VAM, DFA-alpha1 (aerobní i anaerobní práh
z R-R intervalů přes neurokit2), dechová frekvence z RSA, EPOC, Critical HR
a TATI, fueling model, odhad ztráty tekutin.

Význam, jednotky a směr každé metriky jsou strojově čitelné v
`config/settings.py` → `METRIC_META` a servírují se na `/api/coach/glossary`.

---

## API

```
GET  /health
GET  /api/daily?from=&to=              denní metriky (časová řada)
GET  /api/daily/latest                 dnešní snapshot
GET  /api/pmc?days=180                 CTL / ATL / TSB / ACWR
GET  /api/activities?from=&to=&sport=  seznam aktivit
GET  /api/activities/{id}              detail včetně odvozených metrik
GET  /api/activities/{id}/records?resolution=10s
                                       downsamplovaná vteřinová data
GET  /api/coach/context?date=          strukturovaný JSON pro LLM
GET  /api/coach/history/{metric}?days= časová řada jedné metriky
GET  /api/coach/glossary               význam a jednotky metrik
POST /api/sync/run                     spustí pipeline na pozadí
GET  /api/sync/status                  stav posledního běhu
```

Vteřinová data se agregují přes TimescaleDB `time_bucket` — aktivita má
desítky tisíc bodů a syrová data nemá smysl posílat do prohlížeče.

### Kontext pro AI trenéra

`/api/coach/context` vrací JSON navržený tak, aby model rozuměl nejen
číslům, ale i jejich významu:

```json
{
  "athlete":  { "max_hr": 199, "hr_zones_bpm": { "Z1": [100, 136] } },
  "today":    { "training_load": {...}, "recovery": {...}, "risk": {...} },
  "trends":   { "7d": {...}, "28d": {...} },
  "recent_activities": [...],
  "metric_glossary": {
    "acwr": { "unit": "poměr", "direction": "sweet_spot", "sweet_spot": [0.8, 1.3] }
  }
}
```

`metric_glossary` je zásadní: bez něj model neví, že u `hrv_cv_pct` je nižší
lepší nebo že ACWR má optimum uprostřed škály. Statická část (profil +
glosář) je stabilní napříč dny, takže se dá cachovat přes prompt caching a
platit plnou cenu jen za měnící se data.

---

## Datový model

| Tabulka | Obsah |
|---|---|
| `activities` | Surová fakta z FIT — nikdy se nepřepočítávají |
| `activity_metrics` | Odvozené metriky + `metrics_version` (přepočitatelné) |
| `records` | Vteřinová data, TimescaleDB hypertable (chunk 7 dní, komprese po 30) |
| `daily_biometrics` | Denní vstupy z Garmin Connect (HRV, spánek, RHR, stres) |
| `daily_metrics` | Výstup analytiky — jeden řádek na den |
| `sync_state` | Kde skončil poslední běh |

Oddělení surových a odvozených dat je to, co dělá inkrementalitu možnou:
změna vzorce se projeví bumpnutím verze, aniž by se sáhlo na vstupy.

---

## Deduplikace Garmin vs. Strava

Tatáž aktivita bývá na disku dvakrát — jednou z hodinek, jednou ze Stravy.
`src/ingestion/dedup.py` v okně ±30 minut rozhodne, který soubor je lepší:

1. **Pojistka integrity** — soubor s >25 % více záznamy vyhrává (druhý je
   oříznutý); Smart Recording se od poškozených dat rozlišuje HR density
2. **Hrudní pás > optika** — ANT+ pás vyhrává bez ohledu na zdroj
3. **Detekce pásu přes HRV zprávy** — Strava exporty nemají `device_info`,
   ale přítomnost R-R intervalů pás spolehlivě prozradí
4. **HR density** ≥ 90 %, jinak se přepne na druhý zdroj
5. **Větší soubor** při naprosté shodě

Rozhodnutí se propíše do `activities.source`, takže je u každé aktivity
dohledatelné.

---

## Kalendář

Denní osa **vždy končí dneškem** a **začíná nejstarším záznamem z jakéhokoli
zdroje** (aktivita nebo biometrie). Obojí je podstatné: bez prvního by se
ztrácela ranní biometrie ve dnech bez tréninku, bez druhého biometrie
z období před prvním zaznamenaným tréninkem.

Den bez tréninku má `trimp = 0` — to je skutečná hodnota, ne chybějící údaj.
Biometrické sloupce naopak zůstávají `NULL`, dokud data z hodinek nedorazí.

---

## Testy

```bash
.venv/bin/python -m pytest tests/ -q
```

| Soubor | Co hlídá |
|---|---|
| `test_parity.py` | Shodu s výstupem původní implementace (`athlete_readiness.csv`) |
| `test_calendar.py` | Kalendář končí dneškem, začíná nejstarším záznamem, nemá díry |
| `test_metrics.py` | Fyziologické invarianty (rozsahy skóre, součet polarizace, TSB posun) |
| `test_load.py` | Vzorce PMC — EMA konstanta, TSB posun, koeficient pro pěší sporty |

Testy vyžadující databázi se automaticky přeskočí, pokud neběží.

---

## Struktura

```
config/settings.py        Parametry atleta, prahy, METRIC_META — single source of truth
src/
  db/                     SQLAlchemy modely, session, repository
  ingestion/
    garmin_sync.py        Stahování z Garmin Connect
    biometrics_import.py  Denní CSV → daily_biometrics
    fit_parser.py         Dekódování FIT (čistá extrakce, nezapisuje na disk)
    dedup.py              Výběr kanonické sady souborů (Garmin vs. Strava)
    loader.py             FIT → databáze
  analytics/
    calendar.py           Denní osa
    activity.py           Per-activity fyziologie
    load.py               TRIMP, CTL/ATL/TSB, ACWR
    biometrics.py         Regenerace, readiness, illness warning
    quality.py            Monotonie, strain, polarizace, efficiency
    advice.py             Textové doporučení
    pipeline.py           Orchestrace analytiky
  coach/context.py        Kontext pro LLM
  api/                    FastAPI (app, schemas, routers)
  pipeline.py             Celý běh na jednom místě (sdílí CLI i API)
scripts/
  main.py                 CLI
  migrate_csv_to_db.py    Jednorázový import historických CSV
```

---

## Plánované

- Frontend dashboard (Next.js + TypeScript; Recharts pro denní řady,
  uPlot pro vteřinová data)
- Napojení `/api/coach/context` na Claude API včetně tool use nad historií
- Převod Apple Health a Strava importu do databáze
