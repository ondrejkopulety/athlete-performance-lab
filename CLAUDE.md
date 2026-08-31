# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Zbytek souboru je česky, protože česky je celý repozitář — dokumentace,
docstringy, komentáře, commit messages i texty v UI. Piš tak i ty.

---

## Co to je

Osobní datová pipeline nad cyklistickým tréninkem a denní biometrií z Garmin
Connect (+ Strava). Stáhne data, dekóduje FIT, spočítá sportovní metriky
(TRIMP/CTL/ATL/TSB, regenerace z HRV, cardiac drift, LTHR, tepová křivka,
souvislé bloky nad prahem), naservíruje je přes FastAPI a vykreslí v React
dashboardu. Navíc dává strukturovaný JSON kontext pro AI trenéra.

**Jeden atlet, jeden kalibrovaný model.** Není to produkt pro veřejnost, takže
se nikde nezjednodušuje kvůli univerzálnosti — konstanty v `config/settings.py`
jsou naměřené, ne odhadnuté.

| Dokument | Co v něm je |
|---|---|
| [README.md](README.md) | **Proč** se metriky počítají takhle. Čti před sáhnutím na `src/analytics/` nebo `src/physio/`. |
| [NAVOD.md](NAVOD.md) | Provozní kuchařka — co napsat do terminálu, co dělat, když něco spadne. |
| [docs/prehled.md](docs/prehled.md) | Vstupní rozcestník + vodítka, podle kterých se to stavělo. |
| [docs/design.md](docs/design.md) | Vizuální jazyk dashboardu, mapování mockupu na `frontend/src`. |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Nasazení (Docker + Traefik + Authentik). |
| [data/README.md](data/README.md) | Co je v `data/` vstup, co výstup, co zálohovat. |

---

## Příkazy

Python **3.12** (ne 3.14 — viz NAVOD.md). Všechno se pouští z kořene projektu
s prefixem `.venv/bin/`, aby se neřešila aktivace prostředí.

### Databáze a migrace

```bash
docker compose up -d db                  # Postgres 16 + TimescaleDB (kontejner garmin_db)
.venv/bin/alembic upgrade head
.venv/bin/alembic downgrade -1
```

Migrace se **píšou ručně**, ne přes `alembic revision` — soubor se jmenuje
`NNNN_popis.py` a `revision`/`down_revision` jsou stringy s pořadovým číslem
(`"0008"` / `"0007"`), ne vygenerované hashe. Drž tu konvenci.

### Pipeline

```bash
.venv/bin/python scripts/main.py                  # celý řetězec sync→import→load→analyze→hr→export
.venv/bin/python scripts/main.py status           # co je v databázi (začni tímhle, když něco nesedí)
.venv/bin/python scripts/main.py analyze          # jen přepočet metrik
.venv/bin/python scripts/main.py load analyze     # kroky lze kombinovat
.venv/bin/python scripts/main.py --skip-download  # bez sítě, jen nad lokálními daty
.venv/bin/python scripts/main.py load --force            # přeparsuj všechny FIT (ignoruj hashe)
.venv/bin/python scripts/main.py analyze --force-metrics # přepočítej i aktuální verze
.venv/bin/python scripts/main.py splits           # CSV po jednotlivých jízdách (není v `all`)
.venv/bin/python scripts/main.py --json           # strojové shrnutí místo logu
```

Kroky: `sync`, `load`, `analyze`, `hr`, `export`, `splits`, `status`, `all`.

### Předvýpočty nad vteřinovými daty

```bash
.venv/bin/python -m src.physio.cli hr             # tepová křivka, bloky, pokrytí, tep po vzdálenosti
.venv/bin/python -m src.physio.cli hr --force     # přepočítej vše (~10 s / 800 aktivit)
.venv/bin/python -m src.physio.cli hr --since 2026-01-01 --dry-run
.venv/bin/python -m src.physio.cli rr data/fit    # R-R intervaly z FIT (--write-db zapíše posudek)
```

`hr` čte tabulku `records` (po sloučení fragmentů a kanonizaci sportu), `rr`
musí z FIT — `hrv` zprávy v databázi nejsou. Stejný krok běží i jako součást
`scripts/main.py`, takže ruční spuštění je potřeba jen pro filtry a `--force`.

### API a dashboard

```bash
.venv/bin/uvicorn src.api.app:app --reload        # http://localhost:8000/docs

cd frontend
npm run dev                                        # :5173, /api proxuje na :8000 (backend musí běžet zvlášť)
npm run build                                      # tsc -b + produkční build
npm run typecheck                                  # jen typy
node scripts/render-check.mjs                      # SSR nad ŽIVÝM API, všechna období a oba motivy
```

`render-check.mjs` je hlavní pojistka frontendu: vykreslí `Dashboard` v Node nad
reálnou odpovědí `/api/dashboard` a hlídá, že v HTML není `NaN`/`undefined`.
Chytá to, co typy nechytí — dělení nulou v prázdném období, chybějící
biometrii, jediný den v okně. Backend při něm musí běžet.
Jiné API: `VITE_API_TARGET=http://jiny-host:8000 npm run dev`.

### Testy a lint

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_load.py -q
.venv/bin/python -m pytest tests/test_load.py::test_name
```

Testy, které potřebují databázi, se **samy přeskočí**, když neběží (fixture
`session` v `tests/conftest.py`). Zelené testy tedy neznamenají, že se to celé
prohnalo daty — u změn v analytice nastartuj DB.

Ruff je nakonfigurovaný v `pyproject.toml` (line-length 100, `E F W I UP B`),
ale **v `.venv` nainstalovaný není**. Když chceš lintovat:
`.venv/bin/pip install ruff && .venv/bin/ruff check .`

---

## Architektura

### Pipeline

```
SYNC ──▶ IMPORT ──▶ LOAD ──▶ ANALYZE ──▶ HR ──▶ EXPORT
```

| Krok | Modul | Co dělá |
|---|---|---|
| SYNC | `src/ingestion/garmin_sync.py` | CSV souhrny + FIT z Garmin Connect na disk |
| IMPORT | `src/ingestion/biometrics_import.py` | denní CSV (HRV, spánek, RHR, stres) → `daily_biometrics` |
| LOAD | `src/ingestion/loader.py`, `dedup.py` | dedup Garmin vs. Strava, zápis do `activities` + `records` |
| ANALYZE | `src/analytics/pipeline.py` | per-activity fyziologie (inkrementálně) + denní metriky (vždy celé) |
| HR | `src/physio/` | tepová křivka, souvislé bloky, pokrytí, tep po vzdálenosti |
| EXPORT | `src/analytics/exports.py` | CSV z databáze |

Celý řetězec je definovaný **jednou** v [src/pipeline.py](src/pipeline.py) a
volá ho jak CLI (`scripts/main.py`), tak API (`POST /api/sync/run`). Nepřidávej
druhou variantu — rozejdou se.

SYNC smí selhat: pipeline pokračuje nad lokálními daty a jen to zaloguje. Jediná
výjimka je HTTP 429, který propaguje nahoru a běh zastaví (další requesty by ban
prodloužily).

### Inkrementalita — hlavní návrhové omezení

Denní běh trvá jednotky sekund, protože se přepočítává jen to, co se změnilo:

- FIT soubory se identifikují **SHA-256 obsahu** — nezměněný soubor se
  neotevře.
- Odvozená data nesou **verzi výpočtu**. Řádek s nižší verzí je zastaralý a
  přepočítá se.
- Denní metriky se počítají **vždy celé**. Je to levné (~1 s / 1650 dní) a
  nutné: `ef_trend` používá `ffill` bez omezení a 42denní EMA nese vliv řádově
  rok zpátky, takže žádné konečné lookback okno nedá přesný výsledek.

Změnil jsi vzorec? **Zvyš verzi v `config/settings.py`**, pipeline si přepočet
udělá sama:

| Konstanta | Co zastaraví |
|---|---|
| `ACTIVITY_METRICS_VERSION` | `activity_metrics` — per-activity fyziologie |
| `HR_CURVE_VERSION` | `activity_hr_curve` |
| `HR_BLOCKS_VERSION` | `activity_hr_blocks` |
| `HR_DISTANCE_VERSION` | `activity_hr_by_distance_bucket` |
| `DAILY_METRICS_VERSION` | jen razítko na řádku — denní metriky se počítají celé vždy |

U komentáře k verzi vždy dopiš, **co** se v ní změnilo (viz stávající zápisy) —
je to jediná stopa, která z dat řekne, jakou logikou které číslo vzniklo.

Past: cache HR kroku se posuzuje podle `activity_hr_blocks` (a křivky). Když
přibude **nová** HR tabulka, existující aktivity se dál tváří jako hotové —
prvotní zaplnění historie chce jeden běh s `--force`.

### Datový model

| Tabulka | Obsah |
|---|---|
| `activities` | Surová fakta z FIT — nikdy se nepřepočítávají |
| `activity_metrics` | Odvozené per-activity metriky + `metrics_version` |
| `records` | Vteřinová data, TimescaleDB hypertable (chunk 7 dní, komprese po 30) |
| `activity_hr_curve` | Max. průměrný tep pro okna 5 s – 60 min |
| `activity_hr_blocks` | Souvislé bloky nad prahem, dvě tolerance přemostění (0 a 15 s) |
| `activity_hr_coverage` | Na jak úplných datech křivka a bloky stojí |
| `activity_hr_by_distance_bucket` | Průměrný tep po 5 km — jediná metrika s osou vzdálenosti, ne času |
| `athlete_threshold` | Historie nastaveného LTHR / max. tepu (append-only) |
| `daily_biometrics` | Denní vstupy z Garminu (HRV, spánek, RHR, stres) |
| `daily_metrics` | Výstup analytiky, jeden řádek na den |
| `sync_state` | Kde skončil poslední běh |

Oddělení surových a odvozených tabulek je to, co dělá inkrementalitu možnou:
změna vzorce je bump verze, ne přepis vstupů.

### Pravidla, která platí napříč kódem

Tohle jsou rozhodnutí, ne preference — když je porušíš, rozbiješ víc než
jeden soubor:

1. **Surová data se nepřepisují.** `activities` a `records` jsou fakta z FIT.
   Všechno odvozené má vlastní tabulku a verzi.
2. **API nepočítá.** Routery jen servírují, co spočítala pipeline — proto jsou
   odpovědi v milisekundách. Nová metrika = nový sloupec v pipeline, ne výpočet
   v routeru. (`src/analytics/hr_panels.py` skládá čísla z **předpočítaných**
   řádků; ani ten nesahá na sekundová data.)
3. **Chybějící ≠ nula.** `recovery_time_h` před 8/2025 zůstává `NULL`, ne
   dopočítaný odhad, který by v exportu vypadal jako měření. Okno křivky, které
   nevyšlo, vrací `None`, ne 0 — „nikdo nejel hodinu naplno“ a „jel jsem hodinu
   na nule“ jsou různá tvrzení. Totéž platí pro glosář, který jde do promptu AI
   trenéra, aby si model chybějící data nevymýšlel.
4. **Ukládají se prahy, ne zóny.** Bloky leží na mřížce absolutních prahů
   (135–185 bpm po 5). Zóna se odvodí z LTHR až při dotazu a zaokrouhlí na
   nejbližší práh — změna LTHR mění **dotaz, ne data**, takže nespouští
   přepočet historie. Tepová křivka je na LTHR nezávislá úplně.
5. **Ze sekundových dat se počítá dopředu, ne za běhu.** Jediná dokumentovaná
   výjimka je pozicování nejdelšího bloku v grafu detailu jízdy — a i tam je
   *číslo* vždycky z `activity_hr_blocks`, dopočítává se jen, KDE v grafu leží.
6. **`config/settings.py` je jediný zdroj konstant.** Žádné natvrdo psané prahy
   ve skriptech. Naměřené zóny mají přednost před `lthr_estimate`, který je
   reference a dolní mez, ne zdroj zón.

### Dedup Garmin vs. Strava (`src/ingestion/dedup.py`)

Tatáž aktivita bývá na disku dvakrát (hodinky + export ze Stravy), páruje se
v okně ±30 min. Priorita: **R-R intervaly** (nenahraditelné, Strava je zahazuje)
→ **pojistka integrity** (>25 % víc záznamů vyhrává, chytá oříznuté soubory) →
**hrudní pás > optika** bez ohledu na zdroj → **HR density ≥ 90 %** → větší
soubor při naprosté shodě. Rozhodnutí se propíše do `activities.source`.

### Kalendář (`src/analytics/calendar.py`)

Denní osa **vždy končí dneškem** a **začíná nejstarším záznamem z jakéhokoli
zdroje** (aktivita nebo biometrie) — obojí je potřeba nezávisle, jinak se ztrácí
buď ranní biometrie ve dnech bez tréninku, nebo biometrie z doby před prvním
tréninkem. Den bez tréninku má `trimp = 0` (skutečná hodnota); biometrické
sloupce zůstávají `NULL`, dokud data nedorazí.

### API (`src/api/`: `app.py` + `routers/` + `schemas.py`)

- `/api/dashboard` — **jediný** požadavek, který dashboard dělá. Celá denní
  historie ve **sloupcovém tvaru** (pole polí, ne objekty — 1650 dní × 50 klíčů
  by byl zbytečně tučný payload), jízdy s minutami v zónách a `last_known`
  fallback (s datem) pro ranní biometrii, která ještě nedorazila.
- `/api/hr/curve`, `/api/hr/blocks` — odděleně od dashboardu, protože jsou to
  agregace přes zvolené období a přes proměnlivý filtr (`complete_only`, práh
  z LTHR), který mění, které řádky do agregace vůbec vstoupí.
- `/api/profile/threshold` (GET/PUT) — nastavený LTHR jako append-only historie.
  PUT nic nepřepočítává, viz pravidlo 4.
- `/api/activities/{id}` + `/records`, `/hr-curve`, `/hr-blocks`,
  `/hr-by-distance` — podklad pro detail jízdy. `records` se agreguje přes
  TimescaleDB `time_bucket`, syrové desítky tisíc bodů do prohlížeče nepatří.
- `/api/coach/context` + `/api/coach/glossary` — JSON pro LLM. Glosář se
  generuje z `METRIC_META` v settings, aby model věděl, že u `hrv_cv_pct` je
  nižší lepší a ACWR má sweet spot uprostřed, ne na kraji. Statická část
  (profil + glosář) je stabilní napříč dny → prompt caching.
- `POST /api/sync/run` + `GET /api/sync/status` — pipeline na pozadí. Noční běh
  přes APScheduler, zapíná se `SYNC_CRON_HOUR` v `.env`.

### Frontend (`frontend/`, React + Vite + TS, žádný další framework)

Přepis návrhu z Claude Design (`Fitness Dashboard design/Readiness
Dashboard.dc.html`, v repu jako vizuální reference — data v něm jsou fiktivní).

- **`App.tsx` (data + motiv) vs. `Dashboard.tsx` (čistý pohled)** není kosmetika:
  díky tomu jde `Dashboard` vykreslit v Node bez prohlížeče, což je celý základ
  `render-check.mjs`. Nepřesouvej fetch do `Dashboard`.
- **`derive/*.ts`** — výpočty vytažené z JSX, jeden soubor na graf/sekci
  (`pmc`, `gauges`, `quality`, `climb`, `hrr`, `rides`, `ranges`, `hrcurve`,
  `hrblocks`, `activityDetail`, `metricHistory`).
- **`theme.ts` a `styles.css` musí zůstat 1:1.** CSS proměnné řeší statické
  barvy, JS paleta ty, které se počítají za běhu (SVG stroke/fill). Když se
  rozejdou, graf barevně nesedí se zbytkem stránky.
- **`useRoute.ts`** — vlastní mini-router nad History API, žádný react-router.
  Detail jízdy (`/activity/{id}`) se vykresluje **nad** dashboardem, takže se
  kvůli němu nikdy neodmountuje a tlačítko Zpět nevynuluje zvolené období.
  Router už umí i `?metric=…` pro drilldown na historii metriky
  (`derive/metricHistory.ts`, `components/MetricHistoryChart.tsx`,
  `MetricValue.tsx`) — ty jsou hotové, ale zatím je nic neimportuje.
- Formátování je české (`cs-CZ`, desetinná čárka, `−` pro záporná čísla,
  sklonění „jízda/jízdy/jízd“) — `format.ts`.

### Ostatní `src/`

- `src/coach/context.py` — payload pro `/api/coach/context`.
- `src/ingestion/sport.py` — kanonický tvar `"hlavní/pod"` pro `activities.sport`.
  Jediné místo, které o tom rozhoduje; dřív existovaly tři tvary téhož sportu.
  Taky `is_ebike()`: **elektrokolo se počítá do zátěže a formy** (TRIMP → CTL/ATL/
  TSB, minuty v zónách), **ne do objemu cyklistiky** (km, převýšení, počet jízd)
  ani do metrik stojících na poměru výkon↔tep — efektivita (TRIMP/km), cardiac
  drift, DFA prahy, odhad LTHR, tepová křivka, HRR, VAM. Rozpozná se z podřetězce
  `e_bike` v kanonickém sportu, takže každá další jízda z e-bike profilu sedne
  správně sama.
- `src/botanical/`, `src/core/`, `src/reporting/` — analýza zastávek a hotspotů,
  oddělená od tréninkové pipeline. Ještě čte CSV, ne databázi.
- `scripts/setup/` — jednorázové (migrace CSV→DB, seed tokenu, preflight).
  `scripts/legacy/` — staré CSV skripty, které pipeline neaktualizuje.

---

## Konvence

- **Česky**: docstringy, komentáře, dokumentace, texty v UI i commit messages.
  Commity jsou conventional commits s českým předmětem:
  `feat(dashboard): pokrytí dat, tepová křivka a souvislé bloky na webu`.
- **Docstring vysvětluje PROČ, s konkrétními čísly z dat.** To je domácí styl a
  je to hodnotnější než popis toho, co kód dělá. Viz komentáře u
  `HR_GRID_FFILL_LIMIT_S` nebo `RHR_BASELINE_LONG_DAYS` v settings — každá
  konstanta má u sebe měření, ze kterého vyšla. Drž to.
- `E402` je v ruffu povolené pro `src/`, `scripts/` a `alembic/`, protože si
  moduly musí přidat kořen projektu na `sys.path` dřív, než importují `config`.
- Tajemství (Garmin heslo, GPS souřadnice domova) patří do `.env`, ne do repa —
  viz `EXCLUDED_LOCATIONS`.

## Pasti

- **Nepouštěj Garmin sync opakovaně.** Dva souběžné běhy nebo opakované pokusy
  po 429 vedou k banu na API.
- **Nesahej na `data/_baseline/`.** Je to zmrazený výstup původní implementace
  pro `tests/test_parity.py`. Když ho přepíšeš živým exportem, test začne
  porovnávat výstup sám se sebou a přestane cokoli hlídat.
- **Needituj CSV v `data/summaries/`** — nejbližší běh pipeline je přepíše.
  Zdrojem pravdy je databáze, CSV jsou export.
- **Zálohovat stačí `data/fit/`** (a `data/Apple/`). Databáze i všechna CSV se
  z FIT vyrobí znovu (`scripts/main.py load --force`), FIT ne.
- Když API ukazuje stará čísla, nepustil se pipeline — API nic nepočítá.
