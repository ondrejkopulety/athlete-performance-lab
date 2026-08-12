# Garmin Training Analytics

Automatizovaná datová pipeline pro analýzu cyklistického tréninku a denní
biometrie z ekosystému Garmin. Stahuje data z Garmin Connect, dekóduje FIT
soubory, počítá sportovní metriky a servíruje je přes REST API — včetně
strukturovaného kontextu pro AI trenéra.

---

## Rychlý start

Tenhle soubor vysvětluje **proč** se věci počítají tak, jak se počítají.
Kuchařka na každodenní provoz (co napsat do terminálu, co dělat, když
něco spadne) je v [NAVOD.md](NAVOD.md).

```bash
# 1. Závislosti
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Konfigurace
cp .env.example .env        # doplň Garmin credentials

# 3. Databáze (PostgreSQL 16 + TimescaleDB)
docker compose up -d db
.venv/bin/alembic upgrade head

# 4. Jednorázový import historických CSV (pokud je máš)
.venv/bin/python scripts/setup/migrate_csv_to_db.py

# 5. Běh pipeline
.venv/bin/python scripts/main.py

# 6. API
.venv/bin/uvicorn src.api.app:app --reload   # http://localhost:8000/docs

# 7. Dashboard
cd frontend && npm install && npm run dev    # http://localhost:5173
```

Nasazení na server (Docker + Traefik + Authentik) popisuje
[docs/DEPLOY.md](docs/DEPLOY.md).

---

## Pipeline

```
SYNC ──▶ IMPORT ──▶ LOAD ──▶ ANALYZE ──▶ HR ──▶ EXPORT
```

| Krok | Co dělá |
|---|---|
| **SYNC** | Stáhne novinky z Garmin Connect (CSV souhrny + FIT soubory na disk) |
| **IMPORT** | Přetaví denní CSV (HRV, spánek, RHR, stres) do `daily_biometrics` |
| **LOAD** | Deduplikuje Garmin vs. Strava FIT soubory a zapíše je do `activities` + `records` |
| **ANALYZE** | Per-activity metriky (inkrementálně) + denní metriky (PMC, regenerace, kvalita) |
| **HR** | Tepová křivka, souvislé bloky nad prahem a pokrytí dat (inkrementálně) |
| **EXPORT** | CSV z databáze, aby nezastarávaly pod rukama |

```bash
python scripts/main.py                  # celá pipeline
python scripts/main.py analyze          # jen přepočet metrik
python scripts/main.py export           # CSV exporty z databáze
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
TSB s jednodenním posunem (ranní forma *před* dnešním tréninkem). TRIMP se
počítá z klidového tepu **platného k datu aktivity** (90denní klouzavý
medián), ne z pevné konstanty — stejný tep při RHR 44 a 52 znamená jinou
relativní zátěž.

**Prevence zranění** — ACWR (7 d / 28 d z EPOC-vážené TRIMP, bez clipování),
CTL ramp rate.

**Kvalita tréninku** — Fosterova monotonie a strain, Whoop logaritmický
strain (0–21), Seilerova polarizace 80/20 s penalizací za Z3 „junk miles".

**Regenerace a biometrie** — Pure Recovery Score (HRV 40 % + RHR 30 % +
spánek 30 %), dual-era Bio-Readiness, sleep performance (Whoop-style),
7denní HRV CV, celodenní Garmin stres, multi-indikátorový illness warning.

**Regenerační čas** — `recovery_time_h` je Garminova (Firstbeat) vlastní
hodnota z hodinek, ne náš odhad. Vedle ní se importuje i Garminovo
`garmin_readiness_score`. Obojí existuje **až od 8. 8. 2025**; pro
starších 2784 dní zůstává NULL a platí tam ATL, TSB a strain.

Dopočítávat starou éru modelem by z odhadu udělalo něco, co v exportu
vypadá stejně jako měření — proto raději NULL. Glosář to říká i chatbotovi,
aby na dotaz o roce 2022 číslo nevymýšlel.

Nahradilo to zrušenou `recovery_tax_hours = min(96, 0,08 × TRIMP^1,2)`.
Ta byla vymyšlená a proti Garminovi obstála takhle:

| | |
|---|---|
| Spearman s `total_trimp` | **0,995** — jen přeznačkovaný TRIMP |
| RMSE proti Garminu | **32,0 h** |
| RMSE nejlepší možné konstanty | 32,1 h — *k nerozeznání* |
| RMSE přeškálovaného `atl` | **26,4 h** — jasně lepší, a už ho máme |

Zajímavé je **jak** selhávala: ne přeháněním, jak by se čekalo od stropu
96 h. Dnů, kdy tvrdila ≥ 48 h a Garmin < 24 h, bylo **nula**. Zato ve
**40 dnech** tvrdila < 6 h, zatímco Garmin hlásil ≥ 48 h. Byl to totiž
**tok** (co přidal dnešek, ve dnech volna nula), zatímco regenerace je
**zásoba**, která dobíhá i když neseš nohy z postele. Medián 1,1 h proti
Garminovým 26,6 h. Rozdíl přes 24 h mělo 36 % dní.

Garminovo readiness se s naším shoduje jen zčásti — a to je informace,
ne chyba (365 společných dní, Pearson):

| proti `garmin_readiness_score` | |
|---|---|
| naše `readiness_score` | +0,47 |
| `atl` | −0,47 |
| `tsb` | +0,34 |
| `pure_recovery_score` | **+0,28** |

Nejnižší shoda je s `pure_recovery_score`, což dává smysl: ten stojí jen
na naměřené biometrii (HRV, RHR, spánek) a o trénink se vůbec neopírá,
kdežto Garmin do svého skóre zátěž započítává. Že se rozcházejí, je tedy
očekávané. **Zatím neověřené** je, které z nich lépe předpovídá skutečný
výkon — na to by bylo potřeba srovnat je s výsledky tréninků.

**Per-activity fyziologie** — cardiac drift (Pa:HR decoupling), maximální
pokles tepu za 60 s, durabilita, VAM, dechová frekvence z RSA, EPOC,
Critical HR a TATI, fueling model, odhad ztráty tekutin.

**Prahový tep** — `lthr_estimate` = 0,95 × nejlepší 20minutový průměr tepu
za 180 dní. Na těchto datech dává 172 bpm, shodně s laktátovým testem.
Je to **reference, ne zdroj zón** — zóny v `settings.py` jsou naměřené
a mají přednost. Zároveň je to dolní mez: po období bez intenzity klesne,
aniž by se práh zhoršil.

DFA-alpha1 je v kódu, ale na těchto datech nedává použitelné prahy —
alpha1 zůstává nad 1,2 i při tepu nad prahem a R-R intervaly má jen 73
z 866 aktivit (Strava je v exportech zahazuje, Garmin je v provozu až
od 8/2025). Běží s diagnostikou, aby bylo z dat vidět, kdy se to změní.

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
GET  /api/dashboard                    kompletní podklad pro webový dashboard
GET  /api/coach/context?date=          strukturovaný JSON pro LLM
GET  /api/coach/history/{metric}?days= časová řada jedné metriky
GET  /api/coach/glossary               význam a jednotky metrik
POST /api/sync/run                     spustí pipeline na pozadí
GET  /api/sync/status                  stav posledního běhu
```

Vteřinová data se agregují přes TimescaleDB `time_bucket` — aktivita má
desítky tisíc bodů a syrová data nemá smysl posílat do prohlížeče.

`/api/dashboard` je jediný požadavek, který dělá webový dashboard: celá
historie denních metrik ve sloupcovém tvaru (pole polí, ne objekty — 1650 dní
krát 50 klíčů by byl zbytečně tučný payload), cyklistické aktivity s minutami
v zónách, stoupáním a zotavovacím tepem, posledních pět jízd a dnešní
biometrie. Navíc `last_known`: poslední den v kalendáři často ranní biometrii
ještě nemá (Garmin ji doplní později), takže se k HRV, klidovému tepu a
spánku posílá i poslední naměřená hodnota s datem — jinak by dashboard místo
čísel ukazoval pomlčky.

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
| `activity_hr_curve` | Tepová křivka — max. průměrný tep za 5 s až 60 min |
| `activity_hr_blocks` | Souvislé bloky nad prahem — jak dlouho tep vydrží v kuse |
| `records` | Vteřinová data, TimescaleDB hypertable (chunk 7 dní, komprese po 30) |
| `daily_biometrics` | Denní vstupy z Garmin Connect (HRV, spánek, RHR, stres) |
| `daily_metrics` | Výstup analytiky — jeden řádek na den |
| `sync_state` | Kde skončil poslední běh |

Oddělení surových a odvozených dat je to, co dělá inkrementalitu možnou:
změna vzorce se projeví bumpnutím verze, aniž by se sáhlo na vstupy.

### Tepová křivka a souvislé bloky

Dvě metriky, které se z vteřinových dat nedají počítat za běhu, takže se
předpočítávají:

```bash
python -m src.physio.cli hr              # co ještě nemá výsledky
python -m src.physio.cli hr --force      # přepočítej všechno (~10 s / 800 aktivit)
```

**Ukládají se prahy, ne zóny.** Zóny se odvozují z LTHR, které se mění
(172 → 177 → po terénním testu znovu). Kdyby v tabulce byly zóny, každá
změna prahu by znamenala přepočet celé historie. Takhle je zóna lookup:
„Z4 při LTHR 177" = práh 168 → nejbližší řádek na mřížce 135–185 po 5 bpm.
Změna prahu mění dotaz, ne data. Tepová křivka je na LTHR nezávislá úplně.

`activity_hr_blocks` odpovídá na otázku, kterou „čas v zónách" nezodpoví:
131 minut nad prahem může být 272 úseků s mediánem 6 sekund, nebo sedm
dvacetiminutových bloků. Ukládají se obě varianty přemostění vedle sebe
(`bridge_tolerance_s` 0 a 15 s), protože rozdíl mezi nimi je sám o sobě
informace o charakteru jízdy.

Obojí teče do CSV standardním exportem: křivka rozvinutá do sloupců
`hr_curve_*` v `master_high_res_summary.csv`, bloky v dlouhém formátu
v `hr_blocks.csv`.

---

## Deduplikace Garmin vs. Strava

Tatáž aktivita bývá na disku dvakrát — jednou z hodinek, jednou ze Stravy.
`src/ingestion/dedup.py` v okně ±30 minut rozhodne, který soubor je lepší:

1. **R-R intervaly** — soubor, který je má, vyhrává. Strava je ve svých
   exportech zahazuje úplně, takže jde o nenahraditelná data; vteřinová
   data se dají interpolovat, R-R ne.
2. **Pojistka integrity** — soubor s >25 % více záznamy vyhrává (druhý je
   oříznutý); Smart Recording se od poškozených dat rozlišuje HR density
3. **Hrudní pás > optika** — ANT+ pás vyhrává bez ohledu na zdroj
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
| `test_parity.py` | Že se neztratil den historie a že posun proti původní implementaci je v očekávaném řádu |
| `test_calendar.py` | Kalendář končí dneškem, začíná nejstarším záznamem, nemá díry |
| `test_metrics.py` | Fyziologické invarianty (rozsahy skóre, součet polarizace, TSB posun) |
| `test_load.py` | Vzorce PMC — EMA konstanta, TSB posun, koeficient pro pěší sporty |
| `test_rhr_baseline.py` | Klidový tep jako časová řada, oddělení zdrojů, vliv na TRIMP |
| `test_rhr_flag.py` | Vlajka klidového tepu relativně k baseline |
| `test_lthr.py` | Odhad prahu z terénních dat vůči laktátovému testu |
| `test_records_merge.py` | Slučování fragmentů vteřinových dat |
| `test_api_dashboard.py` | Že `/api/dashboard` sedí na databázi a do jízd se nevloudí jiný sport |
| `test_recovery_time.py` | Garmin recovery time: převod jednotek a hlavně to, že se předgarminská éra **nedopočítává** |

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
frontend/                 Webový dashboard (React + Vite), viz frontend/README.md
  pipeline.py             Celý běh na jednom místě (sdílí CLI i API)
  analytics/exports.py    CSV exporty z databáze
scripts/
  main.py                 CLI
  setup/                  jednorázové: migrace CSV→DB, tokeny, preflight
  legacy/                 skripty čtoucí CSV, které pipeline neaktualizuje
data/
  fit/                    zdroj pravdy – tohle zálohuj
  _baseline/              zmrazený referenční bod pro testy
  _archive/               osiřelé soubory (~210 MB, lze smazat)
```

---

## Plánované

- Detail jednotlivé jízdy v dashboardu (vteřinová data z `/api/activities/{id}/records`)
- Napojení `/api/coach/context` na Claude API včetně tool use nad historií
- Převod Apple Health a Strava importu do databáze
