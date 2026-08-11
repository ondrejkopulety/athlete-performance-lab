# Návod pro blbečky

Kuchařka na každodenní používání. Proč se věci počítají tak, jak se počítají,
je v [README.md](README.md) — tenhle soubor říká jen **co napsat do terminálu**.

Všechny příkazy se pouštějí z kořene projektu:

```bash
cd ~/garmin
```

---

## TL;DR — 90 % času potřebuješ tohle

```bash
docker compose up -d db          # nastartuj databázi (pokud neběží)
.venv/bin/python scripts/main.py # stáhni novinky a přepočítej všechno
```

Trvá to jednotky sekund. Hotovo.

Pozn.: `.venv/bin/python` píšeš proto, aby ses nemusel starat o aktivaci
virtuálního prostředí. Kdo má rád `source .venv/bin/activate`, může, pak
stačí `python`.

---

## Jednou za život: rozjetí od nuly

Tohle už máš hotové. Je to tu pro případ nového notebooku nebo smazaného `.venv`.

```bash
# 1. Python prostředí (Python 3.12, NE 3.14)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Konfigurace — do .env patří Garmin heslo a GPS souřadnice domova
cp .env.example .env
$EDITOR .env

# 3. Zkontroluj, že je vše připravené (nesahá na Garmin API)
.venv/bin/python scripts/setup/preflight_check.py

# 4. Databáze
docker compose up -d db
.venv/bin/alembic upgrade head

# 5. Přihlášení ke Garminu (uloží token do .garminconnect/)
.venv/bin/python scripts/setup/seed_token.py

# 6. Naplnění databáze z FIT souborů, které máš na disku
.venv/bin/python scripts/main.py --skip-download
```

---

## Denní provoz

### Chci aktuální data

```bash
.venv/bin/python scripts/main.py
```

Udělá celý řetězec: **SYNC** (stáhne z Garminu) → **IMPORT** (biometrie) →
**LOAD** (FIT do databáze) → **ANALYZE** (metriky) → **EXPORT** (CSV).

### Chci jen zjistit, co v databázi je

```bash
.venv/bin/python scripts/main.py status
```

Vypíše počty aktivit, vteřinových záznamů, dnů s biometrií a datum poslední
aktivity. Když ti něco nesedí, začni tímhle.

### Chci se podívat na data v prohlížeči

```bash
.venv/bin/uvicorn src.api.app:app --reload
```

Pak otevři **http://localhost:8000/docs** — je to klikací, endpointy si tam
vyzkoušíš bez psaní URL. Typické dotazy:

```
http://localhost:8000/api/daily/latest          dnešní stav
http://localhost:8000/api/pmc?days=180          forma za půl roku
http://localhost:8000/api/activities?sport=cycling
http://localhost:8000/api/coach/context         všechno pro AI trenéra v jednom JSONu
```

API běží nad databází a **nic nepočítá** — pokud vidí stará čísla, znamená to,
že jsi nepustil pipeline, ne že je rozbité API.

### Chci data v Excelu

Po každém běhu pipeline se samy přepíšou CSV v `data/summaries/`:

| soubor | co v něm je |
|---|---|
| `athlete_readiness.csv` | jeden řádek na den — PMC, regenerace, kvalita |
| `master_high_res_summary.csv` | jeden řádek na aktivitu, včetně odvozených metrik |
| `cycling_summary.csv` | totéž, jen kolo |

Vynutit jen export bez zbytku pipeline:

```bash
.venv/bin/python scripts/main.py export
```

### Chci vteřinová data jednoho tréninku

```bash
.venv/bin/python scripts/main.py splits
```

Vyrobí `data/cycling_splits/` — jeden CSV na cyklo trénink, v něm řádek za
každou vteřinu (tep, rychlost, výška, kadence, GPS, zóna, TRIMP). Název souboru
říká, o co jde: `2026_08_09_bike_73.0km_191min_1021m_23914674353.CSV`.

Není to součást `scripts/main.py` bez argumentů: je to 400+ souborů a půl
giga, což do každodenního běhu nepatří. Zato je to inkrementální — trénink,
který už soubor má, se přeskočí, takže běh trvá sekundy. Když se přepočítaly
metriky a chceš soubory srovnat s databází, přidej `--force`.

---

## Jednotlivé kroky (když nechceš celou pipeline)

```bash
.venv/bin/python scripts/main.py sync      # jen stáhnout z Garminu
.venv/bin/python scripts/main.py load      # jen FIT soubory → databáze
.venv/bin/python scripts/main.py analyze   # jen přepočítat metriky
.venv/bin/python scripts/main.py export    # jen CSV
.venv/bin/python scripts/main.py splits    # jen CSV po jednotlivých trénincích
```

Dají se i kombinovat: `scripts/main.py load analyze`.

Užitečné přepínače:

| přepínač | k čemu |
|---|---|
| `--skip-download` | celá pipeline bez sítě, jen nad tím, co je na disku |
| `--force` | u `load`: přeparsuj **všechny** FIT soubory, ignoruj hashe; u `splits`: přepiš i existující CSV |
| `--force-metrics` | u `analyze`: přepočítej i aktivity s aktuální verzí metrik |
| `--json` | vypiš shrnutí strojově, ne do logu |

---

## Když se něco pokazí

### „Databáze neodpovídá"

```bash
docker compose up -d db
docker ps                      # garmin_db musí být "healthy"
.venv/bin/alembic upgrade head
```

### Garmin sync selhal

Pipeline **pokračuje dál** nad lokálními daty a jen to zaloguje — to je
záměr, ne chyba. Ale pokud potřebuješ nová data:

1. **Vypršel token** → `.venv/bin/python scripts/setup/seed_token.py`
2. **HTTP 429 (rate limit)** → pipeline se sama zastaví, protože další
   requesty by ban jen prodloužily. Počkej hodinu a zkus znovu. Nepouštěj
   sync opakovaně dokola.
3. **Cokoli jiného** → `.venv/bin/python scripts/setup/preflight_check.py`

### Změnil jsem vzorec a čísla se nezměnila

Pipeline je inkrementální — přepočítává jen to, co považuje za zastaralé.
Zvyš `ACTIVITY_METRICS_VERSION` v [config/settings.py](config/settings.py),
pak se dotčené aktivity přepočítají samy. Nebo jednorázově natvrdo:

```bash
.venv/bin/python scripts/main.py analyze --force-metrics
```

### Chci si být jistý, že jsem nic nerozbil

```bash
.venv/bin/python -m pytest tests/ -q
```

Testy, které potřebují databázi, se samy přeskočí, když neběží.

### Logy

Všechno se píše do `logs/main.log` (a zároveň na obrazovku).

---

## Kde co leží

| chci… | je v… |
|---|---|
| změnit prahy, zóny, váhu, FTP | [config/settings.py](config/settings.py) |
| změnit heslo / GPS domova | `.env` (není v gitu) |
| FIT soubory | `data/fit/` — **tohle zálohuj**, nic jiného |
| CSV výstupy | `data/summaries/` |
| logy | `logs/` |

---

## Čtyři pravidla, ať si neublížíš

1. **Zálohuj `data/fit/`.** Databáze i všechna CSV se z FIT souborů dají
   vyrobit znovu (`scripts/main.py load --force`). FIT soubory ne.
2. **Nesahej na `data/_baseline/`.** Je to zmrazený referenční bod pro
   `tests/test_parity.py`. Když ho přepíšeš živým exportem, test začne
   porovnávat výstup sám se sebou a přestane cokoli hlídat.
3. **Needituj CSV v `data/summaries/`.** Nejbližší běh pipeline je přepíše.
   Chceš-li si hrát, zkopíruj si je jinam.
4. **Nepouštěj sync dokola.** Dva souběžné běhy nebo opakované pokusy po
   429 vedou k banu na Garmin API.
