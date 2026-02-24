# Garmin Training Analytics

Advanced training load management & physiological analysis platform.  
Pulls data from **Garmin Connect**, parses raw **.FIT** files, and computes
research-grade metrics: ACWR, cardiac drift, DFA-α1, durability, fueling
models, illness warning, and more.

---

## Project Structure

```
garmin/
├── main.py                  # Central entry point – runs full pipeline
├── requirements.txt         # Python dependencies
├── .env.example             # Template for API credentials
├── .gitignore               # Protects personal data from Git
│
├── config/
│   ├── __init__.py
│   └── settings.py          # All athlete parameters & paths (single source of truth)
│
├── src/
│   ├── garmin_to_csv.py     # Step 1 – Garmin Connect API sync
│   ├── fit_to_highres_csv.py# Step 2 – FIT → per-second CSV extraction
│   ├── master_rebuild.py    # Step 3 – Garmin + Strava deduplication & merge
│   ├── athlete_analytics.py # Step 4 – 23-module physiology & load analysis
│   └── strava_original_export.py  # Strava bulk FIT export (optional)
│
├── data/                    # ⚠️  NOT tracked by Git – create manually
│   ├── raw/                 # Raw JSON responses from Garmin API
│   ├── fit/                 # Downloaded .fit files (Garmin + Strava)
│   ├── summaries/           # Intermediate CSVs (activities, HRV, sleep…)
│   └── processed/           # Final databases (master_high_res_summary.csv…)
│
├── reports/                 # Generated charts & PDFs
└── logs/                    # Runtime logs
```

> **Important:** The `data/` folder contains personal training data and is
> excluded from version control by `.gitignore`. After cloning, create it
> yourself and place your `.fit` files into `data/fit/`.

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url> garmin
cd garmin
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in:
#   GARMIN_EMAIL=your@email.com
#   GARMIN_PASSWORD=your_password
#   GARMIN_DISPLAY_NAME=your-uuid
```

### 3. Configure athlete profile

Open `config/settings.py` and set your values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_HR` | 199 | Maximum heart rate (bpm) |
| `RESTING_HR` | 41 | Resting heart rate (bpm) |
| `ZONE_PCTS` | `[0.50, 0.60, 0.72, 0.82, 0.90, 1.00]` | Karvonen HR-Reserve zone boundaries |
| `ZONE_2_CAP` | 155 | Talk-Test Z2 ceiling (bpm) |

### 4. Create data folder & add FIT files

```bash
mkdir -p data/{raw,fit,summaries,processed}
# Copy your .fit files into data/fit/
```

### 5. Run the pipeline

```bash
# Full pipeline: sync → parse → merge → analyze
python main.py

# Or run individual steps:
python main.py sync       # Download from Garmin Connect
python main.py parse      # Parse FIT files → CSV
python main.py merge      # Deduplicate Garmin + Strava
python main.py analyze    # Compute analytics & readiness

# Chain specific steps:
python main.py parse merge analyze
```

---

## How It Works

### Pipeline Overview

```
Garmin Connect API ──→ garmin_to_csv.py ──→ data/summaries/*.csv
                                            data/fit/*.fit
                                                │
Strava Export (opt.) ──→ data/fit/strava_originals/*.fit
                                                │
                         ┌──────────────────────┘
                         ▼
              fit_to_highres_csv.py  →  per-second HR/speed/power data
                         │
                         ▼
              master_rebuild.py     →  deduplicated master database
                         │              (chest strap > optical > density)
                         ▼
              athlete_analytics.py  →  athlete_readiness.csv
                                       23 physiological modules (A–W)
```

### Load Management – ACWR (Acute : Chronic Workload Ratio)

The system implements the **coupled ACWR model** using EPOC-weighted TRIMP:

- **Acute Load (ATL):** 7-day exponential moving average of daily training load
- **Chronic Load (CTL):** 28-day exponential moving average (fitness)
- **ACWR = ATL / CTL** — the "sweet spot" is **0.8–1.3**

| ACWR Range | Interpretation |
|------------|----------------|
| < 0.8 | Detraining – load too low |
| 0.8 – 1.3 | Sweet spot – optimal adaptation |
| 1.3 – 1.5 | Danger zone – injury risk rising |
| > 1.5 | High risk – reduce load immediately |

**CTL Ramp Rate** is also monitored: an increase > 8 points/week triggers a
burnout warning.

### Durability Model

Measures how well aerobic efficiency (EF = speed / HR) holds up over
prolonged efforts (> 2 hours). Compares EF in the first vs. second half:

- **EF Decay < 5 %** → Excellent durability (elite-level)
- **EF Decay 5–10 %** → Good – keep building long rides/runs
- **EF Decay > 10 %** → Needs work – more Z2 volume recommended

### Full Module List (A–W)

| Module | Metric | Description |
|--------|--------|-------------|
| A | Cardiac Drift | Aerobic decoupling (Pa:HR ratio) |
| B | HR Recovery | Max HR drop in 60 s post-peak |
| C | Monotony & Strain | 7-day training variability + Whoop log-strain |
| D | Efficiency Index | TRIMP per kilometre |
| E | Bio-Readiness | 0–100 % composite score |
| F | Recovery Score | HRV + RHR + Sleep fusion |
| G | Illness Warning | Multi-indicator alert system |
| H | Coach Advice | Daily text recommendation |
| I | Polarization | 80/20 zone distribution (14-day) |
| J | VAM | Rate of ascent (m/h) |
| K | ACWR | Acute:Chronic workload ratio |
| L | Fatigue Index | External vs. internal load |
| M | Fueling Model | Fat/carb kcal split by HR zone |
| N | Fluid Loss | Estimated sweat rate |
| O | Temp Effect | Heat vs. cardiac drift |
| P | Durability | EF decay in long activities |
| Q | TTE | Time to exhaustion estimate |
| R | EPOC | Oxygen debt & sleep need |
| S | DFA-α1 Proxy | Aerobic threshold breakpoint |
| T | Climb Score | Gradient + climb category |
| U | DFA-α1 Real | R-R based AeT/AnT (neurokit2) |
| V | Respiration Rate | RSA via Welch PSD on R-R |
| W | TATI / Critical HR | Monod-Scherrer HR model |

---

## Configuration Reference

All parameters live in **`config/settings.py`** — the single source of truth.  
No hardcoded values exist in the processing scripts.

### Key sections:

- **Athlete Profile** – MAX_HR, RESTING_HR, zone boundaries
- **Training Load Model** – CTL/ATL windows, ACWR, TRIMP constants
- **Deduplication** – tolerance window, HR density threshold
- **Readiness Thresholds** – HRV drop, RHR ceiling, sleep minimums
- **DFA-α1** – window beats, AeT/AnT thresholds
- **File Paths** – all CSVs and directories

---

## Privacy

This project processes **personal health and training data**. The `.gitignore`
is configured to prevent accidental upload of:

- `.env` (API credentials)
- `data/` (all training data, FIT files, CSVs)
- `reports/` (generated outputs)
- `.garminconnect/`, `.garth/` (auth tokens)

**Never commit your `.env` file or `data/` folder to a public repository.**

---

## License

Private project. All rights reserved.
