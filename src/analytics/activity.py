"""
activity.py  –  Per-activity fyziologie
========================================

Metriky počítané pro jednu aktivitu. Dělí se na dvě skupiny:

  • **Vektorové** (compute_activity_table) – pracují jen se souhrnnými
    sloupci, spočítají se pro všechny aktivity najednou: fueling, VAM,
    climb score, EPOC, TTE, ztráta tekutin, Critical HR / TATI.

  • **Nad vteřinovými daty** (compute_activity_series_metrics) – cardiac
    drift, max HRR za 60 s, durabilita, DFA-alpha1 a dechová frekvence
    z RSA. Tohle je drahá část: právě kvůli ní byla původní analytika
    minutová záležitost, protože přepočítávala všech 864 aktivit při
    každém běhu.

R-R intervaly se berou z databáze (activity_metrics.rr_intervals_ms),
uložené při načtení FIT souboru – DFA ani RSA už nesahají na disk.

Vzorce jsou převzaté beze změny z athlete_analytics.py.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    DFA_AET_THRESHOLD,
    DFA_ANT_THRESHOLD,
    DFA_WINDOW_BEATS,
    DURABILITY_MIN_DURATION_MIN,
    FAT_PCT_BY_ZONE,
    HR_GRID_FFILL_LIMIT_S,
    HR_PLAUSIBLE_MAX_BPM,
    HR_PLAUSIBLE_MIN_BPM,
    KCAL_PER_MIN_BY_ZONE,
    LTHR_BEST_WINDOWS_MIN,
    MAX_3S_JUMP,
    MAX_REALISTIC_HRR,
    MIN_STARTING_HR,
    TRIMP_K1,
    TRIMP_K2,
    WARMUP_SECONDS,
)
from config.settings import (
    MAX_HR as ATHLETE_MAX_HR,
)
from config.settings import (
    ZONES as ATHLETE_ZONES,
)
from src.ingestion.sport import (
    EBIKE_SPORT_PATTERN,
    cardio_mask,
    is_cardio_sport,
    is_cycling_sport,
)
from src.physio.hr_curve import max_mean_curve
from src.physio.hr_stream import to_second_grid
from src.physio.quality import assess_rr_authenticity

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("analytics.activity")

# ── Volitelné závislosti pro pokročilou fyziologii ─────────────────────────
try:
    import neurokit2 as nk
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import welch

    HAS_NEUROKIT = True
except ImportError:  # pragma: no cover - závisí na prostředí
    HAS_NEUROKIT = False

MIN_DURATION_DRIFT_MIN = 20

# Max HRR: převzorkování na 1 s, interpolace max 30 s díry, okno 60 s
HRR_RESAMPLE_FREQ = "1s"
HRR_INTERPOLATE_LIMIT = 30
HRR_DROP_WINDOW = 60

# Maximální rozptyl nadmořské výšky (m), nad kterým je EF = speed/HR
# znehodnocené terénem (jen běh – u kola terén řeší wattmetr).
CARDIAC_DRIFT_MAX_ALT_RANGE_M = 30.0

KCAL_PER_G_FAT = 9.0
KCAL_PER_G_CARB = 4.0

DFA_SLIDE_BEATS = 30
DFA_BOX_SIZES = list(range(4, 17))

CHR_MIN_ACTIVITIES = 5

# Sporty, kde má VAM smysl. Gravitační sporty (sjezdovka, snowboard) jsou
# vyloučené záměrně – nahoru veze lanovka, takže VAM nic neměří.
VAM_SPORTS = (
    "running", "trail_running", "cycling", "gravel_cycling", "mountain_biking",
    "hiking", "walking", "mountaineering", "alpinism", "rock_climbing",
    "cross_country_skiing", "skate_skiing",
)


def safe_div(a, b, default=np.nan):
    """Dělení odolné vůči nule a NaN."""
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return default
    result = a / b
    if isinstance(result, float) and np.isnan(result):
        return default
    return result


# ═══════════════════════════════════════════════════════════════════════════
# R-R INTERVALY
# ═══════════════════════════════════════════════════════════════════════════

def clean_rr_intervals(rr_ms: list[float] | np.ndarray, max_pct_change: float = 0.20) -> np.ndarray:
    """
    Vyčistí R-R intervaly (v ms) od fyziologicky nemožných hodnot.

    1) ponechá jen 273–2000 ms (tep 30–220), 2) vyhodí ektopické stahy
    s >20% skokem oproti předchozímu.
    """
    arr = np.asarray(rr_ms, dtype=float)
    arr = arr[(arr >= 273) & (arr <= 2000)]
    if len(arr) < 10:
        return arr

    keep = [0]
    for i in range(1, len(arr)):
        if abs(arr[i] - arr[i - 1]) / arr[i - 1] <= max_pct_change:
            keep.append(i)
    return arr[keep]


# ═══════════════════════════════════════════════════════════════════════════
# MODUL A – CARDIAC DRIFT (aerobní decoupling Pa:HR)
# ═══════════════════════════════════════════════════════════════════════════

def cardiac_drift(tdata: pd.DataFrame, sport: str) -> Optional[float]:
    """
    Přeskočí 10min rozjezd, zbytek rozdělí časovým středem na dvě poloviny.

    Kolo s wattmetrem: EF = power/HR (rychlost je na kole funkcí terénu).
    Kolo bez wattmetru i běh: EF = speed/HR, ale jen na rovině – do kopce
          je vztah rychlost↔tep rozbitý gravitací. U běhu navíc jen v Z1/Z2,
          protože intervaly ten vztah rozbíjejí laktátem.

    Bez fallbacku na speed vycházel drift None pro celou databázi jízd bez
    wattmetru (18 z 884 aktivit mělo hodnotu); na rovné jízdě je speed/HR
    použitelný signál decouplingu.

    Drift = (EF_1 − EF_2) / EF_1 × 100 [%]
    """
    if tdata.empty:
        return None

    is_cycling = is_cycling_sport(sport)
    has_power = (
        "power" in tdata.columns
        and pd.to_numeric(tdata["power"], errors="coerce").fillna(0).gt(0).sum() >= 20
    )
    effort_col = "power" if (is_cycling and has_power) else "speed"
    on_speed = effort_col == "speed"
    if effort_col not in tdata.columns:
        return None

    active = tdata.loc[
        (tdata["is_active"] == True)  # noqa: E712 - sloupec může být objekt
        & tdata["heart_rate"].notna()
        & (tdata["heart_rate"] > 0)
        & tdata[effort_col].notna()
        & (tdata[effort_col] > 0)
    ].copy()
    if len(active) < 20:
        return None

    t0 = active["timestamp"].iloc[0]
    elapsed = (active["timestamp"] - t0).dt.total_seconds().values
    post = active.loc[elapsed >= WARMUP_SECONDS]
    if len(post) < 20:
        return None

    # Rovinatost – vždy, když se počítá ze speed (terén plete speed/HR bez
    # ohledu na sport), ne jen u běhu.
    if on_speed and "altitude" in post.columns:
        alt_vals = pd.to_numeric(post["altitude"], errors="coerce").dropna()
        if len(alt_vals) >= 10:
            if float(alt_vals.max() - alt_vals.min()) > CARDIAC_DRIFT_MAX_ALT_RANGE_M:
                return None

    # Ustálený stav Z1/Z2 (jen běh – cyklista jede rovnoměrné tempo i v Z3).
    if on_speed and not is_cycling and "hr_zone" in post.columns:
        post = post.loc[post["hr_zone"].isin(["Z1", "Z2", ""])]
        if len(post) < 20:
            return None

    # Dělení podle času, ne podle počtu řádků – Smart Recording zapisuje
    # nerovnoměrně a dělení po řádcích by poloviny posunulo.
    t_start = post["timestamp"].iloc[0]
    t_end = post["timestamp"].iloc[-1]
    t_mid = t_start + (t_end - t_start) / 2
    first_half = post[post["timestamp"] <= t_mid]
    second_half = post[post["timestamp"] > t_mid]
    if first_half.empty or second_half.empty:
        return None

    hr1, hr2 = first_half["heart_rate"].mean(), second_half["heart_rate"].mean()
    eff1, eff2 = first_half[effort_col].mean(), second_half[effort_col].mean()
    if hr1 == 0 or hr2 == 0:
        return None

    if effort_col == "power":
        if eff1 < 30 or eff2 < 30:  # W – volnoběh nemá vypovídací hodnotu
            return None
    else:
        if eff1 < 2.0 / 3.6 or eff2 < 2.0 / 3.6:  # < 2 km/h
            return None
        # Steady-state pojistka pro speed: decoupling má smysl jen na
        # rovnoměrném úsilí. Když se průměrná rychlost mezi půlkami liší
        # o víc než 25 %, jde o změnu tempa (semafory, skupina, defekt),
        # ne o kardiovaskulární drift – speed/HR je na kole moc citlivý
        # na vítr, drafting a zastávky, než aby se to dalo rozlišit jinak.
        if not (0.75 <= eff2 / eff1 <= 1.334):
            return None

    ef1 = safe_div(eff1, hr1)
    ef2 = safe_div(eff2, hr2)
    if ef1 is None or ef1 == 0 or np.isnan(ef1):
        return None
    drift = round((ef1 - ef2) / ef1 * 100.0, 2)
    # Fyziologicky možný rozsah decouplingu. Mimo něj to není drift, ale
    # artefakt dělení (jedna půlka skoro stála) – hlavně na speed fallbacku.
    if effort_col == "speed" and not (-25.0 <= drift <= 40.0):
        return None
    return drift


# ═══════════════════════════════════════════════════════════════════════════
# MODUL B – MAXIMÁLNÍ POKLES TEPU ZA 60 s
# ═══════════════════════════════════════════════════════════════════════════

def _window_is_smooth(hr_window: np.ndarray, max_3s_jump: int = MAX_3S_JUMP) -> bool:
    """True, pokud v okně není žádný 3sekundový propad větší než limit."""
    if len(hr_window) < 4:
        return True
    return bool(np.all(hr_window[:-3] - hr_window[3:] <= max_3s_jump))


def max_hrr_60s(tdata: pd.DataFrame) -> Optional[float]:
    """
    Největší pokles tepu za 60 s s ochranou proti artefaktům senzoru.

    Kontroly každého okna: start alespoň MIN_STARTING_HR (zotavení má smysl
    jen ze zátěže), pokles do MAX_REALISTIC_HRR (víc je glitch) a plynulý
    průběh bez skoků.
    """
    if tdata.empty or "heart_rate" not in tdata.columns:
        return None

    df = tdata[["timestamp", "heart_rate"]].dropna(subset=["heart_rate"]).copy()
    if len(df) < 60:
        return None

    df = df.set_index("timestamp").sort_index()
    hr = (
        df.resample(HRR_RESAMPLE_FREQ)
        .mean()
        .interpolate(method="time", limit=HRR_INTERPOLATE_LIMIT)["heart_rate"]
        .dropna()
    )
    n = len(hr)
    if n < HRR_DROP_WINDOW + 1:
        return None

    hr_vals = hr.to_numpy(dtype=float)
    best_drop = 0.0
    for t in range(n - HRR_DROP_WINDOW):
        start_hr = hr_vals[t]
        drop = start_hr - hr_vals[t + HRR_DROP_WINDOW]
        if drop <= 0 or drop <= best_drop:
            continue
        if start_hr < MIN_STARTING_HR or drop > MAX_REALISTIC_HRR:
            continue
        if not _window_is_smooth(hr_vals[t : t + HRR_DROP_WINDOW + 1]):
            continue
        best_drop = drop

    return round(best_drop, 1) if best_drop > 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# MODUL P – DURABILITA (pokles EF u dlouhých aktivit)
# ═══════════════════════════════════════════════════════════════════════════

def durability(tdata: pd.DataFrame, sport: str) -> Optional[float]:
    """
    Porovná efektivitu 1. a 2. poloviny aktivity delší než 2 h.

    Durability = (EF_2 / EF_1 − 1) × 100. Záporné číslo znamená pokles
    výkonu při stejném tepu, tedy únavu.

    Kolo bez wattmetru spadne na speed/HR – ale jen na rovině, jinak by
    profil trasy převážil únavu. Bez fallbacku měla durabilitu jen hrstka
    jízd s power daty (3 z 884).
    """
    if tdata.empty:
        return None

    is_cycling = is_cycling_sport(sport)
    has_power = (
        "power" in tdata.columns
        and pd.to_numeric(tdata["power"], errors="coerce").fillna(0).gt(0).sum() >= 20
    )
    effort_col = "power" if (is_cycling and has_power) else "speed"
    on_speed = effort_col == "speed"
    if effort_col not in tdata.columns:
        return None

    active = tdata.loc[
        (tdata["is_active"] == True)  # noqa: E712
        & tdata["heart_rate"].notna()
        & (tdata["heart_rate"] > 0)
        & tdata[effort_col].notna()
        & (tdata[effort_col] > 0)
    ].copy()
    if len(active) < 20:
        return None

    t0 = active["timestamp"].iloc[0]
    t_end = active["timestamp"].iloc[-1]
    if (t_end - t0).total_seconds() / 60.0 < DURABILITY_MIN_DURATION_MIN:
        return None

    # Rovinatost, když se počítá ze speed – profil trasy jinak přebije únavu.
    if on_speed and "altitude" in active.columns:
        alt_vals = pd.to_numeric(active["altitude"], errors="coerce").dropna()
        if len(alt_vals) >= 10 and float(alt_vals.max() - alt_vals.min()) > CARDIAC_DRIFT_MAX_ALT_RANGE_M:
            return None

    t_mid = t0 + (t_end - t0) / 2
    h1 = active[active["timestamp"] <= t_mid]
    h2 = active[active["timestamp"] > t_mid]
    if h1.empty or h2.empty:
        return None

    hr1, hr2 = h1["heart_rate"].mean(), h2["heart_rate"].mean()
    eff1, eff2 = h1[effort_col].mean(), h2[effort_col].mean()
    if hr1 == 0 or hr2 == 0:
        return None
    if effort_col == "power":
        if eff1 < 30 or eff2 < 30:
            return None
    else:
        if eff1 < 0.56 or eff2 < 0.56:
            return None

    ef1, ef2 = eff1 / hr1, eff2 / hr2
    if ef1 == 0:
        return None
    return round((ef2 / ef1 - 1) * 100.0, 2)


# ═══════════════════════════════════════════════════════════════════════════
# MODUL S – DFA-alpha1 PROXY (aerobní práh z linearity HR vs rychlost)
# ═══════════════════════════════════════════════════════════════════════════

def dfa_alpha1_proxy(tdata: pd.DataFrame) -> Optional[int]:
    """
    Náhradní odhad aerobního prahu, když nejsou k dispozici R-R intervaly.

    Hledá bod, kde se láme linearita nárůstu tepu vůči rychlosti.
    """
    if tdata.empty:
        return None

    # Přímá detekce z HRV sloupce (pokles variability)
    if "hrv" in tdata.columns and tdata["hrv"].notna().sum() > 60:
        df = tdata[["heart_rate", "hrv"]].dropna().copy()
        if len(df) > 100:
            df = df.sort_values("heart_rate")
            df["hr_bin"] = (df["heart_rate"] // 5) * 5
            binned = df.groupby("hr_bin").agg(hrv_mean=("hrv", "mean"), count=("hrv", "count"))
            binned = binned[binned["count"] >= 3]
            if len(binned) >= 4:
                threshold = binned["hrv_mean"].max() * 0.5
                below = binned[binned["hrv_mean"] < threshold]
                if not below.empty:
                    aet_hr = int(below.index[0])
                    if aet_hr > (ATHLETE_MAX_HR * 0.80):
                        return None
                    if 130 <= aet_hr <= (ATHLETE_MAX_HR * 0.85):
                        return aet_hr

    if "speed" not in tdata.columns:
        return None

    active = tdata.loc[
        (tdata["is_active"] == True)  # noqa: E712
        & tdata["heart_rate"].notna()
        & (tdata["heart_rate"] > 0)
        & tdata["speed"].notna()
        & (tdata["speed"] > 0.5)
    ].copy()
    if len(active) < 100:
        return None

    # Filtr sklonu ±2 %: do kopce i z kopce je vztah tep↔rychlost rozbitý
    # gravitací nezávisle na aerobním výkonu.
    if "altitude" in active.columns and "distance" in active.columns:
        delta_alt = pd.to_numeric(active["altitude"], errors="coerce").diff()
        delta_dist = pd.to_numeric(active["distance"], errors="coerce").diff()
        gradient_pct = (delta_alt / delta_dist.replace(0, np.nan)) * 100.0
        active = active[(gradient_pct.abs() <= 2.0).fillna(True)]
        if len(active) < 100:
            return None

    active = active.sort_values("speed")
    n = len(active)
    n_segments = min(10, n // 20)
    if n_segments < 4:
        return None

    seg_size = n // n_segments
    slopes, hr_mids = [], []
    for i in range(n_segments - 1):
        s1 = active.iloc[i * seg_size : (i + 1) * seg_size]
        s2 = active.iloc[(i + 1) * seg_size : (i + 2) * seg_size]
        dhr = s2["heart_rate"].mean() - s1["heart_rate"].mean()
        dsp = s2["speed"].mean() - s1["speed"].mean()
        # abs(), aby mírný pokles rychlosti mezi segmenty nedal záporný sklon
        if abs(dsp) > 0.05:
            slopes.append(dhr / dsp)
            hr_mids.append(s2["heart_rate"].mean())

    if len(slopes) < 3:
        return None

    slope_diffs = [slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)]
    aet_hr = int(hr_mids[int(np.argmax(slope_diffs))])

    # Fyziologický strop: AeT/VT1 nemůže být nad 80 % MaxHR; šum z optického
    # snímače umí vyrobit falešný zlom až u prahových hodnot.
    if aet_hr > (ATHLETE_MAX_HR * 0.80):
        return None
    if aet_hr < 130 or aet_hr > (ATHLETE_MAX_HR * 0.85):
        return None
    return aet_hr


# ═══════════════════════════════════════════════════════════════════════════
# MODUL U – DFA-alpha1 z R-R intervalů (neurokit2)
# ═══════════════════════════════════════════════════════════════════════════

def dfa_alpha1_thresholds(rr_ms: list[float] | None, tdata: pd.DataFrame) -> dict:
    """
    Aerobní (α1 = 0.75) a anaerobní (α1 = 0.50) práh z R-R intervalů.

    Když R-R data chybí nebo jsou příliš zašuměná, spadne to na proxy
    z linearity HR vs rychlost (modul S).
    """
    result = {
        "aet_hr_dfa": None, "ant_hr_dfa": None, "dfa_quality": "no_rr",
        "dfa_alpha1_min": None, "dfa_alpha1_median": None, "dfa_window_count": None,
    }

    if not rr_ms or len(rr_ms) < DFA_WINDOW_BEATS:
        return result

    # Přítomnost R-R ještě neznamená, že nesou variabilitu mezi tepy. Bez téhle
    # kontroly se počítalo α1 ≈ 1,56 z kvantizované tepové křivky a výsledek
    # vypadal jako platná diagnostika metody, ne jako vlastnost vstupu.
    authenticity = assess_rr_authenticity(np.asarray(rr_ms, dtype=float) / 1000.0)
    if not authenticity.usable:
        result["dfa_quality"] = (
            "synthetic_rr" if authenticity.verdict == "synthetic" else "no_rr"
        )
        return result

    if not HAS_NEUROKIT:
        result["dfa_quality"] = "failed"
        log.debug("neurokit2 není k dispozici – DFA se nepočítá.")
        return result

    rr_clean = clean_rr_intervals(rr_ms)
    if len(rr_clean) < DFA_WINDOW_BEATS:
        result["dfa_quality"] = "unreliable"
        return result

    hr_alpha_pairs: list[tuple[float, float]] = []
    for start in range(0, len(rr_clean) - DFA_WINDOW_BEATS + 1, DFA_SLIDE_BEATS):
        window = rr_clean[start : start + DFA_WINDOW_BEATS]
        if np.std(window) / np.mean(window) > 0.25:  # okno plné artefaktů
            continue
        avg_hr = 60000.0 / np.mean(window)
        try:
            # neurokit2 >= 0.2 vrací (alpha: float, info: dict) pro
            # multifractal=False; starší (alpha, _). Ověřeno na 0.2.13.
            # `.item()` sundá případný 0-d ndarray, ať `np.isfinite` nespadne.
            dfa_out = nk.fractal_dfa(window, scale=DFA_BOX_SIZES)
            alpha1 = float(np.asarray(dfa_out[0]).reshape(-1)[0])
            if np.isfinite(alpha1) and 0.0 < alpha1 < 2.0:
                hr_alpha_pairs.append((avg_hr, alpha1))
        except Exception as exc:  # noqa: BLE001 – jedno vadné okno běh neshodí
            log.debug("DFA okno přeskočeno: %s", exc)
            continue

    if len(hr_alpha_pairs) < 10:
        result["dfa_quality"] = "unreliable"
        return result

    pairs = sorted(hr_alpha_pairs, key=lambda x: x[0])
    hrs = np.array([p[0] for p in pairs])
    alphas = np.array([p[1] for p in pairs])

    # Diagnostika: ukládá se, aby bylo z dat vidět, kde metoda funguje a kde
    # ne – místo hádání. Nad kvantizovanou tepovou křivkou tady vycházelo
    # α1 ≈ 1,56; tomu teď předchází kontrola pravosti R-R výš.
    result["dfa_alpha1_min"] = round(float(alphas.min()), 3)
    result["dfa_alpha1_median"] = round(float(np.median(alphas)), 3)
    result["dfa_window_count"] = len(pairs)
    kernel = min(5, max(3, len(alphas) // 3))
    alphas_smooth = uniform_filter1d(alphas, size=kernel) if kernel >= 3 else alphas

    def _crossing(threshold: float) -> Optional[float]:
        for i in range(len(alphas_smooth) - 1):
            if alphas_smooth[i] >= threshold > alphas_smooth[i + 1]:
                denom = alphas_smooth[i + 1] - alphas_smooth[i]
                if denom == 0:
                    return None
                frac = (threshold - alphas_smooth[i]) / denom
                return hrs[i] + frac * (hrs[i + 1] - hrs[i])
        return None

    aet_hr = _crossing(DFA_AET_THRESHOLD)
    ant_hr = _crossing(DFA_ANT_THRESHOLD)

    if aet_hr is not None and aet_hr > (ATHLETE_MAX_HR * 0.80):
        aet_hr = None
    if aet_hr is not None and (aet_hr < 100 or aet_hr > ATHLETE_MAX_HR):
        aet_hr = None
    if ant_hr is not None and (ant_hr < 100 or ant_hr > ATHLETE_MAX_HR):
        ant_hr = None
    if aet_hr is not None and ant_hr is not None and ant_hr <= aet_hr:
        ant_hr = None  # anaerobní práh musí ležet nad aerobním

    result["aet_hr_dfa"] = int(aet_hr) if aet_hr else None
    result["ant_hr_dfa"] = int(ant_hr) if ant_hr else None
    result["dfa_quality"] = "rr_ok"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODUL V – DECHOVÁ FREKVENCE Z RSA
# ═══════════════════════════════════════════════════════════════════════════

def respiration_from_rr(rr_ms: list[float] | None) -> Optional[float]:
    """
    Dechová frekvence z respirační sinusové arytmie. EXPERIMENTÁLNÍ.

    HF složka variability R-R (0.15–0.50 Hz) odpovídá dechové modulaci
    tepu; vrchol v tomto pásmu × 60 = dechů za minutu.

    POZOR: na reálných datech vychází 9–13 dech/min, tzn. přilepené na dolní
    hranici pásma (0.15 Hz = 9/min) – metoda zjevně jen vybírá nejnižší koš,
    ne skutečný dechový vrchol. Metrika se dál počítá a ukládá, ale je
    vyřazená z master exportu (viz exports.EXPERIMENTAL_ACTIVITY_COLUMNS) a
    nemá se z ní usuzovat. Kandidát na přepracování (jiné pásmo / metoda)
    nebo úplné odstranění.
    """
    if not HAS_NEUROKIT or not rr_ms or len(rr_ms) < 120:
        return None

    rr_clean = clean_rr_intervals(rr_ms)
    if len(rr_clean) < 120:
        return None

    try:
        rr_times = np.cumsum(rr_clean) / 1000.0  # s
        fs = 4.0
        f_interp = interp1d(rr_times, rr_clean, kind="cubic", fill_value="extrapolate")
        t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / fs)
        rr_uniform = f_interp(t_uniform)
        if len(rr_uniform) < 64:
            return None

        nperseg = min(256, len(rr_uniform) // 2)
        if nperseg < 32:
            return None
        frequencies, psd = welch(rr_uniform - np.mean(rr_uniform), fs=fs, nperseg=nperseg)

        mask = (frequencies >= 0.15) & (frequencies <= 0.50)
        if not np.any(mask):
            return None
        resp_rate = frequencies[mask][int(np.argmax(psd[mask]))] * 60.0
        return round(resp_rate, 1) if 6.0 <= resp_rate <= 40.0 else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# VEKTOROVÉ METRIKY (nepotřebují vteřinová data)
# ═══════════════════════════════════════════════════════════════════════════

def compute_fueling(df: pd.DataFrame) -> pd.DataFrame:
    """Odhad spálených tuků a cukrů podle času v tepových zónách."""
    df = df.copy()
    fat_kcal = pd.Series(0.0, index=df.index)
    carb_kcal = pd.Series(0.0, index=df.index)

    for col, zone in (("time_in_z1", "Z1"), ("time_in_z2", "Z2"), ("time_in_z3", "Z3"),
                      ("time_in_z4", "Z4"), ("time_in_z5", "Z5")):
        if col not in df.columns:
            continue
        mins = pd.to_numeric(df[col], errors="coerce").fillna(0)
        zone_kcal = mins * KCAL_PER_MIN_BY_ZONE.get(zone, 8.0)
        fat_pct = FAT_PCT_BY_ZONE.get(zone, 0.5)
        fat_kcal += zone_kcal * fat_pct
        carb_kcal += zone_kcal * (1 - fat_pct)

    df["fat_kcal"] = fat_kcal.round(1)
    df["carb_kcal"] = carb_kcal.round(1)
    df["fat_g"] = (fat_kcal / KCAL_PER_G_FAT).round(1)
    df["carb_g"] = (carb_kcal / KCAL_PER_G_CARB).round(1)
    return df


def compute_fluid_loss(df: pd.DataFrame) -> pd.DataFrame:
    """fluid_loss_l = hodiny × (0.6 + 0.1 za každý °C nad 20)."""
    df = df.copy()
    dur_h = pd.to_numeric(df.get("duration_minutes"), errors="coerce").fillna(0) / 60.0
    avg_temp = pd.to_numeric(df.get("avg_temp"), errors="coerce").fillna(20.0)
    heat_factor = (avg_temp - 20).clip(lower=0) * 0.1
    df["fluid_loss_l"] = (dur_h * (0.6 + heat_factor)).round(2)
    return df


def compute_heat_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Označí aktivity nad 25 °C – horko je nejčastější příčina driftu."""
    df = df.copy()
    avg_temp = pd.to_numeric(df.get("avg_temp"), errors="coerce")
    df["heat_flag"] = (avg_temp > 25.0).fillna(False)
    return df


def compute_climb_score(df: pd.DataFrame) -> pd.DataFrame:
    """Průměrný sklon a kategorie terénu."""
    df = df.copy()
    ascent = pd.to_numeric(df.get("ascent_m"), errors="coerce").fillna(0)
    dist_m = pd.to_numeric(df.get("distance_km"), errors="coerce").fillna(0) * 1000.0

    gradient = (ascent / dist_m.replace(0, np.nan)) * 100
    df["avg_gradient_pct"] = gradient.round(2)

    def _categorise(g):
        if pd.isna(g):
            return None
        if g < 1.0:
            return "Flat"
        if g < 3.0:
            return "Rolling"
        if g < 6.0:
            return "Hilly"
        return "Mountainous"

    df["climb_category"] = df["avg_gradient_pct"].apply(_categorise)
    return df


def compute_vam(df: pd.DataFrame) -> pd.DataFrame:
    """
    VAM = převýšení / čas strávený stoupáním [m/h].

    Používá uphill_minutes z FIT parseru (skutečný čas do kopce), takže
    sjezdy ani zastávky výsledek neředí. Když chybí, odhadne se polovina
    trvání. Počítá se jen u kopcovitých aktivit (gradient > 4 %).
    """
    df = df.copy()
    df["vam_m_per_h"] = np.nan
    if "ascent_m" not in df.columns or "duration_minutes" not in df.columns:
        return df

    ascent = pd.to_numeric(df["ascent_m"], errors="coerce").fillna(0)
    dur = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0)
    uphill = pd.to_numeric(df.get("uphill_minutes"), errors="coerce").fillna(0)
    climb_time = uphill.where(uphill > 0, dur / 2.0)

    sport_col = df["sport"].astype(str).str.lower().fillna("")
    # Elektrokolo ven – nahoru veze motor, VAM by měřil asistenci, ne výkon
    # (stejný důvod jako u sjezdovky s lanovkou).
    is_vam_sport = sport_col.str.contains("|".join(VAM_SPORTS), na=False) & ~sport_col.str.contains(
        EBIKE_SPORT_PATTERN, na=False
    )
    if not is_vam_sport.any():
        return df

    dist_km = pd.to_numeric(df.get("distance_km"), errors="coerce").fillna(0)
    is_climb = (ascent / (dist_km * 1000 + 1e-5)) > 0.04

    mask = is_vam_sport & (ascent > 50) & (dur > 15) & (climb_time > 0) & is_climb
    df.loc[mask, "vam_m_per_h"] = (ascent[mask] / (climb_time[mask] / 60)).round(1)
    return df


def compute_epoc(df: pd.DataFrame) -> pd.DataFrame:
    """
    EPOC proxy: minuty nad prahem vážené intenzitou (Z4 ×2, Z5 ×5).

    Dřív tu byla i `recovery_tax_hours = min(96, 0.08 × TRIMP^1.2)`.
    Zrušeno: Spearman s total_trimp byl 0.9955, takže nenesla žádnou
    informaci navíc, a proti Garminovu naměřenému recovery time dávala
    RMSE 32.0 h – k nerozeznání od nejlepší možné konstanty (32.1 h),
    zatímco přeškálované atl dá 26.4 h. Kolik hodin do regenerace zbývá
    dnes říká `recovery_time_h` z daily_biometrics – měřeno hodinkami.
    """
    df = df.copy()
    z4 = pd.to_numeric(df.get("time_in_z4"), errors="coerce").fillna(0)
    z5 = pd.to_numeric(df.get("time_in_z5"), errors="coerce").fillna(0)
    df["epoc_score"] = (z4 * 2 + z5 * 5).round(1)
    return df


def compute_tte(df: pd.DataFrame) -> pd.DataFrame:
    """
    Čas nad prahem (Z4+Z5) pro danou aktivitu.

    Dřívější expanding().max() způsoboval únik informace mezi aktivitami:
    jakmile jedna dosáhla vysoké hodnoty, dědily ji i všechny následující
    včetně těch s nulovým časem v Z4/Z5.
    """
    df = df.copy()
    z4 = pd.to_numeric(df.get("time_in_z4"), errors="coerce").fillna(0)
    z5 = pd.to_numeric(df.get("time_in_z5"), errors="coerce").fillna(0)
    df["time_at_threshold_min"] = (z4 + z5).round(2)
    df["tte_z4z5_min"] = df["time_at_threshold_min"]
    return df


def compute_critical_hr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Critical Heart Rate a TATI (Monod-Scherrer adaptovaný na tep).

    CHR = 85. percentil průměrného tepu z aktivit nad 30 min, stropovaný
    na 85 % MaxHR. Percentil místo maxima záměrně: jeden všestartový
    závod by jinak CHR natrvalo nafoukl.

    TATI = akumulovaná práce nad CHR [bpm·min]. Nahrazuje dřívější
    „W'", protože W' je pojem z výkonové domény a počítat ho z tepu
    je metodicky špatně.
    """
    df = df.copy()
    df["critical_hr"] = np.nan
    df["tati_score"] = np.nan

    if "sport" not in df.columns or df.empty:
        return df

    # Elektrokolo ven z populace pro práh – dlouhá jízda s asistencí drží nízký
    # průměrný tep a stáhla by 85. percentil. tati_score se elektru níž počítá
    # dál (čas v Z4/Z5 je skutečná tepová zátěž).
    is_cardio = cardio_mask(df["sport"])
    avg_hr = pd.to_numeric(df.get("avg_hr"), errors="coerce")
    dur_min = pd.to_numeric(df.get("duration_minutes"), errors="coerce")

    eligible = df.loc[is_cardio & avg_hr.notna() & (dur_min >= 20)]
    if len(eligible) < CHR_MIN_ACTIVITIES:
        return df

    elig_hr = pd.to_numeric(eligible["avg_hr"], errors="coerce")
    elig_dur = pd.to_numeric(eligible["duration_minutes"], errors="coerce")
    long_mask = elig_dur >= 30

    if long_mask.any() and elig_hr[long_mask].notna().any():
        chr_hr = min(float(elig_hr[long_mask].quantile(0.85)), 0.85 * ATHLETE_MAX_HR)
    else:
        chr_hr = 0.85 * ATHLETE_MAX_HR
    chr_hr = round(float(chr_hr))

    # Střed zóny se bere z nakonfigurovaných ZONES, ne z procent tepové
    # rezervy. Minuty v time_in_z4/z5 pocházejí z laktátových zón, takže
    # vážit je odhadem tepu odvozeným jinou definicí by míchalo dvě různé
    # škály – a hlavně by se to tiše rozešlo, kdyby se zóny po dalším
    # laktátovém testu upravily.
    z4_lo, z4_hi = ATHLETE_ZONES["Z4"]
    z5_lo, z5_hi = ATHLETE_ZONES["Z5"]
    avg_z4_hr = (z4_lo + z4_hi) / 2.0
    avg_z5_hr = (z5_lo + z5_hi) / 2.0

    z4_min = pd.to_numeric(df.get("time_in_z4"), errors="coerce").fillna(0)
    z5_min = pd.to_numeric(df.get("time_in_z5"), errors="coerce").fillna(0)

    df["critical_hr"] = chr_hr
    df["tati_score"] = (
        z4_min * max(0, avg_z4_hr - chr_hr) + z5_min * max(0, avg_z5_hr - chr_hr)
    ).round(1)
    return df


def compute_trimp_load_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentilové pořadí zátěže (TRIMP) této aktivity mezi kardio aktivitami
    v celé historii atleta.

    Slouží jako kalibrovaný podklad pro verdikt a strop gauge v detailu
    aktivity, náhrada za vymyšlené pevné prahy (300/150 TRIMP z mockupu).
    Přednost má ``trimp_adjusted`` (RHR platný k datu aktivity), fallback
    ``total_trimp`` z parseru pro aktivity, které se ještě nepřepočítaly.

    Stejná populace jako u ``compute_critical_hr`` (běh + kolo), ne jen
    kolo – i běžecká zátěž patří do srovnání "jak těžký byl tenhle trénink
    vůči zbytku", a rozdělovat podle sportu by při pár desítkách běhů dalo
    šumový percentil. Elektrokolo se tu NEvylučuje: jeho TRIMP je skutečná
    zátěž a do srovnání "jak těžký byl trénink" patří.
    """
    df = df.copy()
    df["trimp_load_percentile"] = np.nan
    if "sport" not in df.columns or df.empty:
        return df

    is_cardio = cardio_mask(df["sport"], exclude_ebike=False)
    trimp = pd.to_numeric(df.get("trimp_adjusted"), errors="coerce")
    if "total_trimp" in df.columns:
        trimp = trimp.fillna(pd.to_numeric(df["total_trimp"], errors="coerce"))

    eligible = is_cardio & trimp.notna()
    if eligible.sum() < CHR_MIN_ACTIVITIES:
        return df

    ranked = trimp[eligible].rank(pct=True, method="max") * 100
    df.loc[eligible, "trimp_load_percentile"] = ranked.round(1)
    return df


def compute_activity_table(activities: pd.DataFrame) -> pd.DataFrame:
    """
    Všechny vektorové per-activity metriky najednou.

    Pozor: compute_critical_hr potřebuje vidět celou historii (počítá
    percentil napříč aktivitami), takže se sem musí předat kompletní
    tabulka, ne jen nově načtené aktivity.
    """
    if activities is None or activities.empty:
        return activities

    df = activities.copy()
    df = compute_fueling(df)
    df = compute_fluid_loss(df)
    df = compute_heat_flag(df)
    df = compute_climb_score(df)
    df = compute_vam(df)
    df = compute_epoc(df)
    df = compute_tte(df)
    df = compute_critical_hr(df)
    df = compute_trimp_load_percentile(df)
    return df


def compute_trimp_from_records(tdata: pd.DataFrame, rhr: float) -> Optional[float]:
    """
    Banisterův TRIMP přepočítaný z vteřinových dat pro zadaný klidový tep.

        TRIMP = Σ (dt/60) × ratio × k1 × e^(k2 × ratio)
        ratio = (HR − RHR) / (MaxHR − RHR),  jen kde is_active

    Proč znovu, když ho parser počítá při načtení: parser použil pevnou
    konstantu ze settings, ale klidový tep se v čase mění. Stejný tep při
    RHR 44 a při RHR 52 znamená jinou relativní zátěž.

    Výsledek má přednost před parserovým total_trimp (viz load.py), takže
    tenhle přepočet musí sedět s parserem ve všem ostatním – proto se tu
    doplňují mezery v tepu stejně jako v parseru (poslední naměřená hodnota,
    nejvýš 15 záznamů dopředu). Bez toho by u Strava exportů, které vzorkují
    tep jednou za pět vteřin, vyšla čtyři pětiny zátěže nulové.
    """
    if tdata.empty or "heart_rate" not in tdata.columns:
        return None

    hrr = ATHLETE_MAX_HR - rhr
    if hrr <= 0:
        return None

    df = tdata.sort_values("timestamp")
    # ffill(limit=15) zrcadlí hr_effective v parseru – viz docstring.
    # Implausibilní vzorek se zahodí PŘED ffillem, ať se glitch nešíří dál
    # (parser dělá totéž přes _plausible_hr).
    hr_raw = pd.to_numeric(df["heart_rate"], errors="coerce")
    hr_raw = hr_raw.where(
        (hr_raw >= HR_PLAUSIBLE_MIN_BPM) & (hr_raw <= HR_PLAUSIBLE_MAX_BPM)
    )
    hr = hr_raw.ffill(limit=15).to_numpy(dtype=float)

    # Časový krok mezi záznamy, stejně jako v parseru: strop 120 s brání
    # tomu, aby dlouhá pauza v nahrávání nafoukla zátěž.
    seg = df["timestamp"].diff().dt.total_seconds().to_numpy(dtype=float)
    seg = np.clip(np.nan_to_num(seg, nan=1.0), 0.0, 120.0)
    if len(seg):
        seg[0] = 1.0

    active = (
        df["is_active"].fillna(False).to_numpy(dtype=bool)
        if "is_active" in df.columns
        else np.ones(len(df), dtype=bool)
    )

    valid = active & np.isfinite(hr) & (hr > rhr) & (seg > 0)
    if not valid.any():
        return 0.0

    ratio = np.clip((hr[valid] - rhr) / hrr, 0.0, 1.0)
    trimp = (seg[valid] / 60.0) * ratio * TRIMP_K1 * np.exp(TRIMP_K2 * ratio)
    return round(float(trimp.sum()), 2)


def compute_best_hr_windows(
    tdata: pd.DataFrame, windows_min: list[int] | None = None
) -> dict:
    """
    Nejlepší klouzavý průměr tepu pro zadaná okna – podklad pro odhad prahu.

    Na rozdíl od DFA-alpha1 funguje nad každou aktivitou s tepem, ne jen
    nad těmi s hrudním pásem, a nevyžaduje beat-to-beat data.

    Počítá to ``src.physio.hr_curve`` – totéž jádro, které plní tabulku
    ``activity_hr_curve``. Tahle funkce je jen adaptér z DataFrame na
    sloupce ``best_20min_hr`` … a existuje proto, že analytika běží dřív
    než předvýpočet křivky, takže se na uložené řádky spolehnout nemůže.

    Dřív tu bylo vlastní počítání s ``min_periods = 0.9 × okno`` nad
    interpolací do 30 s. To znamenalo, že se průměr z 18 minut dat vydával
    za dvacetiminutové maximum – a proti uložené křivce, která vyžaduje
    plné pokrytí, vycházelo na stejnou aktivitu jiné číslo. Jedna metrika,
    jedno jádro: okno bez plného pokrytí nevznikne.
    """
    windows_min = windows_min or LTHR_BEST_WINDOWS_MIN
    out: dict = {f"best_{m}min_hr": None for m in windows_min}

    if tdata.empty or not {"timestamp", "heart_rate"} <= set(tdata.columns):
        return out

    grid = to_second_grid(
        tdata["timestamp"].to_numpy(),
        pd.to_numeric(tdata["heart_rate"], errors="coerce").to_numpy(dtype=float),
        ffill_limit_s=HR_GRID_FFILL_LIMIT_S,
    )
    curve = max_mean_curve(grid, [m * 60 for m in windows_min])

    for m in windows_min:
        value = curve.get(m * 60)
        if value is not None:
            out[f"best_{m}min_hr"] = value
    return out


def is_series_eligible(sport: str | None, duration_minutes: float | None) -> bool:
    """
    Vteřinová analýza má smysl jen u kardio aktivit delších než 20 min.

    Elektrokolo je vyloučené: drift, DFA prahy, tepová křivka i HRR stojí na
    vztahu výkon/rychlost ↔ tep, který motor rozbíjí. TRIMP a minuty v zónách
    se počítají jinde (parser), takže elektro dál přispívá do formy.
    """
    if not sport or duration_minutes is None or pd.isna(duration_minutes):
        return False
    if duration_minutes < MIN_DURATION_DRIFT_MIN:
        return False
    # is_cardio_sport už elektrokolo vylučuje; drží i MTB / gravel / "Ride"
    # ze Stravy, které holé `"cycling" in sport` míjelo.
    return is_cardio_sport(sport)


def compute_activity_series_metrics(
    tdata: pd.DataFrame,
    sport: str,
    rr_ms: list[float] | None,
) -> dict:
    """
    Metriky vyžadující vteřinová data a R-R intervaly jedné aktivity.

    Tohle je jediná drahá část analytiky – proto se počítá jen pro
    aktivity se zastaralou metrics_version.
    """
    sport_lower = str(sport or "").lower()
    dfa = dfa_alpha1_thresholds(rr_ms, tdata)
    return {
        **compute_best_hr_windows(tdata),
        "dfa_alpha1_min": dfa["dfa_alpha1_min"],
        "dfa_alpha1_median": dfa["dfa_alpha1_median"],
        "dfa_window_count": dfa["dfa_window_count"],
        "cardiac_drift": cardiac_drift(tdata, sport_lower),
        "max_hrr_60s": max_hrr_60s(tdata),
        "durability_pct": durability(tdata, sport_lower),
        "aet_hr_dfa": dfa["aet_hr_dfa"],
        "ant_hr_dfa": dfa["ant_hr_dfa"],
        "dfa_quality": dfa["dfa_quality"],
        # Proxy se počítá vždy a zůstává jen tady. Dřív se při chybějící DFA
        # propsal i do aet_hr_dfa, takže odhad ze zlomu linearity tep↔rychlost
        # vystupoval jako práh spočítaný z R-R intervalů.
        "aet_hr_proxy": dfa_alpha1_proxy(tdata),
        "resp_rate_rsa": respiration_from_rr(rr_ms),
    }
