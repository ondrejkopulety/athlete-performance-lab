"""
Repository vrstva – jediné místo, které ví, jak se čte/zapisuje do DB.

Analytika i API pracují s pandas DataFrame; tenhle modul překládá mezi
DataFrame a SQL a řeší dvě věci, na kterých se to jinak vždycky rozbije:

  1. NaN → NULL. pandas NaN/NaT/pd.NA nejsou platné hodnoty pro Postgres
     ani pro JSON; `records_to_dicts()` je převádí na None.
  2. Bulk zápis vteřinových dat přes COPY, ne INSERT. U ~40 tisíc řádků
     na aktivitu je to rozdíl sekund vs. minut.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.db.models import (
    Activity,
    ActivityHrBlocks,
    ActivityHrCoverage,
    ActivityHrCurve,
    ActivityMetrics,
    AthleteThreshold,
    DailyBiometrics,
    DailyMetrics,
    Record,
    SyncState,
)
from src.db.session import raw_connection

# ═══════════════════════════════════════════════════════════════════════════
# NaN / typová sanitizace
# ═══════════════════════════════════════════════════════════════════════════

def _clean_value(v: Any) -> Any:
    """pandas/numpy hodnota → něco, co spolkne psycopg i json.dumps."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    # pd.isna vyhodí ValueError na polích/listech – ty projdou beze změny
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def records_to_dicts(df: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict]:
    """DataFrame → list dictů připravený pro upsert (NaN už jsou None)."""
    if df.empty:
        return []
    cols = [c for c in (columns or df.columns) if c in df.columns]
    out: list[dict] = []
    for row in df[cols].to_dict(orient="records"):
        out.append({k: _clean_value(v) for k, v in row.items()})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Generický upsert
# ═══════════════════════════════════════════════════════════════════════════

def upsert(
    session: Session,
    model,
    rows: list[dict],
    index_elements: Sequence[str],
    update_columns: Sequence[str] | None = None,
    chunk_size: int = 1000,
) -> int:
    """
    INSERT ... ON CONFLICT DO UPDATE nad ORM modelem.

    `update_columns=None` znamená „aktualizuj všechny sloupce kromě klíče".
    Vrací počet zapsaných řádků.
    """
    if not rows:
        return 0

    table_cols = {c.name for c in model.__table__.columns}
    written = 0

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        # Odfiltruj klíče, které v tabulce nejsou (CSV migrace nosí i sloupce navíc)
        payload = [{k: v for k, v in r.items() if k in table_cols} for r in chunk]
        if not payload:
            continue

        stmt = pg_insert(model).values(payload)
        present = {k for r in payload for k in r}
        targets = update_columns or [
            c for c in present if c not in index_elements and c in table_cols
        ]
        if targets:
            stmt = stmt.on_conflict_do_update(
                index_elements=list(index_elements),
                set_={c: getattr(stmt.excluded, c) for c in targets},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(index_elements))

        session.execute(stmt)
        written += len(payload)

    return written


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITIES
# ═══════════════════════════════════════════════════════════════════════════

def upsert_activities(session: Session, rows: list[dict]) -> int:
    return upsert(session, Activity, rows, index_elements=["activity_id"])


def upsert_activity_metrics(session: Session, rows: list[dict]) -> int:
    return upsert(session, ActivityMetrics, rows, index_elements=["activity_id"])


def upsert_hr_curve(session: Session, rows: list[dict]) -> int:
    return upsert(session, ActivityHrCurve, rows, index_elements=["activity_id", "duration_s"])


def upsert_hr_blocks(session: Session, rows: list[dict]) -> int:
    return upsert(
        session,
        ActivityHrBlocks,
        rows,
        index_elements=["activity_id", "threshold_bpm", "bridge_tolerance_s"],
    )


def upsert_hr_coverage(session: Session, rows: list[dict]) -> int:
    return upsert(session, ActivityHrCoverage, rows, index_elements=["activity_id"])


def hr_computed_ids(session: Session, model, calc_version: int) -> set[str]:
    """
    Aktivity, které už mají řádky v současné verzi výpočtu.

    Tohle je cache pro CLI: aktivita, která tu je, se nepřepočítává, pokud
    nepřijde ``--force``. Řádky se starší ``calc_version`` se nezapočítávají,
    takže bump verze vynutí přepočet sám od sebe.
    """
    stmt = select(model.activity_id).where(model.calc_version >= calc_version).distinct()
    return {row[0] for row in session.execute(stmt).all()}


def read_hr_curve(session: Session, wide: bool = False) -> pd.DataFrame:
    """
    Tepová křivka všech aktivit.

    Args:
        session: Otevřená session.
        wide: ``True`` pivotuje na jeden řádek na aktivitu se sloupci
            ``hr_curve_5s … hr_curve_3600s`` – v tomhle tvaru se křivka
            připojuje k master CSV.
    """
    stmt = select(
        ActivityHrCurve.activity_id, ActivityHrCurve.duration_s, ActivityHrCurve.max_mean_hr
    ).order_by(ActivityHrCurve.activity_id, ActivityHrCurve.duration_s)
    df = pd.DataFrame(session.execute(stmt).mappings().all())
    if df.empty or not wide:
        return df

    df["max_mean_hr"] = df["max_mean_hr"].astype(float)
    pivot = df.pivot(index="activity_id", columns="duration_s", values="max_mean_hr")
    pivot.columns = [f"hr_curve_{int(c)}s" for c in pivot.columns]
    return pivot.reset_index()


def read_hr_blocks(session: Session) -> pd.DataFrame:
    """Souvislé bloky nad prahem – dlouhý formát, jeden řádek na práh × toleranci."""
    stmt = select(ActivityHrBlocks).order_by(
        ActivityHrBlocks.activity_id,
        ActivityHrBlocks.threshold_bpm,
        ActivityHrBlocks.bridge_tolerance_s,
    )
    return pd.DataFrame(
        [
            {c.name: getattr(b, c.name) for c in ActivityHrBlocks.__table__.columns}
            for b in session.scalars(stmt)
        ]
    )


# ── Podklad pro panely dashboardu ─────────────────────────────────────────
# Vrací se seznamy slovníků, ne DataFrame: jde o desítky až stovky řádků,
# které putují rovnou do JSON, takže převod přes pandas by jen přidal krok.


def _activity_label(sport: str | None, km: float | None, minutes: float | None) -> str:
    """
    Popisek jízdy do panelu – "odkud ten bod je".

    Nepoužívá ``activities.activity_name``: ten je v celé databázi NULL
    (Garmin ho v exportu neposílá), takže by z popisku zbylo prázdno.
    """
    parts = [(sport or "aktivita").split("/")[0]]
    if km:
        parts.append(f"{km:.0f} km")
    if minutes:
        hours, mins = divmod(int(minutes), 60)
        parts.append(f"{hours}:{mins:02d}" if hours else f"{mins} min")
    return " · ".join(parts)


def read_curve_rows(
    session: Session,
    since: date | None = None,
    until: date | None = None,
    min_coverage_pct: float | None = None,
) -> list[dict]:
    """
    Řádky tepové křivky za období, i s tím, ze které jízdy jsou.

    Args:
        session: Otevřená session.
        since: Od data včetně.
        until: Do data včetně.
        min_coverage_pct: Když je zadané, projdou jen jízdy s pokrytím nad
            hranicí. Jízdy, ke kterým pokrytí ještě spočítané není, se
            **nevyřazují** – "neznámé pokrytí" není totéž co "špatné".

    Returns:
        Slovníky s ``duration_s``, ``max_mean_hr``, ``activity_id``,
        ``date``, ``label``.
    """
    stmt = (
        select(
            ActivityHrCurve.duration_s,
            ActivityHrCurve.max_mean_hr,
            ActivityHrCurve.activity_id,
            Activity.date,
            Activity.sport,
            Activity.distance_km,
            Activity.duration_minutes,
        )
        .join(Activity, Activity.activity_id == ActivityHrCurve.activity_id)
        .order_by(Activity.date)
    )
    stmt = _filter_period(stmt, since, until)
    if min_coverage_pct is not None:
        stmt = _filter_coverage(stmt, ActivityHrCurve.activity_id, min_coverage_pct)

    return [
        {
            "duration_s": r.duration_s,
            "max_mean_hr": float(r.max_mean_hr),
            "activity_id": r.activity_id,
            "date": r.date,
            "label": _activity_label(r.sport, r.distance_km, r.duration_minutes),
        }
        for r in session.execute(stmt).all()
    ]


def read_block_rows(
    session: Session,
    threshold_bpm: int,
    bridge_tolerance_s: int,
    since: date | None = None,
    until: date | None = None,
    min_coverage_pct: float | None = None,
) -> list[dict]:
    """
    Řádky souvislých bloků jednoho prahu a jedné tolerance za období.

    Práh jde dovnitř jako parametr dotazu – změna LTHR tedy mění ``WHERE``,
    ne uložená data.
    """
    stmt = (
        select(
            ActivityHrBlocks.activity_id,
            ActivityHrBlocks.longest_block_s,
            ActivityHrBlocks.total_time_s,
            ActivityHrBlocks.time_in_long_blocks_s,
            ActivityHrBlocks.segment_count,
            ActivityHrBlocks.segment_hist_counts,
            ActivityHrBlocks.segment_hist_seconds,
            Activity.date,
            Activity.sport,
            Activity.distance_km,
            Activity.duration_minutes,
        )
        .join(Activity, Activity.activity_id == ActivityHrBlocks.activity_id)
        .where(
            ActivityHrBlocks.threshold_bpm == threshold_bpm,
            ActivityHrBlocks.bridge_tolerance_s == bridge_tolerance_s,
        )
        .order_by(Activity.date)
    )
    stmt = _filter_period(stmt, since, until)
    if min_coverage_pct is not None:
        stmt = _filter_coverage(stmt, ActivityHrBlocks.activity_id, min_coverage_pct)

    return [
        {
            "activity_id": r.activity_id,
            "longest_block_s": r.longest_block_s,
            "total_time_s": r.total_time_s,
            "time_in_long_blocks_s": r.time_in_long_blocks_s,
            "segment_count": r.segment_count,
            "segment_hist_counts": list(r.segment_hist_counts or []),
            "segment_hist_seconds": list(r.segment_hist_seconds or []),
            "date": r.date,
            "label": _activity_label(r.sport, r.distance_km, r.duration_minutes),
        }
        for r in session.execute(stmt).all()
    ]


def read_block_totals(
    session: Session, thresholds_bpm: Sequence[int], bridge_tolerance_s: int
) -> dict[str, dict[int, dict[str, int]]]:
    """
    Čas nad prahem a čas v dlouhých úsecích pro vybrané prahy, po aktivitách.

    Podklad pro nezáměrnou Z3, která se počítá rozdílem dvou prahů.

    Returns:
        ``{activity_id: {threshold: {"total_s": …, "long_s": …}}}``.
    """
    if not thresholds_bpm:
        return {}

    stmt = select(
        ActivityHrBlocks.activity_id,
        ActivityHrBlocks.threshold_bpm,
        ActivityHrBlocks.total_time_s,
        ActivityHrBlocks.time_in_long_blocks_s,
    ).where(
        ActivityHrBlocks.threshold_bpm.in_(list(thresholds_bpm)),
        ActivityHrBlocks.bridge_tolerance_s == bridge_tolerance_s,
    )

    out: dict[str, dict[int, dict[str, int]]] = {}
    for r in session.execute(stmt).all():
        out.setdefault(r.activity_id, {})[r.threshold_bpm] = {
            "total_s": int(r.total_time_s or 0),
            "long_s": int(r.time_in_long_blocks_s or 0),
        }
    return out


def read_hr_coverage(session: Session) -> dict[str, dict]:
    """Pokrytí všech aktivit, podle ``activity_id``."""
    stmt = select(
        ActivityHrCoverage.activity_id,
        ActivityHrCoverage.span_s,
        ActivityHrCoverage.measured_s,
        ActivityHrCoverage.usable_s,
        ActivityHrCoverage.longest_gap_s,
        ActivityHrCoverage.max_curve_duration_s,
    )
    return {
        r.activity_id: {
            "span_s": r.span_s,
            "measured_s": r.measured_s,
            "usable_s": r.usable_s,
            "longest_gap_s": r.longest_gap_s,
            "max_curve_duration_s": r.max_curve_duration_s,
        }
        for r in session.execute(stmt).all()
    }


def count_rides_in_period(
    session: Session,
    sport_pattern: str,
    since: date | None = None,
    until: date | None = None,
    min_coverage_pct: float | None = None,
) -> int:
    """Kolik jízd v období projde filtrem – podklad pro "3 z 24 vyloučeny"."""
    stmt = select(func.count(func.distinct(Activity.activity_id))).where(
        Activity.sport.op("~*")(sport_pattern)
    )
    stmt = _filter_period(stmt, since, until)
    if min_coverage_pct is not None:
        stmt = _filter_coverage(stmt, Activity.activity_id, min_coverage_pct)
    return int(session.scalar(stmt) or 0)


def _filter_period(stmt, since: date | None, until: date | None):
    if since is not None:
        stmt = stmt.where(Activity.date >= since)
    if until is not None:
        stmt = stmt.where(Activity.date <= until)
    return stmt


def _filter_coverage(stmt, activity_id_col, min_coverage_pct: float):
    """
    Propustí jen jízdy s dostatečným pokrytím po ffillu.

    Aktivita bez řádku v ``activity_hr_coverage`` projde: chybějící posudek
    znamená "ještě se nepočítalo", ne "špatná data", a tiché vyřazení by
    graf zmenšilo bez vysvětlení. Pokrytí se počítá v SQL, aby se do Pythonu
    netahaly řádky, které stejně vypadnou.
    """
    return stmt.join(
        ActivityHrCoverage,
        ActivityHrCoverage.activity_id == activity_id_col,
        isouter=True,
    ).where(
        (ActivityHrCoverage.activity_id.is_(None))
        | (ActivityHrCoverage.span_s == 0)
        | (
            100.0 * ActivityHrCoverage.usable_s / func.nullif(ActivityHrCoverage.span_s, 0)
            >= min_coverage_pct
        )
    )


# ── Práh (LTHR / maximální tep) ───────────────────────────────────────────

def read_current_threshold(session: Session) -> dict | None:
    """Poslední platné nastavení prahu; ``None``, když si ho uživatel nikdy
    nenastavil (pak platí měřené hodnoty ze settings)."""
    stmt = (
        select(AthleteThreshold)
        .order_by(AthleteThreshold.valid_from.desc(), AthleteThreshold.id.desc())
        .limit(1)
    )
    row = session.scalars(stmt).first()
    if row is None:
        return None
    return {
        "lthr_bpm": row.lthr_bpm,
        "hr_max_bpm": row.hr_max_bpm,
        "valid_from": row.valid_from,
        "note": row.note,
    }


def insert_threshold(
    session: Session,
    lthr_bpm: int,
    hr_max_bpm: int,
    valid_from: date,
    note: str | None = None,
) -> dict:
    """
    Zapíše nové nastavení prahu jako další řádek historie.

    Nepřepisuje předchozí: "nastaveno před N dny" musí být z čeho spočítat a
    posun prahu v čase je sám o sobě informace. Uložených dat se to nedotkne –
    LTHR řídí jen lookup prahu při zobrazení.
    """
    row = AthleteThreshold(
        lthr_bpm=lthr_bpm, hr_max_bpm=hr_max_bpm, valid_from=valid_from, note=note
    )
    session.add(row)
    session.flush()
    return {
        "lthr_bpm": row.lthr_bpm,
        "hr_max_bpm": row.hr_max_bpm,
        "valid_from": row.valid_from,
        "note": row.note,
    }


def read_hr_series(
    session: Session, activity_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Tep všech (nebo vybraných) aktivit jedním dotazem.

    Vteřinová data se tu čtou jinak než přes ``read_records``: bere se jen
    ``timestamp`` a ``heart_rate`` a rovnou přes ``read_sql``, bez ORM. Pro
    celou databázi je to 2,6 milionu řádků za ~5 sekund, kdežto stejná data
    po aktivitách přes ORM trvají minuty. Předvýpočet křivky a bloků nad tím
    pak běží v jednotkách sekund, takže nepotřebuje paralelizaci.
    """
    sql = (
        "SELECT activity_id, timestamp, heart_rate FROM records "
        "WHERE heart_rate IS NOT NULL"
    )
    params: dict[str, Any] = {}
    if activity_ids is not None:
        if not len(activity_ids):
            return pd.DataFrame(columns=["activity_id", "timestamp", "heart_rate"])
        sql += " AND activity_id = ANY(:ids)"
        params["ids"] = list(activity_ids)

    df = pd.read_sql(text(sql), session.connection(), params=params)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def read_activities(
    session: Session,
    since: date | None = None,
    until: date | None = None,
    with_metrics: bool = True,
) -> pd.DataFrame:
    """Aktivity (volitelně joinuté s odvozenými metrikami) jako DataFrame."""
    stmt = select(Activity)
    if since is not None:
        stmt = stmt.where(Activity.date >= since)
    if until is not None:
        stmt = stmt.where(Activity.date <= until)
    stmt = stmt.order_by(Activity.date, Activity.activity_id)

    acts = pd.DataFrame(
        [
            {c.name: getattr(a, c.name) for c in Activity.__table__.columns}
            for a in session.scalars(stmt)
        ]
    )
    if acts.empty:
        return acts

    if not with_metrics:
        return acts

    m_stmt = select(ActivityMetrics).where(
        ActivityMetrics.activity_id.in_(acts["activity_id"].tolist())
    )
    metrics = pd.DataFrame(
        [
            {c.name: getattr(m, c.name) for c in ActivityMetrics.__table__.columns}
            for m in session.scalars(m_stmt)
        ]
    )
    if metrics.empty:
        return acts
    metrics = metrics.drop(columns=["computed_at", "rr_intervals_ms"], errors="ignore")
    return acts.merge(metrics, on="activity_id", how="left", suffixes=("", "_m"))


def existing_activity_hashes(session: Session) -> dict[str, str | None]:
    """activity_id → fit_sha256 pro rychlý skip už načtených FIT souborů."""
    rows = session.execute(select(Activity.activity_id, Activity.fit_sha256)).all()
    return {aid: sha for aid, sha in rows}


def stale_activity_ids(session: Session, required_version: int) -> list[str]:
    """
    Aktivity, kterým chybí odvozené metriky nebo mají zastaralou verzi vzorců.
    Tohle je jádro inkrementality – při denním běhu vrátí typicky 0–1 položek.
    """
    stmt = (
        select(Activity.activity_id)
        .outerjoin(ActivityMetrics, Activity.activity_id == ActivityMetrics.activity_id)
        .where(
            (ActivityMetrics.activity_id.is_(None))
            | (ActivityMetrics.metrics_version < required_version)
        )
        .order_by(Activity.date)
    )
    return [r[0] for r in session.execute(stmt).all()]


def read_rr_intervals(session: Session, activity_id: str) -> list[float] | None:
    """Uložené R-R intervaly (ms) – DFA i RSA se počítají bez sáhnutí na disk."""
    return session.scalar(
        select(ActivityMetrics.rr_intervals_ms).where(
            ActivityMetrics.activity_id == activity_id
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# RECORDS – vteřinová data
# ═══════════════════════════════════════════════════════════════════════════

RECORD_COLUMNS = [
    "activity_id", "timestamp", "heart_rate", "speed", "power", "cadence",
    "altitude", "distance", "temperature", "vertical_oscillation", "stance_time",
    "respiratory_rate", "hrv", "position_lat", "position_long", "hr_zone",
    "is_active", "trimp_increment",
]


def _copy_value(v: Any) -> str:
    """Serializace pro COPY ... WITH CSV. None → prázdné pole = NULL."""
    v = _clean_value(v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return "t" if v else "f"
    s = str(v)
    if any(ch in s for ch in (',', '"', "\n", "\r")):
        return '"' + s.replace('"', '""') + '"'
    return s


def copy_records(session: Session, rows: Iterable[dict], chunk_size: int = 50_000) -> int:
    """
    Bulk zápis vteřinových dat přes COPY.

    COPY běží na **spojení dané session**, ne na vlastním. Je to podstatné:
    loader nejdřív smaže stará vteřinová data aktivity a teprve pak zapisuje
    nová. Kdyby COPY jel po svém spojení, čekal by na zámek, který drží
    nezacommitovaný DELETE téže session – proces by se zablokoval navždy.

    Duplicity řeší staging tabulka; COPY sám by na konfliktu spadl a
    re-import jedné aktivity je běžná operace. Při shodě
    (activity_id, timestamp) vyhrává **poslední** zapsaný řádek: historické
    CSV obsahovalo tytéž aktivity naparsované dvakrát pod různými definicemi
    tepových zón a novější parse je ten správný. Pořadí drží sloupec _seq,
    protože SQL sám žádné pořadí řádků negarantuje.
    """
    connection = session.connection()
    dbapi_conn = connection.connection
    total = 0

    with dbapi_conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _records_stage "
            "(LIKE records INCLUDING DEFAULTS, _seq bigserial) ON COMMIT DROP"
        )
        cur.execute("TRUNCATE _records_stage")

        buf = io.StringIO()
        n_buf = 0

        def flush() -> int:
            nonlocal buf, n_buf
            if n_buf == 0:
                return 0
            buf.seek(0)
            with cur.copy(
                f"COPY _records_stage ({', '.join(RECORD_COLUMNS)}) "
                "FROM STDIN WITH (FORMAT csv)"
            ) as copy:
                copy.write(buf.read())
            written = n_buf
            buf = io.StringIO()
            n_buf = 0
            return written

        for row in rows:
            buf.write(",".join(_copy_value(row.get(c)) for c in RECORD_COLUMNS) + "\n")
            n_buf += 1
            if n_buf >= chunk_size:
                total += flush()
        total += flush()

        cols = ", ".join(RECORD_COLUMNS)
        value_cols = [c for c in RECORD_COLUMNS if c not in ("activity_id", "timestamp")]

        # Sloučení fragmentů: pro každou vteřinu se z každého sloupce vezme
        # nejnovější NEPRÁZDNÁ hodnota.
        #
        # Nutné kvůli Strava exportům, které jednu vteřinu rozdělují do
        # několika record zpráv, každou s jinou podmnožinou polí:
        #     15:44:45  {distance: 38.24}
        #     15:44:45  {speed: 0.018, position_lat: …}
        #     15:44:45  {heart_rate: 88}
        # Prosté „ponech poslední" by z té vteřiny nechalo jen tep a zbytek
        # přepsalo NULLy. Řazení podle _seq DESC zároveň zajistí, že při
        # skutečném re-importu vyhraje novější parse.
        merged = ",\n       ".join(
            f"(array_agg({c} ORDER BY _seq DESC) "
            f"FILTER (WHERE {c} IS NOT NULL))[1] AS {c}"
            for c in value_cols
        )
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in value_cols)

        cur.execute(
            f"INSERT INTO records ({cols})\n"
            f"SELECT activity_id, timestamp,\n       {merged}\n"
            "FROM _records_stage\n"
            "GROUP BY activity_id, timestamp\n"
            f"ON CONFLICT (activity_id, timestamp) DO UPDATE SET {updates}"
        )

    return total


def delete_records(session: Session, activity_id: str) -> None:
    """Smaže vteřinová data aktivity (re-import po změně FIT souboru)."""
    session.execute(delete(Record).where(Record.activity_id == activity_id))


def read_records(session: Session, activity_id: str) -> pd.DataFrame:
    """Vteřinová data jedné aktivity, seřazená podle času."""
    stmt = (
        select(Record)
        .where(Record.activity_id == activity_id)
        .order_by(Record.timestamp)
    )
    df = pd.DataFrame(
        [
            {c.name: getattr(r, c.name) for c in Record.__table__.columns}
            for r in session.scalars(stmt)
        ]
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def iter_records(session: Session, activity_ids: Sequence[str]) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Postupně vydává (activity_id, DataFrame) – paměť roste jen s největší
    jednotlivou aktivitou, ne s celou databází.
    """
    for aid in activity_ids:
        yield aid, read_records(session, aid)


def downsample_records(
    session: Session, activity_id: str, bucket: str = "10 seconds"
) -> pd.DataFrame:
    """
    Agregace přes TimescaleDB time_bucket pro grafy – frontend nikdy
    nedostane 40 tisíc syrových bodů.
    """
    sql = text(
        """
        SELECT time_bucket(CAST(:bucket AS interval), timestamp) AS ts,
               avg(heart_rate)  AS heart_rate,
               avg(speed)       AS speed,
               avg(power)       AS power,
               avg(cadence)     AS cadence,
               avg(altitude)    AS altitude,
               max(distance)    AS distance,
               avg(temperature) AS temperature
        FROM records
        WHERE activity_id = :aid
        GROUP BY ts
        ORDER BY ts
        """
    )
    rows = session.execute(sql, {"bucket": bucket, "aid": activity_id}).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


# ═══════════════════════════════════════════════════════════════════════════
# DAILY BIOMETRICS / METRICS
# ═══════════════════════════════════════════════════════════════════════════

def upsert_biometrics(session: Session, rows: list[dict]) -> int:
    return upsert(session, DailyBiometrics, rows, index_elements=["date", "source"])


# Priorita zdrojů biometrie. Garmin má přednost: měří přes noc s lepší
# detekcí spánku, zatímco Apple Watch reportuje jinou hodnotu (mediány
# 49 vs 46 bpm). Zdroje se proto neslévají, jen se vybírá.
SOURCE_PRIORITY = ["garmin", "apple"]


def read_biometrics_resolved(
    session: Session, since: date | None = None, until: date | None = None
) -> pd.DataFrame:
    """
    Jedna řádka na den – z každého dne vítězí zdroj s nejvyšší prioritou,
    který pro daný den vůbec něco naměřil.

    Sloupec `source` zůstává ve výstupu, aby bylo dohledatelné, odkud
    hodnota pochází. Analytika i chatbot tak nikdy nepracují s hodnotou
    neznámého původu.
    """
    df = read_biometrics(session, since=since, until=until)
    if df.empty:
        return df

    if "source" not in df.columns:
        return df

    order = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    df = df.assign(_rank=df["source"].map(order).fillna(len(order)))
    df = df.sort_values(["date", "_rank"]).drop_duplicates("date", keep="first")
    return df.drop(columns=["_rank"]).reset_index(drop=True)


def upsert_daily_metrics(session: Session, rows: list[dict]) -> int:
    return upsert(session, DailyMetrics, rows, index_elements=["date"])


def _read_daily(session: Session, model, since, until) -> pd.DataFrame:
    stmt = select(model)
    if since is not None:
        stmt = stmt.where(model.date >= since)
    if until is not None:
        stmt = stmt.where(model.date <= until)
    stmt = stmt.order_by(model.date)
    df = pd.DataFrame(
        [
            {c.name: getattr(r, c.name) for c in model.__table__.columns}
            for r in session.scalars(stmt)
        ]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_biometrics(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    return _read_daily(session, DailyBiometrics, since, until)


def read_daily_metrics(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    return _read_daily(session, DailyMetrics, since, until)


def read_daily_metrics_row(session: Session, day: date) -> DailyMetrics | None:
    return session.get(DailyMetrics, day)


def min_data_date(session: Session) -> date | None:
    """
    Nejstarší datum napříč aktivitami A biometrií.

    Klíčové pro kalendář: dřívější verze začínala první aktivitou, takže
    biometrie z doby před prvním zaznamenaným tréninkem se zahazovala.
    """
    a_min = session.scalar(select(Activity.date).order_by(Activity.date).limit(1))
    b_min = session.scalar(select(DailyBiometrics.date).order_by(DailyBiometrics.date).limit(1))
    candidates = [d for d in (a_min, b_min) if d is not None]
    return min(candidates) if candidates else None


def min_activity_date(session: Session) -> date | None:
    """Datum nejstarší aktivity – počátek denní osy analytiky."""
    return session.scalar(select(Activity.date).order_by(Activity.date).limit(1))


def delete_daily_metrics_before(session: Session, cutoff: date) -> int:
    """
    Smaže denní metriky před počátkem osy.

    Nutné po zúžení kalendáře: persist_daily_metrics jen upsertuje, takže
    dny mimo novou osu by v tabulce zůstaly a export by je dál sypal do CSV.
    """
    result = session.execute(delete(DailyMetrics).where(DailyMetrics.date < cutoff))
    return result.rowcount or 0


# ═══════════════════════════════════════════════════════════════════════════
# SYNC STATE
# ═══════════════════════════════════════════════════════════════════════════

def get_state(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(SyncState, key)
    return row.value if row is not None else default


def set_state(session: Session, key: str, value: dict) -> None:
    upsert(session, SyncState, [{"key": key, "value": value}], index_elements=["key"])
