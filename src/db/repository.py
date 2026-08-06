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
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.db.models import (
    Activity,
    ActivityMetrics,
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
    return upsert(session, DailyBiometrics, rows, index_elements=["date"])


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


# ═══════════════════════════════════════════════════════════════════════════
# SYNC STATE
# ═══════════════════════════════════════════════════════════════════════════

def get_state(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(SyncState, key)
    return row.value if row is not None else default


def set_state(session: Session, key: str, value: dict) -> None:
    upsert(session, SyncState, [{"key": key, "value": value}], index_elements=["key"])
