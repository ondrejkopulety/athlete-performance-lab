"""
sanitize.py  –  převod pandas/ORM hodnot na tvar, který spolkne DB a JSON
========================================================================

Vytažené z ``repository.py``, protože je to samostatná starost: analytika
počítá v pandas (NaN/NaT/pd.NA, numpy skaláry), ale Postgres přes psycopg
ani ``json.dumps`` je neberou. Každý zápis do DB i každá odpověď API tudy
musí projít.

Tři vrstvy:
  * ``clean_value`` – jedna hodnota → None / python skalár
  * ``records_to_dicts`` – DataFrame → list dictů pro upsert
  * ``copy_value`` – jedna hodnota → pole pro ``COPY ... WITH CSV``
  * ``orm_rows_to_df`` – ORM objekty → DataFrame (dřív šest kopií téhož
    dict-comprehension přes ``__table__.columns``)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def clean_value(v: Any) -> Any:
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
        out.append({k: clean_value(v) for k, v in row.items()})
    return out


def copy_value(v: Any) -> str:
    """Serializace pro COPY ... WITH CSV. None → prázdné pole = NULL."""
    v = clean_value(v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return "t" if v else "f"
    s = str(v)
    if any(ch in s for ch in (",", '"', "\n", "\r")):
        return '"' + s.replace('"', '""') + '"'
    return s


def orm_rows_to_df(rows: Iterable, model) -> pd.DataFrame:
    """
    ORM objekty → DataFrame se sloupci přesně podle tabulky modelu.

    ``columns=`` drží tvar i pro prázdný výsledek, takže volající si můžou
    dělat ``df["sloupec"]`` bez kontroly na prázdno.
    """
    cols = [c.name for c in model.__table__.columns]
    return pd.DataFrame(
        [{c: getattr(r, c) for c in cols} for r in rows], columns=cols
    )
