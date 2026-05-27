"""
FRED macro overlay.

Pulls HY OAS, IG OAS, 10y Treasury, VIX, and yield curve slope.
Attaches point-in-time values and 30-day changes to a bond DataFrame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FRED_API_KEY, FRED_SERIES

_CACHE_DIR = Path(__file__).parent.parent / "data" / "interim" / "macro_cache"


def _fetch_series(series_id: str, start: str = "2020-01-01") -> pd.Series:
    """Fetch one FRED series. Returns daily Series indexed by date."""
    cache_file = _CACHE_DIR / f"{series_id}.parquet"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        s = pd.read_parquet(cache_file).squeeze()
        return s

    if not FRED_API_KEY:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to .env or set the environment variable."
        )

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(series_id, observation_start=start)
    s.name = series_id
    s.to_frame().to_parquet(cache_file)
    return s


def get_macro_as_of(
    as_of_date: pd.Timestamp,
    series_map: dict[str, str] = FRED_SERIES,
) -> dict[str, float]:
    """
    Return macro values on or before as_of_date, plus 30-day changes.
    Values are forward-filled (FRED series are not published every day).
    """
    result = {}
    for col_name, series_id in series_map.items():
        try:
            s = _fetch_series(series_id)
            s = s[s.index <= as_of_date].dropna()
            if s.empty:
                result[col_name] = np.nan
                result[f"{col_name}_30d_chg"] = np.nan
                continue

            current = float(s.iloc[-1])
            result[col_name] = current

            # 30-day change
            cutoff_30d = as_of_date - pd.Timedelta(days=30)
            s_30d = s[s.index <= cutoff_30d].dropna()
            result[f"{col_name}_30d_chg"] = (
                current - float(s_30d.iloc[-1]) if not s_30d.empty else np.nan
            )
        except Exception as e:
            print(f"WARNING: Could not fetch FRED series {series_id}: {e}")
            result[col_name] = np.nan
            result[f"{col_name}_30d_chg"] = np.nan

    return result


def attach_macro(bonds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro columns to bonds_df using point-in-time values on each bond's as_of_date.

    For a single-date snapshot (our case), all bonds get the same macro values.
    """
    as_of = pd.Timestamp(bonds_df["as_of_date"].iloc[0])
    macro_vals = get_macro_as_of(as_of)
    df = bonds_df.copy()
    for k, v in macro_vals.items():
        df[k] = v
    return df
