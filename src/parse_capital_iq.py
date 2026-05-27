"""
Parsers for S&P Capital IQ exports.

Three file formats:
  1. bonddata.csv  — wide with metadata rows; bond market + embedded fundamentals
  2. mikey.csv     — entity → MI Key crosswalk
  3. FinancialHighlights_*.csv — transposed; companies as columns, metrics as rows
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BONDDATA_CSV,
    BOND_AS_OF_DATE,
    BONDS_PARQUET,
    DATA_INTERIM,
    FH_2026_CSV,
    FH_5Y_DIR,
    FUNDAMENTALS_PARQUET,
    MIKEY_CSV,
    MIKEY_PARQUET,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURRENT_RE = re.compile(r"^\s*current\s*$", re.IGNORECASE)
_SECTION_LABELS = {
    "Balance Sheet ($000)",
    "Income Statement ($000)",
    "Cash Flow ($000)",
    "Profitability (%)",
    "Per Share Information ($)",
}
_BLANK_METRIC = re.compile(r"^\s*$")


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """Coerce to float; turn 'Current', blanks, and bracketed negatives to NaN or float."""
    def _clean_val(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if _CURRENT_RE.match(s) or s in ("", "NA", "N/A", "—", "-"):
            return np.nan
        # bracketed negatives: (1,234.5) → -1234.5
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(_clean_val)


def _parse_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def _period_to_quarter(period_str: str) -> str | None:
    """Convert 'FQ12026' → '2026Q1', 'FY2025' → '2025Q4', etc."""
    if not isinstance(period_str, str):
        return None
    m = re.match(r"FQ(\d)(20\d{2})", period_str.strip())
    if m:
        return f"{m.group(2)}Q{m.group(1)}"
    m2 = re.match(r"FY(20\d{2})", period_str.strip())
    if m2:
        return f"{m2.group(1)}Q4"
    return period_str.strip()


# ---------------------------------------------------------------------------
# 1. bonddata.csv
# ---------------------------------------------------------------------------

_BOND_COL_MAP = {
    # (position_index): (clean_name)
    0: "issuer_name",
    1: "instrument_id",
    2: "mi_key",
    3: "cusip",
    4: "description",
    10: "coupon",
    11: "offering_yield",
    12: "bid_price",
    13: "ask_price",
    14: "amount_outstanding_000",
    15: "maturity_date",
    16: "issue_date",
    17: "trade_volume_30d",
    18: "trade_volume_60d",
    19: "trade_volume_90d",
    21: "bid_ask_spread",
    22: "pv01",
    23: "accrued_interest",
    24: "ytm_mid",
    25: "ytm_bid",
    28: "z_spread_bid",
    29: "g_spread_bid",
    30: "oas_bid",
    31: "a_spread_bid",
    32: "macaulay_duration",
    33: "modified_duration",
    34: "convexity",
    35: "sp_rating_action_current",
    36: "sp_rating_action_history_3y",
    37: "sp_rating_action_history_2y",
    38: "sp_creditwatch_outlook",
    39: "sp_rating",
    40: "sp_rating_date",
    42: "fixed_income_type",
    43: "seniority",
    # embedded current-period fundamentals
    44: "total_debt_to_ebitda_cur",
    45: "rev_growth_1y",
    48: "ebitda_interest_cov_cur",
    49: "ebitda_cur",
    52: "net_debt_cur",
    53: "current_ratio_cur",
    54: "total_debt_cur",
    55: "total_assets_cur",
    56: "cfo_cur",
    61: "unlevered_fcf_cur",
    62: "levered_fcf_cur",
    63: "capex_to_rev_cur",
    110: "total_equity_cur",
}

_NUMERIC_BOND_COLS = {
    "coupon", "offering_yield", "bid_price", "ask_price", "amount_outstanding_000",
    "trade_volume_30d", "trade_volume_60d", "trade_volume_90d",
    "bid_ask_spread", "pv01", "accrued_interest",
    "ytm_mid", "ytm_bid", "z_spread_bid", "g_spread_bid", "oas_bid",
    "a_spread_bid", "macaulay_duration", "modified_duration", "convexity",
    "total_debt_to_ebitda_cur", "rev_growth_1y", "ebitda_interest_cov_cur",
    "ebitda_cur", "net_debt_cur", "current_ratio_cur", "total_debt_cur",
    "total_assets_cur", "cfo_cur", "unlevered_fcf_cur", "levered_fcf_cur",
    "capex_to_rev_cur", "total_equity_cur",
}


def parse_bonddata(path: Path = BONDDATA_CSV) -> pd.DataFrame:
    """
    Parse bonddata.csv → clean DataFrame saved to bonds.parquet.

    File structure:
      row 0: SPGTable metadata       → skipped
      row 1: human column labels     → used as header
      row 2: SPT field codes         → skipped
      row 3: qualifier row           → skipped
      row 4+: bond observations
    """
    raw = pd.read_csv(
        path,
        skiprows=[0, 2, 3],
        header=0,
        encoding="utf-8-sig",
        low_memory=False,
        dtype=str,  # read everything as str; we'll coerce below
    )

    # Select and rename columns by position
    selected = {}
    for pos, name in _BOND_COL_MAP.items():
        if pos < len(raw.columns):
            selected[name] = raw.iloc[:, pos]

    df = pd.DataFrame(selected)

    # Coerce numeric columns
    for col in _NUMERIC_BOND_COLS:
        if col in df.columns:
            df[col] = _to_numeric_clean(df[col])

    # Parse dates
    df["maturity_date"] = _parse_date_col(df["maturity_date"])
    df["issue_date"] = _parse_date_col(df["issue_date"])
    df["sp_rating_date"] = _parse_date_col(df.get("sp_rating_date", pd.Series(dtype=str)))

    # Issuer name cleanup
    df["issuer_name"] = df["issuer_name"].str.strip()

    # mi_key as nullable integer string
    df["mi_key"] = pd.to_numeric(df["mi_key"], errors="coerce").astype("Int64").astype(str)
    df["mi_key"] = df["mi_key"].replace("<NA>", pd.NA)

    # as_of_date
    df["as_of_date"] = pd.Timestamp(BOND_AS_OF_DATE)

    # Forward-fill issuer_name for continuation rows (bonds where CIQ left issuer blank)
    df["issuer_name"] = df["issuer_name"].replace("", pd.NA).ffill()

    # Keep rows that have an instrument ID (SPS...) OR a CUSIP; drop pure artifacts
    has_sps = df["instrument_id"].str.startswith("SPS", na=False)
    has_cusip = df["cusip"].notna() & (df["cusip"].str.strip() != "")
    df = df[has_sps | has_cusip].copy()
    df = df.reset_index(drop=True)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BONDS_PARQUET, index=False)
    print(f"bonds.parquet: {len(df)} rows, {df['issuer_name'].nunique()} unique issuers")
    return df


# ---------------------------------------------------------------------------
# 2. FinancialHighlights (transposed CIQ export)
# ---------------------------------------------------------------------------

def _parse_financial_highlights_file(path: Path) -> pd.DataFrame:
    """
    Parse one FinancialHighlights_*.csv → long-format DataFrame.

    File structure (0-indexed lines):
      0: SPGTable metadata
      1: MI KEYs (col 2+)
      2: Period code (FQ12026 etc.) for each MI KEY
      3: Qualifier (Current/Restated etc.)
      4: Period Ended date for each MI KEY
      5: Financial Filing Date for each MI KEY   ← CRITICAL for PIT joins
      6+: metric rows; col 0 = metric name, col 2+ = values
    """
    raw_lines = open(path, encoding="utf-8-sig", errors="replace").readlines()

    def _split(line: str) -> list[str]:
        # Simple CSV split (handles basic quoting)
        import csv, io
        return next(csv.reader(io.StringIO(line)))

    mi_keys_raw = _split(raw_lines[1])       # row 1
    periods_raw = _split(raw_lines[2])        # row 2
    period_ends_raw = _split(raw_lines[4])    # row 4
    filing_dates_raw = _split(raw_lines[5])   # row 5

    # Columns 2+ are the MI KEYs
    start = 2
    n_companies = max(len(mi_keys_raw), len(periods_raw)) - start

    mi_keys = [mi_keys_raw[i + start].strip() if i + start < len(mi_keys_raw) else "" for i in range(n_companies)]
    periods = [periods_raw[i + start].strip() if i + start < len(periods_raw) else "" for i in range(n_companies)]
    period_ends = [period_ends_raw[i + start].strip() if i + start < len(period_ends_raw) else "" for i in range(n_companies)]
    filing_dates = [filing_dates_raw[i + start].strip() if i + start < len(filing_dates_raw) else "" for i in range(n_companies)]

    records = []
    for line in raw_lines[6:]:
        parts = _split(line)
        if not parts:
            continue
        metric = parts[0].strip() if parts else ""
        if not metric or _BLANK_METRIC.match(metric) or metric in _SECTION_LABELS:
            continue

        for i in range(n_companies):
            mi_key = mi_keys[i]
            if not mi_key:
                continue
            val_raw = parts[i + start].strip() if i + start < len(parts) else ""
            val = _to_numeric_clean(pd.Series([val_raw])).iloc[0]

            records.append({
                "mi_key": mi_key,
                "period": _period_to_quarter(periods[i]) if periods[i] else "",
                "period_end_date": pd.to_datetime(period_ends[i], errors="coerce"),
                "filing_date": pd.to_datetime(filing_dates[i], errors="coerce"),
                "metric": metric,
                "value": val,
                "source_file": path.name,
            })

    df = pd.DataFrame(records)
    return df


def parse_financial_highlights(
    fh_2026: Path = FH_2026_CSV,
    fh_5y_dir: Path = FH_5Y_DIR,
) -> pd.DataFrame:
    """
    Parse all FinancialHighlights files → single long-format panel.
    Resolves Current/Restated priority: if both exist for a period, prefer Restated.
    Saves to fundamentals_panel.parquet.
    """
    all_dfs = []

    # 2026 (current quarter)
    if fh_2026.exists():
        df_ = _parse_financial_highlights_file(fh_2026)
        all_dfs.append(df_)

    # 5y historical files
    if fh_5y_dir.exists():
        for p in sorted(fh_5y_dir.glob("FinancialHighlights_*.csv")):
            df_ = _parse_financial_highlights_file(p)
            all_dfs.append(df_)

    if not all_dfs:
        raise FileNotFoundError("No FinancialHighlights files found.")

    df = pd.concat(all_dfs, ignore_index=True)

    # Drop rows with no MI key, no value, no filing date
    df = df[df["mi_key"].notna() & (df["mi_key"] != "")]
    df = df[df["filing_date"].notna()]

    # Deduplicate: if same mi_key + period + metric appears in multiple files,
    # prefer the one with the later filing_date (most restated)
    df = (
        df.sort_values("filing_date")
        .drop_duplicates(subset=["mi_key", "period", "metric"], keep="last")
        .reset_index(drop=True)
    )

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUNDAMENTALS_PARQUET, index=False)

    n_issuers = df["mi_key"].nunique()
    n_periods = df["period"].nunique()
    print(
        f"fundamentals_panel.parquet: {len(df):,} rows, "
        f"{n_issuers} issuers, {n_periods} periods"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Entry point
# ---------------------------------------------------------------------------

def run_all_parsers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all parsers in dependency order. Returns (bonds_df, fundamentals_df)."""
    print("=== Parsing bonddata ===")
    bonds_df = parse_bonddata()

    print("\n=== Parsing FinancialHighlights ===")
    fund_df = parse_financial_highlights()

    return bonds_df, fund_df


if __name__ == "__main__":
    run_all_parsers()
