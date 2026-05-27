"""
Tests for src/parse_capital_iq.py.

Run: pytest tests/test_parsers.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.parse_capital_iq import parse_bonddata, parse_financial_highlights


# ---------------------------------------------------------------------------
# bonddata
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bonds_df():
    return parse_bonddata()


def test_bonddata_row_count(bonds_df):
    assert len(bonds_df) == 168, f"Expected 168 bond rows, got {len(bonds_df)}"


def test_bonddata_unique_issuers(bonds_df):
    n = bonds_df["issuer_name"].nunique()
    assert n == 72, f"Expected 72 unique issuers, got {n}"


def test_bonddata_required_columns(bonds_df):
    required = [
        "issuer_name", "instrument_id", "cusip", "oas_bid",
        "modified_duration", "convexity", "sp_rating",
        "sp_rating_action_history_3y", "maturity_date", "as_of_date",
        "seniority", "fixed_income_type",
    ]
    missing = [c for c in required if c not in bonds_df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_bonddata_oas_is_numeric(bonds_df):
    assert pd.api.types.is_float_dtype(bonds_df["oas_bid"]), "oas_bid must be float"
    valid = bonds_df["oas_bid"].dropna()
    assert len(valid) > 0, "No valid OAS values"
    assert (valid >= 0).all(), "Negative OAS values found"


def test_bonddata_no_current_strings_in_numeric_cols(bonds_df):
    """'Current' qualifier strings must not leak into numeric columns."""
    numeric_cols = ["oas_bid", "modified_duration", "convexity", "coupon"]
    for col in numeric_cols:
        if col in bonds_df.columns:
            # Column must be float dtype — string values would make it object
            assert bonds_df[col].dtype != object, \
                f"Column {col!r} is object dtype; likely has 'Current' strings"


def test_bonddata_instrument_id_format(bonds_df):
    """All instrument IDs start with 'SPS'."""
    assert bonds_df["instrument_id"].str.startswith("SPS").all()


def test_bonddata_as_of_date(bonds_df):
    assert (bonds_df["as_of_date"] == pd.Timestamp("2026-05-27")).all()


def test_bonddata_maturity_dates_valid(bonds_df):
    maturities = bonds_df["maturity_date"].dropna()
    assert len(maturities) > 0
    # All known maturities must be after issue date
    both = bonds_df[bonds_df["maturity_date"].notna() & bonds_df["issue_date"].notna()]
    assert (both["maturity_date"] > both["issue_date"]).all(), \
        "Found bonds where maturity <= issue date"


# ---------------------------------------------------------------------------
# FinancialHighlights
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fundamentals_df():
    return parse_financial_highlights()


def test_fundamentals_long_format(fundamentals_df):
    required = ["mi_key", "period", "period_end_date", "filing_date", "metric", "value"]
    missing = [c for c in required if c not in fundamentals_df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_fundamentals_row_count(fundamentals_df):
    # 6 files × ~137 companies × ~60 metrics ≈ 49,320 max; after NaN drop expect >> 10k
    assert len(fundamentals_df) > 10_000, \
        f"Suspiciously few rows: {len(fundamentals_df)}"


def test_fundamentals_filing_dates_present(fundamentals_df):
    """Filing dates must exist — they drive PIT joins."""
    missing_fd = fundamentals_df["filing_date"].isna().sum()
    total = len(fundamentals_df)
    pct = missing_fd / total
    assert pct < 0.05, f"{pct:.1%} of rows missing filing_date"


def test_fundamentals_filing_date_dtype(fundamentals_df):
    assert pd.api.types.is_datetime64_any_dtype(fundamentals_df["filing_date"])


def test_fundamentals_period_end_date_dtype(fundamentals_df):
    assert pd.api.types.is_datetime64_any_dtype(fundamentals_df["period_end_date"])


def test_fundamentals_filing_after_period_end_possible(fundamentals_df):
    """
    There exist rows where filing_date > period_end_date.
    This is expected (companies take weeks to file after quarter-end).
    If no such rows exist the filing_date column is wrong.
    """
    both = fundamentals_df[
        fundamentals_df["filing_date"].notna() &
        fundamentals_df["period_end_date"].notna()
    ]
    later_filings = (both["filing_date"] > both["period_end_date"]).sum()
    assert later_filings > 0, \
        "No rows where filing_date > period_end_date — filing_date column may be wrong"


def test_fundamentals_dedup(fundamentals_df):
    """No duplicate (mi_key, period, metric) combinations after restated resolution."""
    dup = fundamentals_df.duplicated(subset=["mi_key", "period", "metric"]).sum()
    assert dup == 0, f"{dup} duplicate (mi_key, period, metric) rows"


def test_fundamentals_key_metrics_present(fundamentals_df):
    key_metrics = ["Total Debt", "EBITDA", "Total Assets", "Cash from Ops."]
    present = set(fundamentals_df["metric"].unique())
    missing = [m for m in key_metrics if m not in present]
    assert not missing, f"Key metrics not found in panel: {missing}"
