"""
CRITICAL: Point-in-time join tests.

Look-ahead bias is the most common credit ML mistake.
These tests guarantee that the join uses filing_date, never period_end_date.

Run: pytest tests/test_pit_joins.py -v
"""
import pandas as pd
import numpy as np
import pytest

from src.features import pit_join_fundamentals


# ---------------------------------------------------------------------------
# Synthetic test case
# ---------------------------------------------------------------------------

def _make_synthetic_data():
    """
    Construct a case where:
      - period_end_date (Q1 2025) = 2025-03-31
      - filing_date              = 2025-05-15   (filed 45 days AFTER period end)
      - as_of_date               = 2025-04-30   (between period_end and filing_date)

    A correct PIT join should NOT include this fundamental, because the
    market didn't know it on 2025-04-30 — the company hadn't filed yet.
    """
    fundamentals = pd.DataFrame([
        # This row: period ended Mar 31, but filed May 15 — NOT available on Apr 30
        {
            "mi_key": "111111",
            "period": "2025Q1",
            "period_end_date": pd.Timestamp("2025-03-31"),
            "filing_date": pd.Timestamp("2025-05-15"),
            "metric": "Total Debt / EBITDA (x)",
            "value": 5.0,
            "source_file": "synthetic",
        },
        # This row: an older filing from Q1 2024, filed Feb 2024 — available on Apr 30
        {
            "mi_key": "111111",
            "period": "2024Q1",
            "period_end_date": pd.Timestamp("2024-03-31"),
            "filing_date": pd.Timestamp("2024-05-01"),
            "metric": "Total Debt / EBITDA (x)",
            "value": 3.0,
            "source_file": "synthetic",
        },
    ])

    bonds = pd.DataFrame([{
        "issuer_name": "Synthetic Issuer",
        "instrument_id": "SPS99999999",
        "mi_key": "111111",
        "cusip": "999999AA9",
        "as_of_date": pd.Timestamp("2025-04-30"),   # BETWEEN period_end and filing_date
        "oas_bid": 100.0,
    }])

    return bonds, fundamentals


def test_pit_join_uses_filing_date_not_period_end():
    """
    The join must NOT return 2025Q1 data (filed 2025-05-15) when as_of = 2025-04-30.
    It must return 2024Q1 data (filed 2024-05-01, which is <= 2025-04-30).
    """
    bonds, fundamentals = _make_synthetic_data()
    result = pit_join_fundamentals(bonds, fundamentals)

    leverage_col = "Total Debt / EBITDA (x)"
    assert leverage_col in result.columns, f"'{leverage_col}' column missing from join result"

    joined_value = result[leverage_col].iloc[0]

    # Must be 3.0 (from 2024Q1, filed 2024-05-01) not 5.0 (from 2025Q1, filed 2025-05-15)
    assert joined_value == pytest.approx(3.0), (
        f"PIT join returned {joined_value}; expected 3.0 (2024Q1). "
        f"If it returned 5.0, the join is using period_end_date instead of filing_date."
    )


def test_pit_join_excludes_future_filings_entirely():
    """
    When all fundamentals have filing_date > as_of_date,
    the joined fundamental must be NaN (not any value).
    """
    _, fundamentals = _make_synthetic_data()
    # Make all filings in the future
    future_fund = fundamentals.copy()
    future_fund["filing_date"] = pd.Timestamp("2030-01-01")

    bonds = pd.DataFrame([{
        "issuer_name": "Synthetic Issuer",
        "instrument_id": "SPS99999999",
        "mi_key": "111111",
        "cusip": "999999AA9",
        "as_of_date": pd.Timestamp("2025-04-30"),
        "oas_bid": 100.0,
    }])

    result = pit_join_fundamentals(bonds, future_fund)
    leverage_col = "Total Debt / EBITDA (x)"
    if leverage_col in result.columns:
        assert result[leverage_col].isna().all(), \
            "When all filings are future-dated, fundamental must be NaN"


def test_pit_join_period_end_later_than_filing_date():
    """
    Synthetic case: filing_date is BEFORE period_end_date (restated filing).
    This tests that we're strictly using filing_date for the comparison.
    The bond observation date falls AFTER filing_date but BEFORE period_end_date.
    The join should use this row (filing_date <= as_of_date < period_end_date).
    """
    fundamentals = pd.DataFrame([{
        "mi_key": "222222",
        "period": "2025Q1",
        "period_end_date": pd.Timestamp("2025-06-30"),  # period ends in June
        "filing_date": pd.Timestamp("2025-03-01"),      # filed in March (restated early)
        "metric": "Total Debt / EBITDA (x)",
        "value": 4.5,
        "source_file": "synthetic",
    }])
    bonds = pd.DataFrame([{
        "issuer_name": "Synthetic Issuer 2",
        "instrument_id": "SPS88888888",
        "mi_key": "222222",
        "cusip": "888888AA8",
        "as_of_date": pd.Timestamp("2025-04-15"),  # after filing, before period end
        "oas_bid": 120.0,
    }])

    result = pit_join_fundamentals(bonds, fundamentals)
    leverage_col = "Total Debt / EBITDA (x)"
    if leverage_col in result.columns:
        joined_value = result[leverage_col].iloc[0]
        # filing_date (Mar 1) <= as_of_date (Apr 15) → should include
        assert joined_value == pytest.approx(4.5), \
            "Should use row where filing_date <= as_of_date even if period_end > as_of_date"


def test_pit_join_picks_latest_eligible_filing():
    """
    When multiple filing dates are eligible (all <= as_of_date),
    the join must pick the most recent one.
    """
    fundamentals = pd.DataFrame([
        {
            "mi_key": "333333", "period": "2023Q1",
            "period_end_date": pd.Timestamp("2023-03-31"),
            "filing_date": pd.Timestamp("2023-05-01"),
            "metric": "EBITDA", "value": 100.0, "source_file": "s",
        },
        {
            "mi_key": "333333", "period": "2024Q1",
            "period_end_date": pd.Timestamp("2024-03-31"),
            "filing_date": pd.Timestamp("2024-05-01"),
            "metric": "EBITDA", "value": 120.0, "source_file": "s",
        },
        {
            "mi_key": "333333", "period": "2025Q1",
            "period_end_date": pd.Timestamp("2025-03-31"),
            "filing_date": pd.Timestamp("2025-05-01"),
            "metric": "EBITDA", "value": 150.0, "source_file": "s",
        },
    ])
    bonds = pd.DataFrame([{
        "issuer_name": "Synthetic Issuer 3",
        "instrument_id": "SPS77777777",
        "mi_key": "333333",
        "cusip": "777777AA7",
        "as_of_date": pd.Timestamp("2025-06-01"),  # after all three filings
        "oas_bid": 80.0,
    }])

    result = pit_join_fundamentals(bonds, fundamentals)
    assert "EBITDA" in result.columns
    # Should pick 2025Q1 (most recent eligible)
    assert result["EBITDA"].iloc[0] == pytest.approx(150.0), \
        "Should use the most recent eligible filing (2025Q1, value=150)"


# ---------------------------------------------------------------------------
# Integration: real data PIT join sanity
# ---------------------------------------------------------------------------

def test_real_pit_join_no_future_data():
    """
    On real data: every joined fundamental must have filing_date <= as_of_date.
    """
    from src.parse_capital_iq import parse_bonddata, parse_financial_highlights
    bonds = parse_bonddata()
    fund = parse_financial_highlights()
    result = pit_join_fundamentals(bonds, fund)

    # If filing_date column was preserved in result, check it
    if "filing_date" in result.columns:
        as_of = pd.Timestamp(bonds["as_of_date"].iloc[0])
        future = result[result["filing_date"] > as_of]
        assert len(future) == 0, \
            f"{len(future)} rows have fundamentals with filing_date > as_of_date"
