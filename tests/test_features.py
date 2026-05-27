"""
Tests for src/features.py.

Run: pytest tests/test_features.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    _parse_action_history,
    apply_staleness_filter,
    build_downgrade_labels,
    compute_bond_features,
    compute_ratios,
    compute_trajectories,
    rating_to_numeric,
    _pivot_period_wide,
)
from src.config import SP_RATING_SCALE, RATING_NA_STRINGS


# ---------------------------------------------------------------------------
# Rating numeric scale
# ---------------------------------------------------------------------------

def test_rating_scale_all_ratings_present():
    for r in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
              "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
              "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]:
        assert rating_to_numeric(r) is not np.nan, f"Rating {r!r} missing from scale"
        assert isinstance(rating_to_numeric(r), float)


@pytest.mark.parametrize("na_str", ["NR", "WD", "", "N/A", "nan", "NaN", "SD"])
def test_rating_na_strings_return_nan(na_str):
    result = rating_to_numeric(na_str)
    assert np.isnan(result), f"Expected NaN for {na_str!r}, got {result}"


def test_rating_monotonic_ig_before_hy():
    assert SP_RATING_SCALE["BBB-"] < SP_RATING_SCALE["BB+"], \
        "IG/HY boundary must be monotonic: BBB- < BB+"


# ---------------------------------------------------------------------------
# Trajectory features
# ---------------------------------------------------------------------------

def _make_panel():
    """Create a minimal panel with 4 periods for one issuer."""
    rows = []
    for i, period in enumerate(["2022Q1", "2023Q1", "2024Q1", "2025Q1"]):
        rows.append({
            "mi_key": "AAAA",
            "period": period,
            "period_end_date": pd.Timestamp(f"202{i+2}-03-31"),
            "filing_date": pd.Timestamp(f"202{i+2}-05-01"),
            "Total Debt / EBITDA (x)": 2.0 + i * 0.5,  # 2.0, 2.5, 3.0, 3.5
            "EBITDA / Interest Expense (x)": 10.0 - i * 1.0,
            "EBITDA Margin": 30.0 - i * 2.0,
        })
    return pd.DataFrame(rows)


def test_trajectory_delta_1yr_correct():
    panel = _make_panel()
    result = compute_trajectories(panel)
    result = result.set_index(["mi_key", "period"])

    # 2025Q1: leverage = 3.5, 2024Q1 leverage = 3.0 → delta = 0.5
    col = "total_debt_/_ebitda_x_delta_1yr"
    # Find the trajectory column for leverage
    traj_cols = [c for c in result.columns if "debt" in c.lower() and "delta_1yr" in c]
    assert traj_cols, f"No 1yr leverage trajectory column found; columns: {list(result.columns)}"
    val = result.loc[("AAAA", "2025Q1"), traj_cols[0]]
    assert abs(val - 0.5) < 1e-9, f"Expected Δ1yr leverage = 0.5, got {val}"


def test_trajectory_nan_for_missing_prior():
    """Earliest period (2022Q1) has no prior → Δ1yr must be NaN."""
    panel = _make_panel()
    result = compute_trajectories(panel)
    result = result.set_index(["mi_key", "period"])

    traj_cols = [c for c in result.columns if "debt" in c.lower() and "delta_1yr" in c]
    if traj_cols:
        val = result.loc[("AAAA", "2022Q1"), traj_cols[0]]
        assert np.isnan(val), \
            f"Expected NaN for earliest period Δ1yr, got {val}"


def test_trajectory_no_extrapolation():
    """Trajectory for a period with no prior data must be NaN, never an extrapolated value."""
    # Panel with only one observation
    panel = pd.DataFrame([{
        "mi_key": "BBBB", "period": "2025Q1",
        "period_end_date": pd.Timestamp("2025-03-31"),
        "filing_date": pd.Timestamp("2025-05-01"),
        "Total Debt / EBITDA (x)": 4.0,
        "EBITDA / Interest Expense (x)": 5.0,
        "EBITDA Margin": 20.0,
    }])
    result = compute_trajectories(panel)
    traj_cols = [c for c in result.columns if "delta" in c]
    for col in traj_cols:
        val = result[col].iloc[0]
        assert np.isnan(val), f"Expected NaN for single-period issuer in {col}, got {val}"


def test_trajectory_multi_issuer_no_cross_contamination():
    """Trajectory for issuer A must not bleed into issuer B."""
    rows = []
    for mk, base in [("AAAA", 2.0), ("BBBB", 10.0)]:
        for i, period in enumerate(["2024Q1", "2025Q1"]):
            rows.append({
                "mi_key": mk, "period": period,
                "period_end_date": pd.Timestamp(f"202{i+4}-03-31"),
                "filing_date": pd.Timestamp(f"202{i+4}-05-01"),
                "Total Debt / EBITDA (x)": base + i,
                "EBITDA / Interest Expense (x)": 5.0,
                "EBITDA Margin": 20.0,
            })
    panel = pd.DataFrame(rows)
    result = compute_trajectories(panel).set_index(["mi_key", "period"])

    traj_cols = [c for c in result.columns if "debt" in c.lower() and "delta_1yr" in c]
    if traj_cols:
        col = traj_cols[0]
        # A: 2025Q1 leverage = 3.0, 2024Q1 = 2.0 → delta = 1.0
        # B: 2025Q1 leverage = 11.0, 2024Q1 = 10.0 → delta = 1.0
        val_a = result.loc[("AAAA", "2025Q1"), col]
        val_b = result.loc[("BBBB", "2025Q1"), col]
        assert abs(val_a - 1.0) < 1e-9, f"Issuer A delta wrong: {val_a}"
        assert abs(val_b - 1.0) < 1e-9, f"Issuer B delta wrong: {val_b}"


# ---------------------------------------------------------------------------
# Ratio engineering
# ---------------------------------------------------------------------------

def test_ratios_no_div_by_zero():
    df = pd.DataFrame([{
        "Net Debt": 1000.0, "EBITDA": 0.0,
        "Total Debt": 500.0, "Total Assets": 0.0,
        "Total Common Equity": 0.0,
        "Levered Free Cash Flow": 100.0,
        "Cash from Ops.": 200.0,
        "Cash & Short-term Investments": 50.0,
        "Total Revenue": 0.0,
        "Unlevered Free Cash Flow": 80.0,
    }])
    result = compute_ratios(df)
    for col in ["net_debt_to_ebitda", "total_debt_to_assets", "debt_to_equity",
                "fcf_margin", "ebitda_margin_calc"]:
        assert col in result.columns
        assert np.isnan(result[col].iloc[0]), f"{col} should be NaN when denom=0"


def test_ratios_log_assets_positive_only():
    df = pd.DataFrame([
        {"Total Assets": 1_000_000.0},
        {"Total Assets": -100.0},
        {"Total Assets": 0.0},
        {"Total Assets": np.nan},
    ])
    result = compute_ratios(df)
    assert "log_assets" in result.columns
    assert not np.isnan(result.loc[0, "log_assets"])
    for i in [1, 2, 3]:
        assert np.isnan(result.loc[i, "log_assets"]), f"Row {i} log_assets should be NaN"


# ---------------------------------------------------------------------------
# Bond features
# ---------------------------------------------------------------------------

def test_bond_features_time_to_maturity():
    bonds = pd.DataFrame([{
        "issuer_name": "Test", "instrument_id": "SPS1",
        "mi_key": "999", "cusip": "AAA",
        "as_of_date": pd.Timestamp("2026-05-27"),
        "maturity_date": pd.Timestamp("2031-05-27"),
        "issue_date": pd.Timestamp("2021-05-27"),
        "amount_outstanding_000": 500_000.0,
        "seniority": "Senior Unsecured",
        "sp_rating": "BBB",
    }])
    result = compute_bond_features(bonds)
    assert abs(result["time_to_maturity_yrs"].iloc[0] - 5.0) < 0.01
    assert abs(result["age_yrs"].iloc[0] - 5.0) < 0.01
    assert result["seniority_senior_unsecured"].iloc[0] == 1.0
    assert result["rating_numeric"].iloc[0] == SP_RATING_SCALE["BBB"]
    assert result["rating_is_ig"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Staleness filter
# ---------------------------------------------------------------------------

def test_staleness_filter_drops_stale_rows():
    as_of = pd.Timestamp("2026-05-27")
    stale_cutoff = as_of - pd.Timedelta(days=2 * 91)

    df = pd.DataFrame([
        {"mi_key": "A", "filing_date": as_of - pd.Timedelta(days=30)},   # fresh
        {"mi_key": "B", "filing_date": stale_cutoff - pd.Timedelta(days=1)},  # stale
    ])
    result = apply_staleness_filter(df, as_of, max_staleness_quarters=2)
    assert len(result) == 1
    assert result["mi_key"].iloc[0] == "A"


# ---------------------------------------------------------------------------
# Downgrade label parsing
# ---------------------------------------------------------------------------

def test_parse_action_history_extracts_downgrades():
    history = "Downgrade (06/09/2025); Upgrade (02/27/2025); CreditWatch/Outlook (12/17/2024)"
    events = _parse_action_history(history)
    assert len(events) == 3
    assert any("Downgrade" in act and d == pd.Timestamp("2025-06-09") for act, d in events)


def test_parse_action_history_empty():
    assert _parse_action_history("") == []
    assert _parse_action_history(None) == []
    assert _parse_action_history(float("nan")) == []


def test_downgrade_labels_shape(tmp_path):
    """Labels must have one row per (issuer, period) combination."""
    bonds = pd.DataFrame([
        {"mi_key": "111", "sp_rating_action_history_3y": "Downgrade (06/09/2025)"},
        {"mi_key": "222", "sp_rating_action_history_3y": "Upgrade (02/27/2025)"},
    ])
    labels = build_downgrade_labels(bonds)
    # 2 issuers × 5 historical periods
    assert len(labels) == 2 * 5
    assert "downgrade_next_yr" in labels.columns
    assert labels["downgrade_next_yr"].isin([0, 1]).all()


def test_downgrade_labels_correct_flag():
    """Downgrade on 2025-06-09 should flag 2025Q1 (window: 2025-03-31 to 2026-03-31)."""
    bonds = pd.DataFrame([{
        "mi_key": "111",
        "sp_rating_action_history_3y": "Downgrade (06/09/2025)",
    }])
    labels = build_downgrade_labels(bonds)
    row_2025 = labels[labels["period"] == "2025Q1"]
    assert len(row_2025) == 1
    assert row_2025["downgrade_next_yr"].iloc[0] == 1

    row_2021 = labels[labels["period"] == "2021Q1"]
    assert len(row_2021) == 1
    assert row_2021["downgrade_next_yr"].iloc[0] == 0
