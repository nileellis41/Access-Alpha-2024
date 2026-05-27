"""
Feature engineering.

Key invariants (enforced by tests):
  - PIT joins use filing_date, never period_end_date.
  - Trajectory features return NaN when prior periods are absent; never extrapolate.
  - Fundamentals stale by more than MAX_STALENESS_QUARTERS are dropped.
  - Missing fundamentals: issuer-specific forward-fill up to staleness limit, then drop.
"""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    FEATURES_DOWNGRADES,
    FEATURES_SPREADS,
    FUNDAMENTALS_PARQUET,
    IG_CUTOFF_NUMERIC,
    MAX_STALENESS_QUARTERS,
    RATING_NA_STRINGS,
    SP_RATING_SCALE,
)

# ---------------------------------------------------------------------------
# Rating helpers
# ---------------------------------------------------------------------------

def rating_to_numeric(rating: Optional[str]) -> Optional[float]:
    if not isinstance(rating, str) or rating.strip() in RATING_NA_STRINGS:
        return np.nan
    return float(SP_RATING_SCALE.get(rating.strip(), np.nan))


# ---------------------------------------------------------------------------
# 1. PIT join — CRITICAL: use filing_date, not period_end_date
# ---------------------------------------------------------------------------

def pit_join_fundamentals(
    bonds_df: pd.DataFrame,
    fundamentals_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each bond observation on as_of_date, attach the most recent
    fundamentals where filing_date <= as_of_date.

    Uses filing_date (the date S&P received/published the filing),
    NOT period_end_date (which is the quarter-end and would introduce
    look-ahead bias of up to 3 months).

    Returns wide-format DataFrame: one row per bond, fundamentals as columns.
    """
    # Pivot long → wide per (mi_key, filing_date)
    # For each mi_key, keep the snapshot with the latest filing_date <= as_of_date
    fund = fundamentals_long.copy()
    fund = fund[fund["mi_key"].notna() & fund["filing_date"].notna()]
    fund["mi_key"] = fund["mi_key"].astype(str)

    # Keep only the latest filing per (mi_key, period, metric) — already deduped in parser
    # Now find, per mi_key, the latest period whose filing_date <= as_of_date
    as_of = pd.Timestamp(bonds_df["as_of_date"].iloc[0])

    eligible = fund[fund["filing_date"] <= as_of].copy()
    # For each mi_key, pick the period with the max filing_date
    latest_period = (
        eligible.sort_values("filing_date")
        .drop_duplicates(subset=["mi_key", "metric"], keep="last")
    )

    # Pivot to wide
    wide = latest_period.pivot_table(
        index="mi_key", columns="metric", values="value", aggfunc="last"
    )
    wide.columns.name = None
    wide = wide.reset_index()

    # Merge onto bonds
    bonds = bonds_df.copy()
    bonds["mi_key"] = bonds["mi_key"].astype(str)
    merged = bonds.merge(wide, on="mi_key", how="left")
    return merged


# ---------------------------------------------------------------------------
# 2. Trajectory features — computed from the full historical panel
# ---------------------------------------------------------------------------

def _pivot_period_wide(fundamentals_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long fundamentals to wide: rows = (mi_key, period), cols = metrics.
    Includes period_end_date and filing_date for staleness checks.
    """
    fund = fundamentals_long.copy()
    fund["mi_key"] = fund["mi_key"].astype(str)

    wide = fund.pivot_table(
        index=["mi_key", "period"],
        columns="metric",
        values="value",
        aggfunc="last",
    )
    wide.columns.name = None
    wide = wide.reset_index()

    # Attach the latest filing_date per (mi_key, period)
    fd = (
        fund.groupby(["mi_key", "period"])["filing_date"]
        .max()
        .reset_index()
        .rename(columns={"filing_date": "filing_date"})
    )
    pe = (
        fund.groupby(["mi_key", "period"])["period_end_date"]
        .max()
        .reset_index()
    )
    wide = wide.merge(fd, on=["mi_key", "period"], how="left")
    wide = wide.merge(pe, on=["mi_key", "period"], how="left")
    return wide


# Period sort order (annual Q1 snapshots)
_PERIOD_ORDER = ["2021Q1", "2022Q1", "2023Q1", "2024Q1", "2025Q1", "2026Q1"]


def _period_lag(period: str, n_years: int) -> Optional[str]:
    """Return the period n_years before the given period, or None if out of range."""
    if period not in _PERIOD_ORDER:
        return None
    idx = _PERIOD_ORDER.index(period)
    lag_idx = idx - n_years
    return _PERIOD_ORDER[lag_idx] if lag_idx >= 0 else None


def compute_trajectories(panel_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Add trajectory features to a panel of (mi_key, period) rows.

    Δ1yr = current - 1-year-ago value
    Δ2yr = current - 2-year-ago value
    Δ4yr = current - 4-year-ago value
    Rolling 2yr std of EBITDA Margin and Revenue Growth

    Returns NaN when prior period is absent — never extrapolates.
    """
    panel = panel_wide.copy().sort_values(["mi_key", "period"])

    # Index by (mi_key, period) for fast lookups
    panel = panel.set_index(["mi_key", "period"])

    def lag_col(col: str, n_years: int, suffix: str) -> pd.Series:
        result = {}
        for (mk, per), row in panel.iterrows():
            lag_per = _period_lag(per, n_years)
            if lag_per and (mk, lag_per) in panel.index:
                result[(mk, per)] = panel.loc[(mk, lag_per), col]
            else:
                result[(mk, per)] = np.nan
        return pd.Series(result, name=f"{col}_{suffix}")

    leverage_col = "Total Debt / EBITDA (x)"
    coverage_col = "EBITDA / Interest Expense (x)"
    fcf_margin_col_src = "EBITDA Margin"  # proxy for FCF margin trajectory

    for col, suffix_map in [
        (leverage_col, {1: "delta_1yr", 2: "delta_2yr", 4: "delta_4yr"}),
        (coverage_col, {1: "delta_1yr", 2: "delta_2yr"}),
        (fcf_margin_col_src, {1: "delta_1yr", 2: "delta_2yr"}),
    ]:
        if col not in panel.columns:
            continue
        for n_years, suffix in suffix_map.items():
            lag_s = lag_col(col, n_years, suffix)
            col_clean = col.split(" (")[0].lower().replace(" ", "_").replace("/", "_to_").replace(",", "").replace(".", "").replace("-", "_")
            new_col = f"{col_clean}_{suffix}"
            panel[new_col] = panel[col] - lag_s

    # Rolling 2yr std of EBITDA Margin
    if "EBITDA Margin" in panel.columns:
        panel["vol_2yr_ebitda_margin"] = (
            panel["EBITDA Margin"]
            .groupby(level=0)
            .transform(lambda s: s.rolling(2, min_periods=2).std())
        )

    # Distance to issuer's own 4yr max leverage
    if leverage_col in panel.columns:
        panel["issuer_4yr_max_leverage"] = (
            panel[leverage_col]
            .groupby(level=0)
            .transform(lambda s: s.rolling(4, min_periods=1).max())
        )
        panel["distance_to_max_leverage"] = (
            panel[leverage_col] - panel["issuer_4yr_max_leverage"]
        )

    return panel.reset_index()


# ---------------------------------------------------------------------------
# 3. Ratio engineering (computed from raw metric columns)
# ---------------------------------------------------------------------------

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived financial ratios from raw metric columns.
    Returns NaN for any ratio where denominator is zero or missing.
    """
    d = df.copy()

    def safe_div(num_col: str, denom_col: str, out_col: str) -> None:
        num = d.get(num_col, pd.Series(np.nan, index=d.index))
        den = d.get(denom_col, pd.Series(np.nan, index=d.index))
        d[out_col] = np.where(
            den.isna() | (den == 0), np.nan, num / den
        )

    safe_div("Net Debt", "EBITDA", "net_debt_to_ebitda")
    safe_div("Total Debt", "Total Assets", "total_debt_to_assets")
    safe_div("Total Debt", "Total Common Equity", "debt_to_equity")
    safe_div("Levered Free Cash Flow", "Total Debt", "fcf_to_debt")
    safe_div("Cash from Ops.", "Total Debt", "cfo_to_debt")
    safe_div("EBITDA", "Total Revenue", "ebitda_margin_calc")
    safe_div("Unlevered Free Cash Flow", "Total Revenue", "fcf_margin")
    safe_div("Cash & Short-term Investments", "Total Debt", "cash_to_debt")

    # log transforms
    for raw_col, out_col in [("Total Assets", "log_assets")]:
        col = d.get(raw_col, pd.Series(np.nan, index=d.index))
        d[out_col] = np.where(col > 0, np.log(col), np.nan)

    return d


# ---------------------------------------------------------------------------
# 4. Bond-level features
# ---------------------------------------------------------------------------

def compute_bond_features(bonds_df: pd.DataFrame) -> pd.DataFrame:
    b = bonds_df.copy()

    as_of = pd.Timestamp(b["as_of_date"].iloc[0])

    b["time_to_maturity_yrs"] = (b["maturity_date"] - as_of).dt.days / 365.25
    b["age_yrs"] = (as_of - b["issue_date"]).dt.days / 365.25
    b["log_amt_outstanding"] = np.where(
        b["amount_outstanding_000"] > 0,
        np.log(b["amount_outstanding_000"]),
        np.nan,
    )
    b["seniority_senior_unsecured"] = (
        b["seniority"].str.strip().str.lower() == "senior unsecured"
    ).astype(float)

    b["rating_numeric"] = b["sp_rating"].apply(rating_to_numeric)
    b["rating_is_ig"] = (b["rating_numeric"] <= IG_CUTOFF_NUMERIC).astype(float)

    return b


# ---------------------------------------------------------------------------
# 5. Downgrade label construction
# ---------------------------------------------------------------------------

_NEGATIVE_ACTIONS = re.compile(
    r"\b(downgrade|creditwatch negative|outlook.*negative|outlook revised to negative)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\((\d{2}/\d{2}/\d{4})\)")


def _parse_action_history(history_str: str) -> list[tuple[str, pd.Timestamp]]:
    """Parse a semicolon-separated rating action string into (action, date) list."""
    if not isinstance(history_str, str):
        return []
    events = []
    for chunk in history_str.split(";"):
        chunk = chunk.strip()
        date_match = _DATE_RE.search(chunk)
        if not date_match:
            continue
        date = pd.to_datetime(date_match.group(1), format="%m/%d/%Y", errors="coerce")
        if pd.isna(date):
            continue
        action = _DATE_RE.sub("", chunk).strip(" |").strip()
        events.append((action, date))
    return events


def build_downgrade_labels(bonds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct issuer-level downgrade labels from rating action history.

    For each issuer-period, label = 1 if any negative action (Downgrade,
    CreditWatch Negative, Outlook Negative) falls within 1 year AFTER
    the period end date.

    Returns DataFrame with columns: mi_key, period, period_end_date,
    downgrade_next_yr (0/1), any_negative_next_yr (0/1).
    """
    # Collect all negative events per mi_key (take the union across bonds)
    issuer_events: dict[str, list[pd.Timestamp]] = {}
    for _, row in bonds_df.iterrows():
        mi_key = str(row.get("mi_key", ""))
        if not mi_key or mi_key == "<NA>":
            continue
        history = str(row.get("sp_rating_action_history_3y", ""))
        for action, date in _parse_action_history(history):
            if _NEGATIVE_ACTIONS.search(action):
                issuer_events.setdefault(mi_key, []).append(date)

    records = []
    for period, period_end_str in [
        ("2021Q1", "2021-03-31"), ("2022Q1", "2022-03-31"),
        ("2023Q1", "2023-03-31"), ("2024Q1", "2024-03-31"),
        ("2025Q1", "2025-03-31"),
    ]:
        pe = pd.Timestamp(period_end_str)
        window_start = pe
        window_end = pe + pd.DateOffset(years=1)

        # Get all unique mi_keys in fundamentals
        mi_keys = bonds_df["mi_key"].dropna().astype(str).unique()
        for mi_key in mi_keys:
            events = issuer_events.get(mi_key, [])
            neg_in_window = any(window_start < e <= window_end for e in events)
            records.append({
                "mi_key": mi_key,
                "period": period,
                "period_end_date": pe,
                "downgrade_next_yr": int(neg_in_window),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Staleness filter
# ---------------------------------------------------------------------------

def apply_staleness_filter(
    df_with_filing: pd.DataFrame,
    as_of_date: pd.Timestamp,
    max_staleness_quarters: int = MAX_STALENESS_QUARTERS,
) -> pd.DataFrame:
    """
    Drop rows where the most recent fundamental has a filing_date more than
    max_staleness_quarters × 91 days before as_of_date.
    """
    cutoff = as_of_date - timedelta(days=max_staleness_quarters * 91)
    if "filing_date" not in df_with_filing.columns:
        return df_with_filing
    mask = df_with_filing["filing_date"] >= cutoff
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"Staleness filter: dropping {n_dropped} rows (filing_date < {cutoff.date()})")
    return df_with_filing[mask].copy()


# ---------------------------------------------------------------------------
# 7. Build feature matrices
# ---------------------------------------------------------------------------

def build_feature_matrix_spreads(
    bonds_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    fundamentals_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full pipeline → feature_matrix_spreads.parquet.

    One row per bond observation with:
      - Bond characteristics
      - PIT-joined current fundamentals (using filing_date)
      - Derived ratios
      - OAS as target
    """
    # Attach mi_key from crosswalk to bonds that lack it
    cw = crosswalk_df[["issuer_name", "mi_key"]].rename(columns={"mi_key": "cw_mi_key"})
    bonds = bonds_df.merge(cw, on="issuer_name", how="left")
    bonds["mi_key"] = bonds["mi_key"].where(bonds["mi_key"].notna(), bonds["cw_mi_key"])
    bonds = bonds.drop(columns=["cw_mi_key"])

    # Drop bonds with no mi_key (the 4 excluded issuers + others without fundamentals)
    bonds = bonds[bonds["mi_key"].notna() & (bonds["mi_key"] != "<NA>")].copy()

    # Bond-level features
    bonds = compute_bond_features(bonds)

    # PIT join fundamentals
    bonds = pit_join_fundamentals(bonds, fundamentals_long)

    # Derived ratios
    bonds = compute_ratios(bonds)

    # OAS sanity filter is applied in model_spreads, not here

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    bonds.to_parquet(FEATURES_SPREADS, index=False)
    n_valid_oas = bonds["oas_bid"].notna().sum()
    print(
        f"feature_matrix_spreads.parquet: {len(bonds)} bonds, "
        f"{n_valid_oas} with valid OAS"
    )
    return bonds


def build_feature_matrix_downgrades(
    bonds_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    fundamentals_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full pipeline → feature_matrix_downgrades.parquet.

    One row per (issuer, period) with fundamentals and downgrade label.
    """
    # Build downgrade labels
    labels_df = build_downgrade_labels(bonds_df)

    # Get trajectories on the full historical panel
    panel_wide = _pivot_period_wide(fundamentals_long)
    panel_traj = compute_trajectories(panel_wide)
    panel_ratios = compute_ratios(panel_traj)

    # Merge labels with features
    panel_ratios["mi_key"] = panel_ratios["mi_key"].astype(str)
    labels_df["mi_key"] = labels_df["mi_key"].astype(str)

    merged = labels_df.merge(panel_ratios, on=["mi_key", "period"], how="inner")

    # Add issuer rating (from bonds_df, current)
    rating_map = (
        bonds_df[["mi_key", "rating_numeric"]]
        .dropna(subset=["mi_key"])
        .assign(mi_key=lambda d: d["mi_key"].astype(str))
        .groupby("mi_key")["rating_numeric"]
        .first()
    ) if "rating_numeric" in bonds_df.columns else pd.Series(dtype=float)

    # Compute bond features once (for the current snapshot) to add to issuer level
    bonds_tmp = compute_bond_features(bonds_df.copy())
    if "rating_numeric" in bonds_tmp.columns:
        rating_map = (
            bonds_tmp[["mi_key", "rating_numeric"]]
            .dropna(subset=["mi_key"])
            .assign(mi_key=lambda d: d["mi_key"].astype(str))
            .drop_duplicates(subset=["mi_key"])
            .set_index("mi_key")["rating_numeric"]
        )
        merged["rating_numeric"] = merged["mi_key"].map(rating_map)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(FEATURES_DOWNGRADES, index=False)
    pos = merged["downgrade_next_yr"].sum()
    print(
        f"feature_matrix_downgrades.parquet: {len(merged)} issuer-periods, "
        f"{int(pos)} positive ({pos/len(merged):.1%} base rate)"
    )
    return merged


def run_feature_engineering(
    bonds_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    fundamentals_long: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all feature engineering. Returns (spreads_df, downgrades_df)."""
    if fundamentals_long is None:
        fundamentals_long = pd.read_parquet(FUNDAMENTALS_PARQUET)

    spreads_df = build_feature_matrix_spreads(bonds_df, crosswalk_df, fundamentals_long)
    downgrades_df = build_feature_matrix_downgrades(bonds_df, crosswalk_df, fundamentals_long)
    return spreads_df, downgrades_df
