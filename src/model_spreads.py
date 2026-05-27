"""
Model 1: Spread Prediction (Z-spread primary, G-spread fallback).

LightGBM regressor with monotonic constraints and issuer-grouped 5-fold CV.
Target: z_spread_bid (bps); rows where z_spread missing use g_spread_bid.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import lightgbm as lgb

from src.config import (
    DATA_PROCESSED,
    FEATURES_SPREADS,
    FIGURES,
    SPREAD_FALLBACK,
    SPREAD_MAX_BPS,
    SPREAD_MIN_BPS,
    SPREAD_TARGET,
    REPORTS,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

_FUNDAMENTAL_FEATURES = [
    "Total Debt / EBITDA (x)",
    "net_debt_to_ebitda",
    "total_debt_to_assets",
    "debt_to_equity",
    "EBITDA / Interest Expense (x)",
    "EBIT / Interest Expense (x)",
    "fcf_to_debt",
    "cfo_to_debt",
    "EBITDA Margin",
    "log_assets",
    "Current Ratio (x)",
    "Quick Ratio (x)",
    "cash_to_debt",
]

_BOND_FEATURES = [
    "time_to_maturity_yrs",
    "age_yrs",
    "log_amt_outstanding",
    "seniority_senior_unsecured",
    "rating_numeric",
    "rating_is_ig",
    "modified_duration",
    "convexity",
]

_MACRO_FEATURES = [
    "hy_oas",
    "ig_oas",
    "treasury_10y",
    "vix",
    "yield_curve",
    "hy_oas_30d_chg",
    "treasury_10y_30d_chg",
    "vix_30d_chg",
]

ALL_FEATURES = _FUNDAMENTAL_FEATURES + _BOND_FEATURES + _MACRO_FEATURES

# Monotonic constraint direction: +1 = feature↑ → OAS↑, -1 = feature↑ → OAS↓, 0 = unconstrained
_MONOTONE_MAP = {
    "Total Debt / EBITDA (x)":      1,
    "net_debt_to_ebitda":            1,
    "total_debt_to_assets":          1,
    "debt_to_equity":                1,
    "EBITDA / Interest Expense (x)": -1,
    "EBIT / Interest Expense (x)":   -1,
    "fcf_to_debt":                   -1,
    "cfo_to_debt":                   -1,
    "Current Ratio (x)":             -1,
    "Quick Ratio (x)":               -1,
    "cash_to_debt":                  -1,
    "rating_numeric":                1,   # higher numeric = worse rating = wider spread
    "modified_duration":             1,
    "hy_oas":                        1,
    "ig_oas":                        1,
    "vix":                           1,
}

_LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=8,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    n_estimators=500,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _build_spread_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the spread target column.

    Priority: z_spread_bid → g_spread_bid → a_spread_bid.
    Records which source was used in 'spread_source'.
    """
    df = df.copy()
    df["spread_target"] = np.nan
    df["spread_source"] = ""

    for col in [SPREAD_TARGET, SPREAD_FALLBACK, "a_spread_bid"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            fill = df["spread_target"].isna() & s.notna()
            df.loc[fill, "spread_target"] = s[fill]
            df.loc[fill, "spread_source"] = col

    return df


def prepare_spread_data(df: pd.DataFrame) -> tuple:
    """
    Filter and prepare the feature matrix for Model 1.

    Returns (X, y, groups, feature_cols, df_filtered).
    """
    df = _build_spread_target(df)

    mask = (
        df["spread_target"].notna()
        & (df["spread_target"] >= SPREAD_MIN_BPS)
        & (df["spread_target"] <= SPREAD_MAX_BPS)
    )
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"Spread filter: dropping {n_dropped} rows (spread missing or extreme)")
    df = df[mask].copy()

    src_counts = df["spread_source"].value_counts()
    print(f"Spread sources: {src_counts.to_dict()}")

    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"INFO: {len(missing)} features absent from data: {missing[:5]}{'...' if len(missing)>5 else ''}")

    X = df[feature_cols].copy()
    y = df["spread_target"].copy()
    groups = df["mi_key"].astype(str)

    print(f"Spread model: {len(X)} samples, {len(feature_cols)} features, {groups.nunique()} issuers")
    return X, y, groups, feature_cols, df


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_spread_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """
    Issuer-grouped 5-fold CV.
    Returns dict with fold metrics, OOF predictions, and trained models.
    """
    monotone = [_MONOTONE_MAP.get(f, 0) for f in feature_cols]

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    oof_preds = np.full(len(y), np.nan)
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = dict(_LGB_PARAMS)
        params["monotone_constraints"] = monotone

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        residuals = y_val - preds
        ss_res = (residuals ** 2).sum()
        ss_tot = ((y_val - y_val.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt((residuals ** 2).mean())

        from scipy.stats import spearmanr
        rho, _ = spearmanr(y_val, preds)

        fold_metrics.append({"fold": fold + 1, "r2": r2, "rmse": rmse, "spearman": rho, "n_val": len(y_val)})
        models.append(model)
        print(f"  Fold {fold+1}: R²={r2:.3f}  RMSE={rmse:.1f}bps  Spearman={rho:.3f}  (n={len(y_val)})")

    metrics_df = pd.DataFrame(fold_metrics)
    print(
        f"\nCV summary: R²={metrics_df['r2'].mean():.3f}±{metrics_df['r2'].std():.3f}  "
        f"RMSE={metrics_df['rmse'].mean():.1f}±{metrics_df['rmse'].std():.1f}bps  "
        f"Spearman={metrics_df['spearman'].mean():.3f}±{metrics_df['spearman'].std():.3f}"
    )

    if metrics_df["r2"].mean() > 0.80:
        print("WARNING: R² > 0.80 out-of-sample. Investigate for data leakage before proceeding.")

    return {
        "fold_metrics": metrics_df,
        "oof_predictions": oof_preds,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Final model on full data
# ---------------------------------------------------------------------------

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> lgb.LGBMRegressor:
    """Train final model on all data (used for the RV framework)."""
    monotone = [_MONOTONE_MAP.get(f, 0) for f in feature_cols]
    params = dict(_LGB_PARAMS)
    params["monotone_constraints"] = monotone

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    return model


# ---------------------------------------------------------------------------
# SHAP for top residuals
# ---------------------------------------------------------------------------

def compute_shap_top_residuals(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    oof_preds: np.ndarray,
    df: pd.DataFrame,
    n_top: int = 5,
) -> pd.DataFrame:
    """SHAP values for top-n largest positive and negative residuals."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        residuals = y.values - oof_preds
        valid = ~np.isnan(residuals)
        sorted_idx = np.argsort(residuals[valid])
        top_neg = np.where(valid)[0][sorted_idx[:n_top]]
        top_pos = np.where(valid)[0][sorted_idx[-n_top:]]
        top_idx = np.concatenate([top_neg, top_pos])

        shap_df = pd.DataFrame(shap_vals[top_idx], columns=X.columns)
        shap_df["residual"] = residuals[top_idx]
        shap_df["issuer"] = df["issuer_name"].values[top_idx] if "issuer_name" in df.columns else ""
        return shap_df
    except Exception as e:
        print(f"SHAP computation skipped: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_spread_model(df: pd.DataFrame | None = None) -> dict:
    """
    Full spread model pipeline.

    Reads feature_matrix_spreads.parquet if df is None.
    Returns dict with model, metrics, OOF predictions, feature importance.
    """
    if df is None:
        df = pd.read_parquet(FEATURES_SPREADS)

    print("=== Model 1: Spread Prediction (Z-spread / G-spread / A-spread) ===")
    X, y, groups, feature_cols, df_filtered = prepare_spread_data(df)

    if len(X) < 20:
        raise ValueError(f"Too few samples for CV: {len(X)}. Check spread data availability.")

    cv_result = run_spread_cv(X, y, groups, feature_cols)

    # Final model
    final_model = train_final_model(X, y, feature_cols)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)

    # Save metrics
    REPORTS.mkdir(parents=True, exist_ok=True)
    cv_result["fold_metrics"].to_csv(REPORTS / "spread_cv_metrics.csv", index=False)
    importance_df.to_csv(REPORTS / "spread_feature_importance.csv", index=False)

    # OOF residuals — the cheapness signal for RV framework
    oof_df = df_filtered.reset_index(drop=True).copy()
    oof_df["predicted_spread"] = cv_result["oof_predictions"]
    oof_df["cheapness_bps"] = oof_df["spread_target"] - oof_df["predicted_spread"]

    save_cols = ["issuer_name", "cusip", "instrument_id", "mi_key",
                 "spread_target", "spread_source", "predicted_spread", "cheapness_bps",
                 "sp_rating", "rating_numeric", "modified_duration"]
    oof_df[[c for c in save_cols if c in oof_df.columns]].to_csv(
        REPORTS / "spread_oof_residuals.csv", index=False
    )

    print(f"\nFeature importance (top 10):")
    print(importance_df.head(10).to_string(index=False))

    return {
        "model": final_model,
        "feature_cols": feature_cols,
        "cv_metrics": cv_result["fold_metrics"],
        "oof_df": oof_df,
        "importance_df": importance_df,
    }


if __name__ == "__main__":
    from src.parse_capital_iq import parse_bonddata, parse_financial_highlights
    from src.crosswalk import run_crosswalk
    from src.features import run_feature_engineering
    from src.macro import attach_macro

    bonds = parse_bonddata()
    fund = parse_financial_highlights()
    cw = run_crosswalk(bonds)
    spreads_df, _ = run_feature_engineering(bonds, cw, fund)
    spreads_df = attach_macro(spreads_df)
    spreads_df.to_parquet(FEATURES_SPREADS, index=False)

    run_spread_model(spreads_df)
