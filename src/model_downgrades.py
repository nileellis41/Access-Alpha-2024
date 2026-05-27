"""
Model 2: Downgrade Risk.

Two models ensembled:
  2a. LightGBM classifier (binary, scale_pos_weight, PR-AUC)
  2b. Cox proportional hazards (lifelines, time-to-downgrade)

Validation: time-forward holdout (not random split, not grouped-only).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

from src.config import FEATURES_DOWNGRADES, REPORTS
from src.model_spreads import _MONOTONE_MAP  # reuse sign conventions

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns for downgrade model
# ---------------------------------------------------------------------------

_DOWNGRADE_FEATURES = [
    # Level features
    "Total Debt / EBITDA (x)",
    "net_debt_to_ebitda",
    "EBITDA / Interest Expense (x)",
    "EBIT / Interest Expense (x)",
    "fcf_to_debt",
    "cfo_to_debt",
    "Current Ratio (x)",
    "cash_to_debt",
    "log_assets",
    # Trajectory features (alpha for downgrade prediction)
    "total_debt__to__ebitda_delta_1yr",
    "total_debt__to__ebitda_delta_2yr",
    "ebitda__to__interest_expense_delta_1yr",
    "ebitda__to__interest_expense_delta_2yr",
    "ebitda_margin_delta_1yr",
    "vol_2yr_ebitda_margin",
    "distance_to_max_leverage",
    # Rating
    "rating_numeric",
]

_DOWNGRADE_MONOTONE = {
    "Total Debt / EBITDA (x)":            1,
    "net_debt_to_ebitda":                  1,
    "EBITDA / Interest Expense (x)":       -1,
    "fcf_to_debt":                         -1,
    "Current Ratio (x)":                   -1,
    "rating_numeric":                      1,
    "total_debt__to__ebitda_delta_1yr":       1,
    "total_debt__to__ebitda_delta_2yr":       1,
    "ebitda__to__interest_expense_delta_1yr": -1,
    "distance_to_max_leverage":            1,
}

_LGB_CLF_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.03,
    num_leaves=15,
    min_data_in_leaf=5,
    feature_fraction=0.7,
    lambda_l2=2.0,
    n_estimators=300,
    verbose=-1,
)

_PERIOD_ORDER = ["2021Q1", "2022Q1", "2023Q1", "2024Q1", "2025Q1"]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prepare_downgrade_data(df: pd.DataFrame) -> tuple:
    """
    Prepare issuer-period data for downgrade model.
    Returns (X, y, periods) sorted by period for time-forward validation.
    """
    # Only training periods (exclude 2026Q1 — no forward labels)
    df = df[df["period"].isin(_PERIOD_ORDER)].copy()

    feature_cols = [f for f in _DOWNGRADE_FEATURES if f in df.columns]
    missing = [f for f in _DOWNGRADE_FEATURES if f not in df.columns]
    if missing:
        print(f"Downgrade model — missing features: {missing}")

    X = df[feature_cols].copy()
    y = df["downgrade_next_yr"].copy()
    periods = df["period"].copy()

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(
        f"Downgrade data: {len(X)} issuer-periods, {int(n_pos)} positive "
        f"({n_pos/len(y):.1%} base rate), features={len(feature_cols)}"
    )
    return X, y, periods, feature_cols, n_pos, n_neg


# ---------------------------------------------------------------------------
# Time-forward validation split
# ---------------------------------------------------------------------------

def time_forward_split(periods: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Train: all periods except the last two.
    Validate: second-to-last period.
    Test:  last period.

    Returns (train_idx, val_idx, test_idx).
    """
    sorted_periods = sorted(periods.unique())
    n = len(sorted_periods)
    if n < 3:
        raise ValueError(f"Need at least 3 periods for time-forward split; got {n}")

    train_periods = sorted_periods[:n - 2]
    val_periods = sorted_periods[n - 2: n - 1]
    test_periods = sorted_periods[n - 1:]

    train_idx = np.where(periods.isin(train_periods))[0]
    val_idx = np.where(periods.isin(val_periods))[0]
    test_idx = np.where(periods.isin(test_periods))[0]

    print(
        f"Time-forward split: train={train_periods}, "
        f"val={val_periods}, test={test_periods}"
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Model 2a: LightGBM classifier
# ---------------------------------------------------------------------------

def train_lgb_classifier(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    scale_pos_weight: float,
) -> lgb.LGBMClassifier:
    monotone = [_DOWNGRADE_MONOTONE.get(f, 0) for f in feature_cols]
    params = dict(_LGB_CLF_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    params["monotone_constraints"] = monotone

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


# ---------------------------------------------------------------------------
# Model 2b: Cox proportional hazards
# ---------------------------------------------------------------------------

def train_cox(df: pd.DataFrame, feature_cols: list[str]) -> object:
    """
    Fit a Cox PH model using lifelines.
    Duration = position of period in sorted order (1..N).
    Event = downgrade_next_yr.
    """
    try:
        from lifelines import CoxPHFitter

        period_to_int = {p: i + 1 for i, p in enumerate(_PERIOD_ORDER)}
        available_cols = [f for f in feature_cols if f in df.columns]
        cox_df = df[available_cols + ["downgrade_next_yr", "period"]].copy()
        cox_df["duration"] = cox_df["period"].map(period_to_int).astype(float)
        # Drop non-numeric columns before fitting
        cox_fit_cols = available_cols + ["duration", "downgrade_next_yr"]
        cox_df = cox_df[cox_fit_cols].dropna()

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(
            cox_df,
            duration_col="duration",
            event_col="downgrade_next_yr",
            show_progress=False,
        )
        print("Cox model fitted.")
        return cph
    except Exception as e:
        print(f"Cox model failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_predict(
    lgb_model: lgb.LGBMClassifier,
    cox_model,
    X: pd.DataFrame,
    df_with_period: pd.DataFrame,
    feature_cols: list[str],
    horizon_periods: int = 1,
) -> np.ndarray:
    """
    Mean of (calibrated LGB probability, 1 − Cox survival over horizon).
    Falls back to LGB-only if Cox unavailable.
    """
    lgb_prob = lgb_model.predict_proba(X[feature_cols])[:, 1]

    if cox_model is None:
        return lgb_prob

    try:
        cox_feature_cols = [f for f in feature_cols if f in cox_model.params_.index]
        cox_df = X[cox_feature_cols].copy().fillna(0)
        sf = cox_model.predict_survival_function(cox_df, times=[horizon_periods])
        cox_prob = 1 - sf.values[0]
        return (lgb_prob + cox_prob) / 2
    except Exception:
        return lgb_prob


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_downgrade_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> dict:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        brier_score_loss,
    )

    n = len(y_true)
    n_pos = y_true.sum()
    if n_pos == 0:
        print(f"  {label}: no positive cases, skipping metrics")
        return {}

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Recall at top decile
    thresh_decile = np.percentile(y_prob, 90)
    top_decile_mask = y_prob >= thresh_decile
    recall_at_decile = y_true[top_decile_mask].sum() / n_pos if n_pos else np.nan

    metrics = {
        "split": label,
        "n": n,
        "n_positive": int(n_pos),
        "base_rate": n_pos / n,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "recall_at_top_decile": recall_at_decile,
    }
    print(
        f"  {label}: PR-AUC={pr_auc:.3f}  ROC-AUC={roc_auc:.3f}  "
        f"Brier={brier:.3f}  Recall@Decile={recall_at_decile:.2f}  (n={n}, pos={int(n_pos)})"
    )
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_downgrade_model(df: pd.DataFrame | None = None) -> dict:
    """
    Full downgrade model pipeline. Returns dict with model, metrics, predictions.
    """
    if df is None:
        df = pd.read_parquet(FEATURES_DOWNGRADES)

    print("=== Model 2: Downgrade Risk ===")
    X, y, periods, feature_cols, n_pos, n_neg = prepare_downgrade_data(df)
    df_train_data = df[df["period"].isin(_PERIOD_ORDER)].reset_index(drop=True)

    if len(X) < 10 or n_pos < 2:
        print("WARNING: Insufficient data for downgrade model. Skipping.")
        return {}

    try:
        train_idx, val_idx, test_idx = time_forward_split(periods)
    except ValueError as e:
        print(f"WARNING: {e}. Falling back to single split.")
        n = len(X)
        train_idx = np.arange(int(n * 0.7))
        val_idx = np.arange(int(n * 0.7), int(n * 0.85))
        test_idx = np.arange(int(n * 0.85), n)

    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    # 2a. LGB
    lgb_model = train_lgb_classifier(X_tr, y_tr, X_val, y_val, feature_cols, scale_pos_weight)

    # 2b. Cox (on training data only)
    cox_model = train_cox(df_train_data.iloc[train_idx], feature_cols)

    # Evaluate
    all_metrics = []
    for split_label, xi, yi in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(xi) == 0 or yi.sum() == 0:
            continue
        df_split = df_train_data.iloc[val_idx if split_label == "val" else test_idx]
        probs = ensemble_predict(lgb_model, cox_model, xi, df_split, feature_cols)
        m = compute_downgrade_metrics(yi.values, probs, label=split_label)
        all_metrics.append(m)

    # Full-data predictions (for RV framework)
    df_all = df[df["period"].isin(_PERIOD_ORDER)].reset_index(drop=True)
    X_all = df_all[[f for f in feature_cols if f in df_all.columns]].copy()
    missing_cols = [f for f in feature_cols if f not in X_all.columns]
    for c in missing_cols:
        X_all[c] = np.nan
    X_all = X_all[feature_cols]

    all_probs = ensemble_predict(lgb_model, cox_model, X_all, df_all, feature_cols)
    df_all["downgrade_prob_4q"] = all_probs

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": lgb_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)

    REPORTS.mkdir(parents=True, exist_ok=True)
    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(REPORTS / "downgrade_metrics.csv", index=False)
    importance_df.to_csv(REPORTS / "downgrade_feature_importance.csv", index=False)
    df_all[["mi_key", "period", "downgrade_next_yr", "downgrade_prob_4q"]].to_csv(
        REPORTS / "downgrade_predictions.csv", index=False
    )

    # Cox summary
    if cox_model is not None:
        cox_summary = cox_model.summary
        cox_summary.to_csv(REPORTS / "cox_summary.csv")
        print("\nCox hazard ratios (top 5 by |coef|):")
        print(cox_summary.sort_values("coef", key=abs, ascending=False).head(5)[["coef", "exp(coef)", "p"]].to_string())

    print(f"\nFeature importance (top 8):")
    print(importance_df.head(8).to_string(index=False))

    return {
        "lgb_model": lgb_model,
        "cox_model": cox_model,
        "feature_cols": feature_cols,
        "metrics": all_metrics,
        "downgrade_df": df_all,
        "importance_df": importance_df,
    }


if __name__ == "__main__":
    run_downgrade_model()
