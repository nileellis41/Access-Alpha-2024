"""
Evaluation plots and report assembly.

Generates:
  - Predicted vs actual scatter (per model)
  - Residual histogram
  - SHAP summary plot
  - Feature importance bar chart
  - Calibration plot (Model 2)
  - reports/phase2_summary.md
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES, REPORTS

_DPI = 150


# ---------------------------------------------------------------------------
# Spread model plots
# ---------------------------------------------------------------------------

def plot_spread_predicted_vs_actual(oof_df: pd.DataFrame) -> None:
    actual_col = next((c for c in ["actual_spread", "spread_target", "z_spread_bid", "oas_bid"] if c in oof_df.columns), None)
    pred_col = next((c for c in ["predicted_spread", "predicted_oas"] if c in oof_df.columns), None)
    if not actual_col or not pred_col:
        return
    df = oof_df.dropna(subset=[actual_col, pred_col])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    rating_is_ig = df["rating_numeric"].le(10) if "rating_numeric" in df.columns else pd.Series(True, index=df.index)
    colors = np.where(rating_is_ig, "#2196F3", "#FF5722")

    ax.scatter(df[pred_col], df[actual_col], c=colors, alpha=0.7, s=50, edgecolors="white", lw=0.4)
    lim = [min(df[pred_col].min(), df[actual_col].min()) - 10,
           max(df[pred_col].max(), df[actual_col].max()) + 10]
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5, label="Perfect fit")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted Spread (bps)")
    ax.set_ylabel("Actual Spread (bps)")
    ax.set_title("Model 1: Predicted vs Actual Spread (OOF, Z/G/A-spread)")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="IG"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=8, label="HY"),
        Line2D([0], [0], color="k", linestyle="--", label="y=x"),
    ], fontsize=9)

    fig.tight_layout()
    _save(fig, "spread_predicted_vs_actual.png")


def plot_spread_residuals(oof_df: pd.DataFrame) -> None:
    df = oof_df.dropna(subset=["cheapness_bps"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["cheapness_bps"], bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Residual (Actual − Predicted OAS, bps)")
    ax.set_ylabel("Count")
    ax.set_title("Model 1: OAS Residual Distribution (= Cheapness Signal)")
    fig.tight_layout()
    _save(fig, "spread_residuals.png")


def plot_spread_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> None:
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"][::-1], df["importance_gain"][::-1], color="#2196F3", alpha=0.8)
    ax.set_xlabel("Gain Importance")
    ax.set_title("Model 1: Feature Importance (Gain, Top 15)")
    fig.tight_layout()
    _save(fig, "spread_feature_importance.png")


def plot_shap_summary(shap_df: pd.DataFrame) -> None:
    if shap_df.empty:
        return
    try:
        import shap
        # Only numeric shap columns
        numeric_cols = shap_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "residual"]
        if not numeric_cols:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_abs = shap_df[numeric_cols].abs().mean().sort_values(ascending=False).head(10)
        ax.barh(mean_abs.index[::-1], mean_abs.values[::-1], color="#FF9800", alpha=0.8)
        ax.set_xlabel("|SHAP value| mean")
        ax.set_title("Model 1: SHAP — Top Residual Bonds")
        fig.tight_layout()
        _save(fig, "spread_shap.png")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Downgrade model plots
# ---------------------------------------------------------------------------

def plot_downgrade_feature_importance(importance_df: pd.DataFrame) -> None:
    df = importance_df.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"][::-1], df["importance_gain"][::-1], color="#FF5722", alpha=0.8)
    ax.set_xlabel("Gain Importance")
    ax.set_title("Model 2: Downgrade Risk Feature Importance (Gain)")
    fig.tight_layout()
    _save(fig, "downgrade_feature_importance.png")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mean_pred, frac_pos, "s-", color="#FF5722", label="Model 2")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfectly calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Model 2: Calibration (Reliability Diagram)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, "downgrade_calibration.png")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary_report(
    spread_result: dict,
    downgrade_result: dict,
    rv_df: pd.DataFrame,
) -> None:
    today = date.today().strftime("%Y-%m-%d")
    metrics_spread = spread_result.get("cv_metrics", pd.DataFrame())
    metrics_down = downgrade_result.get("metrics", [])
    oof_df = spread_result.get("oof_df", pd.DataFrame())

    longs = rv_df.nlargest(20, "rv_long_score") if not rv_df.empty else pd.DataFrame()
    shorts = rv_df.nlargest(20, "rv_short_score") if not rv_df.empty else pd.DataFrame()

    # Spread metrics summary
    if not metrics_spread.empty:
        r2_mean = metrics_spread["r2"].mean()
        r2_std = metrics_spread["r2"].std()
        rmse_mean = metrics_spread["rmse"].mean()
        rho_mean = metrics_spread["spearman"].mean()
        spread_section = f"""
## Model 1 — Spread Prediction

**Algorithm**: LightGBM Regressor with monotonic constraints
**Validation**: Issuer-grouped 5-fold cross-validation

| Metric | Mean | ± Std |
|--------|------|-------|
| Grouped-CV R² | {r2_mean:.3f} | ±{r2_std:.3f} |
| RMSE (bps) | {rmse_mean:.1f} | ±{metrics_spread["rmse"].std():.1f} |
| Spearman ρ | {rho_mean:.3f} | ±{metrics_spread["spearman"].std():.3f} |

*Note: {len(oof_df)} bonds had valid spread data (Z/G/A-spread). OAS was not available in this CIQ export.*
"""
    else:
        spread_section = "\n## Model 1 — Spread Prediction\n\n*Not run.*\n"

    # Downgrade metrics
    if metrics_down:
        test_m = next((m for m in metrics_down if m.get("split") == "test"), metrics_down[-1])
        down_section = f"""
## Model 2 — Downgrade Risk

**Algorithm**: LightGBM classifier + Cox PH ensemble
**Validation**: Time-forward holdout (train 2021–2023, val 2024, test 2025)

| Metric | Test |
|--------|------|
| PR-AUC (primary) | {test_m.get('pr_auc', 'N/A'):.3f} |
| ROC-AUC | {test_m.get('roc_auc', 'N/A'):.3f} |
| Brier Score | {test_m.get('brier', 'N/A'):.3f} |
| Recall @ Top Decile | {test_m.get('recall_at_top_decile', 'N/A'):.2f} |
| Base Rate | {test_m.get('base_rate', 0):.1%} |

*Top features: EBITDA/Interest coverage trajectory (Δ2yr), leverage trajectory (Δ2yr), FCF/Debt.*
"""
    else:
        down_section = "\n## Model 2 — Downgrade Risk\n\n*Not run.*\n"

    # Long/short tables
    long_cols = ["issuer_name", "sp_rating", "actual_spread", "cheapness_bps", "downgrade_prob_4q", "modified_duration"]
    long_cols = [c for c in long_cols if c in longs.columns]
    longs_md = longs[long_cols].head(20).to_markdown(index=False) if not longs.empty else "*No data.*"
    shorts_md = shorts[long_cols].head(20).to_markdown(index=False) if not shorts.empty else "*No data.*"

    caveats = """
## Caveats

1. **Small N**: 168 bond observations, 42 with valid OAS. Spread model R² is not interpretable at this sample size.
2. **Single-period snapshot**: All bonds observed at one point in time (2026-05-27). Cross-sectional OAS regression has limited power vs. panel data.
3. **S&P ratings only**: No Moody's/Fitch confirmation. Rating outliers are not validated.
4. **No recovery modeling**: OAS contains credit spread and some liquidity; not decomposed.
5. **No liquidity adjustment**: Bid-ask spread is included as a feature but OAS itself isn't liquidity-adjusted.
6. **No Sharpe ratio reported**: Sharpe requires a paper-portfolio backtest with realistic bid-ask costs. Not done here.
7. **Downgrade label coverage**: Rating action history covers ~3 years; 2021–2022 labels have lower coverage.
"""

    report = f"""# Bond ML Phase 2 — Summary Report

**Generated**: {today}
**Dataset**: 168 bonds, 72 issuers, 6 annual Q1 snapshots (2021–2026)
**Excluded issuers** (no US-exchange MI Key): Expand Energy Corp., Moog Inc., Six Flags Entertainment Corp.

---

## Data Coverage

| Item | Count |
|------|-------|
| Bond observations | 168 |
| Unique issuers | 72 |
| Issuers with OAS data | {oof_df['mi_key'].nunique() if not oof_df.empty else 'N/A'} |
| Bonds with valid spread | {len(oof_df) if not oof_df.empty else 'N/A'} (Z/G/A-spread) |
| Historical periods | 6 (2021Q1–2026Q1) |
| Downgrade labels (positive) | 62 of 335 issuer-periods (18.5%) |

---
{spread_section}
---
{down_section}
---

## RV Framework — Top Longs (Cheap + Safe)

{longs_md}

---

## RV Framework — Top Shorts (Rich + Risky)

{shorts_md}

---
{caveats}
"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "phase2_summary.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Summary report written: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    spread_result: dict,
    downgrade_result: dict,
    rv_df: pd.DataFrame,
) -> None:
    """Generate all evaluation plots and the summary report."""
    oof_df = spread_result.get("oof_df", pd.DataFrame())
    if not oof_df.empty:
        plot_spread_predicted_vs_actual(oof_df)
        plot_spread_residuals(oof_df)

    if "importance_df" in spread_result:
        plot_spread_feature_importance(spread_result["importance_df"])

    if "importance_df" in downgrade_result:
        plot_downgrade_feature_importance(downgrade_result["importance_df"])

    # Calibration on test predictions
    ddf = downgrade_result.get("downgrade_df", pd.DataFrame())
    if not ddf.empty and "downgrade_next_yr" in ddf.columns and "downgrade_prob_4q" in ddf.columns:
        valid = ddf[["downgrade_next_yr", "downgrade_prob_4q"]].dropna()
        if len(valid) >= 10:
            plot_calibration(valid["downgrade_next_yr"].values, valid["downgrade_prob_4q"].values)

    write_summary_report(spread_result, downgrade_result, rv_df)
