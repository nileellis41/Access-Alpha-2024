"""
RV Framework: cheapness × deterioration 2×2.

Combines Model 1 (spread prediction) and Model 2 (downgrade risk) into
a relative-value scatter for current bond observations.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURES_DOWNGRADES, FEATURES_SPREADS, FIGURES, REPORTS


def build_rv_scores(
    spread_result: dict,
    downgrade_result: dict,
) -> pd.DataFrame:
    """
    Combine OOF spread residuals with current-snapshot downgrade probabilities.

    Returns DataFrame with one row per bond with valid OAS:
      cheapness_bps     = actual_oas − predicted_oas  (+ = market sees more risk)
      downgrade_prob_4q = ensemble downgrade probability from Model 2
    """
    oof_df = spread_result["oof_df"].copy()

    # Current-period downgrade probs (2026Q1 or latest)
    downgrade_df = downgrade_result.get("downgrade_df", pd.DataFrame())

    if downgrade_df.empty:
        oof_df["downgrade_prob_4q"] = np.nan
    else:
        # Take the latest period per issuer for the current-state prob
        latest_probs = (
            downgrade_df.sort_values("period")
            .groupby("mi_key")[["mi_key", "downgrade_prob_4q"]]
            .last()
            .reset_index(drop=True)
        )
        oof_df["mi_key"] = oof_df["mi_key"].astype(str)
        latest_probs["mi_key"] = latest_probs["mi_key"].astype(str)
        oof_df = oof_df.merge(latest_probs, on="mi_key", how="left")

    # Ensure columns exist
    if "cheapness_bps" not in oof_df.columns:
        actual_col = next((c for c in ["spread_target", "z_spread_bid", "g_spread_bid", "oas_bid"] if c in oof_df.columns), None)
        pred_col = next((c for c in ["predicted_spread", "predicted_oas"] if c in oof_df.columns), None)
        if actual_col and pred_col:
            oof_df["cheapness_bps"] = oof_df[actual_col] - oof_df[pred_col]
    if "downgrade_prob_4q" not in oof_df.columns:
        oof_df["downgrade_prob_4q"] = np.nan

    # Use whichever actual-spread column exists
    actual_col = next((c for c in ["spread_target", "z_spread_bid", "g_spread_bid", "oas_bid"] if c in oof_df.columns), "spread_target")
    pred_col = next((c for c in ["predicted_spread", "predicted_oas"] if c in oof_df.columns), "predicted_spread")

    keep_cols = ["issuer_name", "cusip", "instrument_id", "mi_key",
                 actual_col, pred_col, "cheapness_bps",
                 "downgrade_prob_4q", "sp_rating", "rating_numeric", "modified_duration"]
    rv = oof_df[[c for c in keep_cols if c in oof_df.columns]].copy()
    rv = rv.rename(columns={actual_col: "actual_spread", pred_col: "predicted_spread"})

    # Quadrant labels
    cheap_thresh = rv["cheapness_bps"].median()
    risk_thresh = rv["downgrade_prob_4q"].median() if rv["downgrade_prob_4q"].notna().any() else 0.5

    def quadrant(row):
        cheap = row["cheapness_bps"] > cheap_thresh
        risky = (row["downgrade_prob_4q"] > risk_thresh) if not np.isnan(row.get("downgrade_prob_4q", np.nan)) else False
        if cheap and risky:
            return "Trap (cheap+risky)"
        elif not cheap and risky:
            return "Short (rich+risky)"
        elif cheap and not risky:
            return "Long (cheap+safe)"
        else:
            return "Hold (rich+safe)"

    rv["quadrant"] = rv.apply(quadrant, axis=1)

    # RV long score: cheap AND safe → high score
    rv["rv_long_score"] = rv["cheapness_bps"] * (1 - rv["downgrade_prob_4q"].fillna(0.5))
    rv["rv_short_score"] = -rv["cheapness_bps"] * rv["downgrade_prob_4q"].fillna(0.5)

    return rv.sort_values("rv_long_score", ascending=False).reset_index(drop=True)


def plot_rv_2x2(rv_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate the 2×2 cheapness × deterioration scatter.
    Annotates quadrant labels; colors by IG/HY.
    """
    df = rv_df.dropna(subset=["cheapness_bps"]).copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by IG/HY
    is_ig = df["rating_numeric"].le(10) if "rating_numeric" in df.columns else pd.Series(True, index=df.index)
    colors = np.where(is_ig, "#2196F3", "#FF5722")

    x = df["cheapness_bps"]
    y = df["downgrade_prob_4q"].fillna(df["downgrade_prob_4q"].median())

    ax.scatter(x, y, c=colors, alpha=0.7, s=60, edgecolors="white", linewidths=0.5)

    # Quadrant lines at medians
    x_med = x.median()
    y_med = y.median() if not y.isna().all() else 0.5
    ax.axvline(x_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Quadrant labels
    x_range = x.max() - x.min()
    y_range = y.max() - y.min() if y.max() != y.min() else 1
    ax.text(x.min() + 0.02 * x_range, y.max() - 0.04 * y_range,
            "Rich + Risky\n(Short candidates)", ha="left", va="top",
            fontsize=9, color="#FF5722", alpha=0.7)
    ax.text(x.max() - 0.02 * x_range, y.max() - 0.04 * y_range,
            "Cheap + Risky\n(Trap — avoid)", ha="right", va="top",
            fontsize=9, color="#FF5722", alpha=0.7)
    ax.text(x.min() + 0.02 * x_range, y.min() + 0.04 * y_range,
            "Rich + Safe\n(Hold)", ha="left", va="bottom",
            fontsize=9, color="#2196F3", alpha=0.7)
    ax.text(x.max() - 0.02 * x_range, y.min() + 0.04 * y_range,
            "Cheap + Safe\n(Long candidates)", ha="right", va="bottom",
            fontsize=9, color="#2196F3", alpha=0.7)

    # Annotate top/bottom bonds
    top_n = min(5, len(df))
    for _, row in df.nlargest(top_n, "rv_long_score").iterrows():
        label = f"{row['issuer_name'].split()[0]}\n{row.get('sp_rating','')}"
        ax.annotate(label, (row["cheapness_bps"], y.loc[row.name] if row.name in y.index else y_med),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", alpha=0.8)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="Investment Grade"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=8, label="High Yield"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_xlabel("Cheapness (Actual OAS − Predicted OAS, bps)\n+ = market prices in more risk than model", fontsize=10)
    ax.set_ylabel("4Q Downgrade Probability (ensemble)", fontsize=10)
    ax.set_title(
        f"Bond RV Framework: Cheapness × Deterioration\n"
        f"(as of {date.today()}, n={len(df)} bonds with valid OAS)",
        fontsize=12,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"RV 2x2 saved: {output_path}")


def run_rv_framework(
    spread_result: dict,
    downgrade_result: dict,
) -> pd.DataFrame:
    """
    Generate RV scores, 2×2 plot, and CSVs.
    Returns rv_df with quadrant assignments.
    """
    rv_df = build_rv_scores(spread_result, downgrade_result)

    today_str = date.today().strftime("%Y%m%d")
    csv_path = REPORTS / f"rv_scores_{today_str}.csv"
    plot_path = FIGURES / f"rv_2x2_{today_str}.png"

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rv_df.to_csv(csv_path, index=False)
    print(f"RV scores saved: {csv_path}")

    plot_rv_2x2(rv_df, plot_path)

    # Print top longs and shorts
    longs = rv_df.nlargest(min(20, len(rv_df)), "rv_long_score")
    shorts = rv_df.nlargest(min(20, len(rv_df)), "rv_short_score")

    print_cols = [c for c in ["issuer_name", "sp_rating", "actual_spread", "cheapness_bps",
                               "downgrade_prob_4q", "modified_duration"] if c in rv_df.columns]

    print("\n=== Top Long Candidates (cheap + safe) ===")
    print(longs[print_cols].head(10).to_string(index=False))

    print("\n=== Top Short Candidates (rich + risky) ===")
    print(shorts[print_cols].head(10).to_string(index=False))

    return rv_df
