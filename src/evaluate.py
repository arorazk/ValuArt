"""
evaluate.py – Compute hold-out metrics and produce diagnostic plots.

Outputs
-------
  outputs/reports/model_comparison.csv   – CV + test metrics for all models
  outputs/plots/feature_importance.png   – top-20 feature importances
  outputs/plots/actual_vs_predicted.png  – scatter of best model predictions
  outputs/plots/residuals.png            – residual distribution + fitted plot
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils import get_logger, ensure_dir, outputs_path, rmse, save_fig

logger = get_logger(__name__)

PLOT_DIR   = outputs_path("plots")
REPORT_DIR = outputs_path("reports")


# ── Hold-out metrics ────────────────────────────────────────────────────────

def evaluate_on_test(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate all fitted models on the hold-out test set and merge with CV results.

    Returns
    -------
    pd.DataFrame with both CV and test-set metrics.
    """
    rows = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "model":    name,
            "test_r2":  round(r2_score(y_test, y_pred), 4),
            "test_mae": round(mean_absolute_error(y_test, y_pred), 4),
            "test_rmse": round(rmse(y_test, y_pred), 4),
        })
        logger.info(
            "Test  %s → R²=%.4f  MAE=%.4f  RMSE=%.4f",
            name, rows[-1]["test_r2"], rows[-1]["test_mae"], rows[-1]["test_rmse"],
        )

    test_df = pd.DataFrame(rows)
    merged  = cv_results.merge(test_df, on="model")
    return merged


def save_model_comparison(results: pd.DataFrame, path: str | None = None) -> str:
    """Write model comparison table to CSV."""
    if path is None:
        path = os.path.join(REPORT_DIR, "model_comparison.csv")
    ensure_dir(os.path.dirname(path))
    results.to_csv(path, index=False)
    logger.info("Model comparison saved → %s", path)
    return path


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "Model",
    top_n: int = 20,
) -> None:
    """
    Bar chart of the top-N feature importances.
    Works with tree-based models that expose feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("%s does not expose feature_importances_ – skipping plot", model_name)
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))  # type: ignore[attr-defined]
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Gini / Gain)", fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "feature_importance.png")
    save_fig(fig, path)
    logger.info("Feature importance plot saved → %s", path)


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
) -> None:
    """Scatter plot of actual vs. predicted sale prices."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#2196F3", edgecolors="none")

    lims = [
        min(y_true.min(), y_pred.min()) * 0.95,
        max(y_true.max(), y_pred.max()) * 1.05,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    r2 = r2_score(y_true, y_pred)
    ax.set_title(
        f"Actual vs. Predicted Sale Price – {model_name}\n$R^2$ = {r2:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Actual Sale Price (ETH)", fontsize=11)
    ax.set_ylabel("Predicted Sale Price (ETH)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "actual_vs_predicted.png")
    save_fig(fig, path)
    logger.info("Actual vs. predicted plot saved → %s", path)


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
) -> None:
    """Two-panel residual diagnostic: residuals vs fitted + histogram."""
    residuals = np.asarray(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 – Residuals vs Fitted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.3, s=16, color="#FF7043", edgecolors="none")
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_title(f"Residuals vs Fitted – {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fitted (Predicted) Values", fontsize=11)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=11)
    ax.grid(alpha=0.3)

    # Panel 2 – Residual distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, color="#7E57C2", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Residual Analysis – {model_name}", fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "residuals.png")
    save_fig(fig, path)
    logger.info("Residual plot saved → %s", path)


# ── Full evaluation run ─────────────────────────────────────────────────────

def run_evaluation(
    fitted_models: dict,
    cv_results: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
) -> pd.DataFrame:
    """
    Orchestrate full evaluation: metrics + all diagnostic plots.

    Returns
    -------
    pd.DataFrame  –  merged CV + test metrics
    """
    ensure_dir(PLOT_DIR)
    ensure_dir(REPORT_DIR)

    results = evaluate_on_test(fitted_models, X_test, y_test, cv_results)
    save_model_comparison(results)

    best_model = fitted_models[best_model_name]
    y_pred     = best_model.predict(X_test)
    feat_names = list(X_test.columns)

    plot_feature_importance(best_model, feat_names, model_name=best_model_name)
    plot_actual_vs_predicted(y_test, y_pred, model_name=best_model_name)
    plot_residuals(y_test, y_pred, model_name=best_model_name)

    return results
