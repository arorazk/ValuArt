"""
train.py – Train multiple regression models with cross-validation.

Models evaluated:
  1. Linear Regression   (baseline, interpretable)
  2. Random Forest       (handles non-linearity, feature importance)
  3. Gradient Boosting   (typically best performer on tabular data)

Each model is assessed via 5-fold cross-validation (R², MAE, RMSE) before
being retrained on the full training set.  The best model by CV R² is
serialised to outputs/models/best_model.pkl with joblib.
"""

import os
import time

import copy

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils import get_logger, ensure_dir, outputs_path, rmse

logger = get_logger(__name__)

RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Model registry ─────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return a dictionary of {name: unfitted estimator}."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


# ── Training pipeline ───────────────────────────────────────────────────────

def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Run 5-fold CV on every model using a manual fold loop.

    Returns
    -------
    pd.DataFrame with columns [model, cv_r2, cv_r2_std, cv_mae, cv_rmse, fit_time_s]
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)
    rows = []

    for name, base_model in models.items():
        logger.info("Cross-validating %s …", name)
        t0 = time.time()

        fold_r2, fold_mae, fold_rmse = [], [], []
        for train_idx, val_idx in kf.split(X_arr):
            m = copy.deepcopy(base_model)
            m.fit(X_arr[train_idx], y_arr[train_idx])
            y_pred = m.predict(X_arr[val_idx])
            fold_r2.append(r2_score(y_arr[val_idx], y_pred))
            fold_mae.append(mean_absolute_error(y_arr[val_idx], y_pred))
            fold_rmse.append(rmse(y_arr[val_idx], y_pred))

        elapsed = round(time.time() - t0, 2)
        rows.append({
            "model":       name,
            "cv_r2":       round(float(np.mean(fold_r2)), 4),
            "cv_r2_std":   round(float(np.std(fold_r2)), 4),
            "cv_mae":      round(float(np.mean(fold_mae)), 4),
            "cv_rmse":     round(float(np.mean(fold_rmse)), 4),
            "fit_time_s":  elapsed,
        })
        logger.info(
            "  %s → R²=%.4f (±%.4f)  MAE=%.4f  RMSE=%.4f  [%.1fs]",
            name,
            rows[-1]["cv_r2"], rows[-1]["cv_r2_std"],
            rows[-1]["cv_mae"], rows[-1]["cv_rmse"],
            elapsed,
        )

    return pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)


def train_final_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Refit every model on the full training set.

    Returns
    -------
    dict of {name: fitted_model}
    """
    models = build_models()
    fitted = {}
    for name, model in models.items():
        logger.info("Training final %s on full training set …", name)
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def save_best_model(
    fitted_models: dict,
    cv_results: pd.DataFrame,
    path: str | None = None,
) -> tuple[str, object]:
    """
    Persist the best model (highest CV R²) to disk with joblib.

    Parameters
    ----------
    fitted_models : dict
        Fitted estimators from train_final_models().
    cv_results : pd.DataFrame
        Output of cross_validate_models().
    path : str or None
        Save path. Defaults to outputs/models/best_model.pkl.

    Returns
    -------
    (best_model_name, fitted_model)
    """
    best_name = cv_results.iloc[0]["model"]
    best_model = fitted_models[best_name]

    if path is None:
        path = outputs_path("models", "best_model.pkl")

    ensure_dir(os.path.dirname(path))
    joblib.dump(best_model, path)
    logger.info("Saved best model ('%s') → %s", best_name, path)
    return best_name, best_model
