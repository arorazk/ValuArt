"""
main.py – End-to-end pipeline for ValuArt: NFT Pricing Model.

Run:
    python main.py

Outputs produced in outputs/:
    models/best_model.pkl
    reports/model_comparison.csv
    plots/feature_importance.png
    plots/actual_vs_predicted.png
    plots/residuals.png
"""

import sys
import time

import joblib

from src.data_loader import load_data
from src.preprocess import NFTPreprocessor, split_data
from src.feature_engineering import engineer_features
from src.train import cross_validate_models, train_final_models, save_best_model
from src.evaluate import run_evaluation
from src.utils import get_logger, ensure_dir, outputs_path

logger = get_logger("valuart.main")


def run_pipeline() -> None:
    logger.info("=" * 60)
    logger.info("  ValuArt – NFT Pricing Model  |  Pipeline Start")
    logger.info("=" * 60)

    t_start = time.time()

    # ── Step 1: Load / generate data ───────────────────────────────────────
    logger.info("[1/6] Loading data …")
    raw_df = load_data()
    logger.info("      Dataset shape: %s", raw_df.shape)

    # ── Step 2: Feature engineering (on raw df, before encoding) ──────────
    logger.info("[2/6] Engineering features …")
    df_eng = engineer_features(raw_df)

    # ── Step 3: Train / test split ─────────────────────────────────────────
    logger.info("[3/6] Splitting into train / test sets …")
    train_df, test_df = split_data(df_eng, test_size=0.20, random_state=42)

    # ── Step 4: Preprocessing (fit on train, transform both) ───────────────
    logger.info("[4/6] Preprocessing …")
    preprocessor = NFTPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test,  y_test  = preprocessor.transform(test_df)
    logger.info("      X_train: %s  |  X_test: %s", X_train.shape, X_test.shape)

    # ── Step 5: Cross-validate + train final models ────────────────────────
    logger.info("[5/6] Cross-validating models …")
    cv_results = cross_validate_models(X_train, y_train)

    logger.info("      Training final models on full training set …")
    fitted_models = train_final_models(X_train, y_train)

    best_name, _ = save_best_model(fitted_models, cv_results)
    logger.info("      Best model by CV R²: %s", best_name)

    # save the fitted preprocessor so the Streamlit app can reuse it
    prep_path = outputs_path("models", "preprocessor.pkl")
    ensure_dir(outputs_path("models"))
    joblib.dump(preprocessor, prep_path)
    logger.info("      Saved preprocessor → %s", prep_path)

    # ── Step 6: Evaluate on hold-out + produce outputs ─────────────────────
    logger.info("[6/6] Evaluating on test set + generating plots …")
    results = run_evaluation(
        fitted_models, cv_results, X_test, y_test, best_name
    )

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 60)

    print("\n── Model Comparison ──────────────────────────────────────")
    print(results.to_string(index=False))
    print("\nOutputs saved in outputs/")


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)
