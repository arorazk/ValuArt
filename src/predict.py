"""
predict.py – Load a saved model and score new NFT listings.

Usage (CLI)
-----------
    python -m src.predict --input data/new_listings.csv --output outputs/predictions.csv

Usage (API)
-----------
    from src.predict import load_model, predict_from_df

    model = load_model()
    preds = predict_from_df(model, preprocessor, new_df)
"""

import argparse
import os

import joblib
import pandas as pd

from src.preprocess import NFTPreprocessor
from src.feature_engineering import engineer_features
from src.utils import get_logger, outputs_path

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = outputs_path("models", "best_model.pkl")


def load_model(path: str = DEFAULT_MODEL_PATH):
    """
    Deserialise a model saved by joblib.

    Parameters
    ----------
    path : str
        Path to the .pkl file.

    Returns
    -------
    Fitted scikit-learn estimator.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run main.py first to train and save the model."
        )
    model = joblib.load(path)
    logger.info("Loaded model from %s  (%s)", path, type(model).__name__)
    return model


def predict_from_df(
    model,
    preprocessor: NFTPreprocessor,
    df: pd.DataFrame,
) -> pd.Series:
    """
    Run the full feature pipeline on df and return price predictions.

    Parameters
    ----------
    model : fitted estimator
    preprocessor : NFTPreprocessor fitted on the training set
    df : pd.DataFrame
        Raw NFT listings; must contain the same columns as the training data
        (sale_price column is optional – set to 0 if absent).

    Returns
    -------
    pd.Series of predicted sale prices (ETH).
    """
    if "sale_price" not in df.columns:
        df = df.copy()
        df["sale_price"] = 0.0   # dummy target so preprocessor doesn't fail

    df_eng   = engineer_features(df)
    X, _     = preprocessor.transform(df_eng)
    y_pred   = model.predict(X)
    return pd.Series(y_pred, name="predicted_sale_price", index=df.index)


# ── CLI entry-point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score new NFT listings with the saved ValuArt model.")
    p.add_argument("--input",  required=True,  help="Path to input CSV")
    p.add_argument("--output", required=True,  help="Path to save predictions CSV")
    p.add_argument("--model",  default=DEFAULT_MODEL_PATH, help="Path to model .pkl file")
    return p.parse_args()


def main() -> None:
    args  = _parse_args()
    model = load_model(args.model)
    df    = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # We need a fitted preprocessor – load from the training artefact if saved,
    # or ask the user to re-run main.py.  For simplicity, we re-fit here on the
    # provided data (acceptable for small demo purposes).
    from src.feature_engineering import engineer_features
    df_eng = engineer_features(df)

    prep = NFTPreprocessor()
    X, _ = prep.fit_transform(df_eng)   # fit on the new data itself (demo only)
    y_pred = model.predict(X)

    out_df = df.copy()
    out_df["predicted_sale_price"] = y_pred
    out_df.to_csv(args.output, index=False)
    logger.info("Predictions saved → %s", args.output)


if __name__ == "__main__":
    main()
