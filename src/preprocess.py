"""
preprocess.py

Handles all the cleaning and encoding before features go into a model.

Main steps:
  1. Drop any rows where sale_price is missing or zero (can't train on those)
  2. Fill missing numeric values with the column median
  3. Ordinal-encode blockchain (only 3 values, so this is fine)
  4. Target-encode collection_name and creator — way too many unique values
     for one-hot encoding, so I use mean sale_price per category instead.
     Unknown categories at inference time fall back to the global mean.

Note: the preprocessor is stateful — fit it on training data only,
then call transform() on the test set. Fitting on the full dataset
would be a data leakage bug.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from src.utils import get_logger

logger = get_logger(__name__)

DROP_COLS = []  # nothing to drop for now, keeping this here for future use

NUMERIC_COLS = [
    "rarity_score", "trait_count", "past_avg_price", "floor_price",
    "sales_last_7d", "sales_last_30d", "twitter_followers", "discord_members",
    "engagement_rate", "listing_count", "days_since_mint",
]

LOW_CARD_CATS  = ["blockchain"]               # ordinal encoded
HIGH_CARD_CATS = ["collection_name", "creator"]  # target encoded

TARGET = "sale_price"


class NFTPreprocessor:
    """
    Fit on training data, then reuse to transform test data.

    Example
    -------
    >>> prep = NFTPreprocessor()
    >>> X_train, y_train = prep.fit_transform(train_df)
    >>> X_test, y_test   = prep.transform(test_df)
    """

    def __init__(self):
        self._ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._target_means: dict[str, pd.Series] = {}
        self._global_mean: float = 0.0
        self._numeric_medians: pd.Series | None = None

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on training data and return (X, y)."""
        df = self._basic_clean(df)
        self._fit_imputer(df)
        self._fit_ordinal(df)
        self._fit_target_enc(df)
        return self._apply(df)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform using parameters already fitted on training data."""
        df = self._basic_clean(df)
        return self._apply(df)

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before = len(df)
        df = df[df[TARGET].notna() & (df[TARGET] > 0)]
        removed = before - len(df)
        if removed:
            logger.warning("Dropped %d rows with missing/non-positive sale_price", removed)
        if DROP_COLS:
            df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
        return df

    def _fit_imputer(self, df: pd.DataFrame) -> None:
        # median is more robust than mean when prices are skewed
        self._numeric_medians = df[NUMERIC_COLS].median()
        logger.info("Fitted numeric imputer (median strategy)")

    def _fit_ordinal(self, df: pd.DataFrame) -> None:
        self._ordinal_enc.fit(df[LOW_CARD_CATS])
        logger.info("Fitted ordinal encoder on: %s", LOW_CARD_CATS)

    def _fit_target_enc(self, df: pd.DataFrame) -> None:
        self._global_mean = df[TARGET].mean()
        for col in HIGH_CARD_CATS:
            self._target_means[col] = df.groupby(col)[TARGET].mean()
        logger.info("Fitted target encoder on: %s", HIGH_CARD_CATS)

    def _apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        y = df.pop(TARGET)

        df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(self._numeric_medians)
        df[LOW_CARD_CATS] = self._ordinal_enc.transform(df[LOW_CARD_CATS])

        # unknown collections/creators get the global mean — not perfect but good enough
        for col in HIGH_CARD_CATS:
            df[col] = df[col].map(self._target_means[col]).fillna(self._global_mean)

        X = df[[*NUMERIC_COLS, *LOW_CARD_CATS, *HIGH_CARD_CATS]].astype(float)
        return X, y


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple reproducible train/test split.
    Keeping this manual (no sklearn) so it's easy to understand what's happening.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(df))
    split = int(len(df) * (1 - test_size))
    train_df = df.iloc[idx[:split]].reset_index(drop=True)
    test_df  = df.iloc[idx[split:]].reset_index(drop=True)
    logger.info(
        "Train/test split: %d train rows, %d test rows (%.0f%% / %.0f%%)",
        len(train_df), len(test_df),
        (1 - test_size) * 100, test_size * 100,
    )
    return train_df, test_df
