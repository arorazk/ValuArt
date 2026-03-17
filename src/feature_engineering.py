"""
feature_engineering.py

Creates derived features on top of the raw columns.

Most of these came from thinking about what actually drives NFT prices:
- How does this NFT's historical avg compare to the floor? (price_to_floor_ratio)
- Are people buying or just listing? (supply_demand_ratio, sales_velocity)
- How big is the community and how active? (social_score, engagement_weighted)
- Log transforms on skewed columns (floor price, social counts) help tree models
  split more effectively on the lower end of the distribution
- Age bucket captures that newer collections sometimes trade at a premium

Nothing fancy here, just trying to give the models useful signals.
"""

import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered columns to the dataframe and returns a copy.
    Call this after loading raw data, before preprocessing.
    """
    df = df.copy()

    # how much has this collection historically traded above floor?
    df["price_to_floor_ratio"] = df["past_avg_price"] / (df["floor_price"] + 1e-6)

    # high ratio = lots of listings relative to sales = less demand
    df["supply_demand_ratio"] = df["listing_count"] / (df["sales_last_30d"] + 1)

    # what share of monthly volume happened in the last week
    df["sales_velocity"] = df["sales_last_7d"] / (df["sales_last_30d"] + 1)

    # log-scale social counts — raw follower numbers are way too skewed
    df["log_twitter"]  = np.log1p(df["twitter_followers"])
    df["log_discord"]  = np.log1p(df["discord_members"])
    df["social_score"] = df["log_twitter"] + df["log_discord"]

    # bigger community + higher engagement = stronger signal
    df["engagement_weighted"] = df["social_score"] * df["engagement_rate"]

    # log transform the price columns too
    df["log_floor_price"]    = np.log1p(df["floor_price"])
    df["log_past_avg_price"] = np.log1p(df["past_avg_price"])

    # bin collection age into rough stages (new / growing / mature / old)
    bins   = [0, 30, 90, 180, 365, 730, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    df["age_bucket"] = pd.cut(
        df["days_since_mint"], bins=bins, labels=labels, right=True
    ).astype(float)

    new_cols = [
        "price_to_floor_ratio", "supply_demand_ratio", "sales_velocity",
        "log_twitter", "log_discord", "social_score", "engagement_weighted",
        "log_floor_price", "log_past_avg_price", "age_bucket",
    ]
    logger.info("Engineered %d new features: %s", len(new_cols), new_cols)
    return df


def get_feature_columns(df: pd.DataFrame, target: str = "sale_price") -> list[str]:
    """Returns all numeric columns except the target — handy for quick feature lists."""
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
