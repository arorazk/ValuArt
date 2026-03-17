"""
data_loader.py

Handles loading the dataset. If the CSV doesn't exist yet it generates
a synthetic one and saves it to data/nft_sales.csv so you only pay the
generation cost once.

I couldn't find a clean pre-labelled NFT sales dataset with all the
features I wanted, so I built a synthetic generator that produces
realistic-looking data (lognormal prices, proper blockchain splits, etc.)
The price formula is deliberately learnable so models can actually do
something useful with it.
"""

import os
import numpy as np
import pandas as pd
from src.utils import get_logger, ensure_dir, project_root

logger = get_logger(__name__)

RNG_SEED = 42

# real collection/creator names make the dataset feel less synthetic
COLLECTIONS = [
    "CryptoPunks", "Bored Ape YC", "Azuki", "Doodles", "CloneX",
    "Moonbirds", "Pudgy Penguins", "Cool Cats", "World of Women", "VeeFriends",
    "Chromie Squiggle", "Art Blocks", "Meebits", "Mutant Ape YC", "CyberKongz",
    "Sandbox Land", "Decentraland", "Otherdeed", "Invisible Friends", "RTFKT",
]

CREATORS = [
    "LarvaLabs", "YugaLabs", "Azuki_Team", "Evan_Conceicao", "RTFKTstudios",
    "KevinRose_Team", "CoolCats_Dev", "WoW_NFT", "GaryVee", "SnowFro",
    "Meebits_Dev", "Sandbox_AG", "Decentraland_Fdn", "Invisible_Studio", "CyberKongz_Dev",
]

BLOCKCHAINS = ["ethereum", "polygon", "solana"]


def _simulate_sale_price(row: pd.Series, rng: np.random.Generator) -> float:
    """
    Builds the sale price from a weighted mix of features plus some noise.
    Weights are rough guesses based on what actually drives NFT prices —
    historical avg and floor price dominate, social stuff matters less.
    ETH commands a premium over SOL and MATIC which felt realistic.
    """
    base = (
        0.40 * row["past_avg_price"]
        + 0.25 * row["floor_price"]
        + 0.10 * row["rarity_score"] / 10.0
        + 0.08 * np.log1p(row["sales_last_30d"])
        + 0.05 * np.log1p(row["twitter_followers"])
        + 0.04 * np.log1p(row["discord_members"])
        + 0.03 * row["engagement_rate"] * 10
        - 0.02 * np.log1p(row["days_since_mint"])
        - 0.01 * row["listing_count"] / 100.0
    )
    # ETH listings trade higher than SOL or MATIC in practice
    chain_mult = {"ethereum": 1.0, "solana": 0.55, "polygon": 0.30}
    base *= chain_mult.get(row["blockchain"], 1.0)
    # add ±20% noise so it's not perfectly deterministic
    noise = rng.normal(1.0, 0.20)
    return max(round(base * noise, 4), 0.01)


def generate_dataset(n_samples: int = 3000, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Generates the synthetic NFT sales dataset.

    Tried to keep the distributions realistic:
    - floor prices are lognormal (most things cheap, few very expensive)
    - past_avg_price is always >= floor_price
    - ~65% of listings are on ETH, rest split between SOL and MATIC
    - about 3% of some feature columns are missing (happens in real scraped data)
    """
    rng = np.random.default_rng(seed)

    blockchain = rng.choice(BLOCKCHAINS, size=n_samples, p=[0.65, 0.20, 0.15])

    floor_price      = np.round(rng.lognormal(mean=1.5, sigma=1.2, size=n_samples), 4)
    past_avg_price   = np.round(floor_price * rng.uniform(1.0, 3.5, size=n_samples), 4)
    rarity_score     = np.round(rng.uniform(1, 500, size=n_samples), 2)
    trait_count      = rng.integers(1, 15, size=n_samples)
    sales_last_7d    = rng.integers(0, 200, size=n_samples)
    sales_last_30d   = sales_last_7d * rng.integers(2, 6, size=n_samples)
    twitter_followers = rng.integers(500, 2_000_000, size=n_samples)
    discord_members  = rng.integers(100, 500_000, size=n_samples)
    engagement_rate  = np.round(rng.uniform(0.001, 0.15, size=n_samples), 4)
    listing_count    = rng.integers(10, 5000, size=n_samples)
    days_since_mint  = rng.integers(1, 1200, size=n_samples)
    collection_name  = rng.choice(COLLECTIONS, size=n_samples)
    creator          = rng.choice(CREATORS, size=n_samples)

    df = pd.DataFrame({
        "collection_name":   collection_name,
        "creator":           creator,
        "rarity_score":      rarity_score,
        "trait_count":       trait_count,
        "past_avg_price":    past_avg_price,
        "floor_price":       floor_price,
        "sales_last_7d":     sales_last_7d,
        "sales_last_30d":    sales_last_30d,
        "twitter_followers": twitter_followers,
        "discord_members":   discord_members,
        "engagement_rate":   engagement_rate,
        "listing_count":     listing_count,
        "days_since_mint":   days_since_mint,
        "blockchain":        blockchain,
    })

    # compute sale_price before injecting nulls so the target is always clean
    df["sale_price"] = df.apply(lambda r: _simulate_sale_price(r, rng), axis=1)

    # inject a small percentage of nulls into a few columns to make cleaning step realistic
    for col in ["rarity_score", "engagement_rate", "discord_members", "trait_count"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    logger.info("Generated synthetic dataset: %d rows × %d cols", *df.shape)
    return df


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Loads data from csv_path. If the file doesn't exist it generates
    a fresh dataset and saves it so subsequent runs are faster.
    """
    if csv_path is None:
        csv_path = os.path.join(project_root(), "data", "nft_sales.csv")

    if os.path.exists(csv_path):
        logger.info("Loading existing dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        logger.info("Dataset not found – generating synthetic data …")
        df = generate_dataset()
        ensure_dir(os.path.dirname(csv_path))
        df.to_csv(csv_path, index=False)
        logger.info("Saved to %s", csv_path)

    return df
