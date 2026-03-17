# ValuArt – NFT Pricing Model

> A reproducible machine learning pipeline that estimates NFT sale prices from collection stats, rarity signals, and social engagement data.

---

## Problem Statement

NFT prices are notoriously volatile and opaque. Buyers and sellers lack objective pricing signals beyond floor price. This project builds a regression pipeline that learns from historical sales and social features to produce a **data-driven price estimate** for any given NFT listing.

The goal is **not** to replace market sentiment, but to surface a quantitative baseline useful for:
- Buyers assessing whether a listing is overpriced
- Sellers setting competitive ask prices
- Researchers studying NFT market structure

---

## Project Structure

```
ValuArt/
├── data/
│   └── nft_sales.csv           ← auto-generated on first run
├── notebooks/
│   └── valuart_demo.ipynb      ← full walkthrough notebook
├── src/
│   ├── data_loader.py          ← load / generate the dataset
│   ├── preprocess.py           ← clean, impute, encode
│   ├── feature_engineering.py  ← create derived features
│   ├── train.py                ← CV + final model training
│   ├── evaluate.py             ← metrics + diagnostic plots
│   ├── predict.py              ← score new listings
│   └── utils.py                ← shared helpers (logging, paths, plots)
├── outputs/
│   ├── models/
│   │   └── best_model.pkl
│   ├── plots/
│   │   ├── feature_importance.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals.png
│   │   └── cv_r2_comparison.png
│   └── reports/
│       └── model_comparison.csv
├── main.py                     ← end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Dataset

Because a clean, labelled public NFT sales dataset with all required features is not freely available in a static form, this project **generates a realistic synthetic dataset** (3 000 rows) on first run.  The generator lives in `src/data_loader.py` and produces a CSV at `data/nft_sales.csv`.

### Feature Descriptions

| Column | Type | Description |
|---|---|---|
| `collection_name` | categorical | NFT collection (e.g. Bored Ape YC, Azuki) |
| `creator` | categorical | Deployer / studio name |
| `rarity_score` | float | Composite rarity metric (1–500) |
| `trait_count` | int | Number of distinct traits on the token |
| `past_avg_price` | float | Collection's historical average sale price (ETH) |
| `floor_price` | float | Current collection floor price (ETH) |
| `sales_last_7d` | int | Number of sales in the past 7 days |
| `sales_last_30d` | int | Number of sales in the past 30 days |
| `twitter_followers` | int | Creator / collection Twitter followers |
| `discord_members` | int | Discord server member count |
| `engagement_rate` | float | Social engagement rate (likes+comments / followers) |
| `listing_count` | int | Active listings on secondary markets |
| `days_since_mint` | int | Age of the collection in days |
| `blockchain` | categorical | `ethereum`, `polygon`, or `solana` |
| **`sale_price`** | **float** | **Target – final sale price in ETH** |

~3% of `rarity_score`, `engagement_rate`, `discord_members`, and `trait_count` values are deliberately missing to make the cleaning step realistic.

---

## ML Pipeline

```
Raw CSV
  │
  ▼
Feature Engineering      (ratio features, log transforms, age buckets)
  │
  ▼
Train / Test Split        (80 / 20, seeded)
  │
  ▼
Preprocessing             (median imputation, ordinal + target encoding)
  │
  ▼
5-Fold Cross-Validation   (R², MAE, RMSE per fold)
  │
  ▼
Final Model Training      (refit on full training set)
  │
  ▼
Hold-Out Evaluation       (test-set R², MAE, RMSE)
  │
  ▼
Outputs                   (model .pkl, comparison CSV, plots)
```

---

## Models

| Model | Why included |
|---|---|
| **Linear Regression** | Interpretable baseline; reveals linear relationships |
| **Random Forest** | Handles non-linearity; robust to outliers; provides feature importances |
| **Gradient Boosting** | Typically strongest tabular-data performer; iterative error correction |

All models share `random_state=42` for reproducibility. Hyperparameters are set to sensible defaults rather than exhaustively tuned—a deliberate choice to keep the project scope realistic.

---

## Engineered Features

Beyond the raw columns, the pipeline adds:

| Feature | Formula |
|---|---|
| `price_to_floor_ratio` | `past_avg_price / floor_price` |
| `supply_demand_ratio` | `listing_count / (sales_last_30d + 1)` |
| `sales_velocity` | `sales_last_7d / (sales_last_30d + 1)` |
| `social_score` | `log1p(twitter_followers) + log1p(discord_members)` |
| `engagement_weighted` | `social_score × engagement_rate` |
| `log_floor_price` | `log1p(floor_price)` |
| `log_past_avg_price` | `log1p(past_avg_price)` |
| `age_bucket` | Ordinal bin of `days_since_mint` (0–5) |

---

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| **R²** | Proportion of variance explained (higher is better, max 1.0) |
| **MAE** | Mean Absolute Error – average magnitude of price error in ETH |
| **RMSE** | Root Mean Squared Error – penalises large errors more heavily |

Cross-validation reports mean ± std across 5 folds.

---

## Sample Results

> Results are from the synthetic dataset with default hyperparameters.

| Model | CV R² | CV MAE | CV RMSE | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|
| GradientBoosting | 0.848 | 2.51 | 7.56 | 0.907 | 1.93 | 4.77 |
| RandomForest | 0.817 | 2.51 | 8.22 | **0.960** | **1.70** | **3.14** |
| LinearRegression | 0.794 | 3.99 | 8.26 | 0.900 | 3.18 | 4.95 |

RandomForest achieves the highest hold-out R² (0.960) despite GradientBoosting leading on CV R², which is a useful reminder that CV rank order does not always match hold-out rank order. Both ensemble models outperform Linear Regression substantially, confirming non-linear interactions in the features.

These figures reflect a synthetic dataset where prices are a deterministic (but noisy) function of the features. Real-world NFT data would produce lower R² due to irrational market dynamics, wash trading, and information not captured in the feature set.

### Output Plots

**Feature Importance (Gradient Boosting)**
`outputs/plots/feature_importance.png`

**Actual vs. Predicted**
`outputs/plots/actual_vs_predicted.png`

**Residual Diagnostic**
`outputs/plots/residuals.png`

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ValuArt.git
cd ValuArt
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This will:
- Generate `data/nft_sales.csv` (if absent)
- Train all three models with 5-fold CV
- Save the best model to `outputs/models/best_model.pkl`
- Write plots to `outputs/plots/`
- Print a model comparison table

### 5. Explore the notebook

```bash
jupyter notebook notebooks/valuart_demo.ipynb
```

### 6. Score new listings (CLI)

```bash
python -m src.predict --input data/new_listings.csv --output outputs/predictions.csv
```

---

## Project Limitations

- **Synthetic data**: The dataset was generated with a known price formula, so reported R² figures are higher than what real NFT data would produce. This is a portfolio demonstration, not a production trading signal.
- **No time-series modelling**: Prices are modelled cross-sectionally. Temporal trends, wash-trading patterns, and market regime shifts are not captured.
- **No on-chain data**: Transaction fees, wallet history, and smart-contract metadata are omitted.
- **Static social features**: Twitter/Discord counts are point-in-time snapshots; momentum (e.g., follower growth rate) is not modelled.
- **No hyperparameter search**: Models use manually set defaults. A proper grid/random search would likely improve results.

---

## Future Improvements

- [ ] Integrate real data from OpenSea or Dune Analytics APIs
- [ ] Add time-aware train/test splits to prevent data leakage
- [ ] Implement `Optuna` or `GridSearchCV` hyperparameter tuning
- [ ] Explore `XGBoost` / `LightGBM` for faster training on larger datasets
- [ ] Build a lightweight Flask / Streamlit demo app
- [ ] Add SHAP values for per-prediction explainability
- [ ] Dockerise the pipeline for reproducible deployments

---

## Tech Stack

- **Python 3.11**
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **scikit-learn** – modelling, preprocessing, evaluation
- **Matplotlib** – visualization
- **Joblib** – model serialization

---

## Author

Built as a portfolio ML project demonstrating a reproducible regression pipeline: synthetic data generation → feature engineering → cross-validated model selection → diagnostic evaluation.

*"Built a reproducible pipeline from NFT sales and social features and trained regressors with cross-validated R²/MAE to estimate sale prices."*
