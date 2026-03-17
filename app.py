"""
app.py – ValuArt Streamlit web app.
Loads the trained model + preprocessor and lets users predict NFT sale prices.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# make sure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, generate_dataset
from src.feature_engineering import engineer_features
from src.preprocess import NFTPreprocessor, split_data
from src.train import cross_validate_models, train_final_models, save_best_model
from src.utils import outputs_path, ensure_dir

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ValuArt – NFT Price Estimator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

MODEL_PATH = outputs_path("models", "best_model.pkl")
PREP_PATH  = outputs_path("models", "preprocessor.pkl")
FI_PATH    = outputs_path("plots", "feature_importance.png")


@st.cache_resource(show_spinner="Training model on first run — this takes ~30 seconds …")
def load_or_train():
    """
    Load pre-trained artefacts if they exist, otherwise run the full pipeline.
    Cached so it only runs once per Streamlit session.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH):
        model = joblib.load(MODEL_PATH)
        prep  = joblib.load(PREP_PATH)
        return model, prep

    # artefacts missing → train from scratch
    raw_df  = load_data()
    df_eng  = engineer_features(raw_df)
    train_df, _ = split_data(df_eng, test_size=0.20, random_state=42)

    prep = NFTPreprocessor()
    X_train, y_train = prep.fit_transform(train_df)

    cv_results    = cross_validate_models(X_train, y_train)
    fitted_models = train_final_models(X_train, y_train)
    save_best_model(fitted_models, cv_results)

    ensure_dir(outputs_path("models"))
    joblib.dump(prep, PREP_PATH)

    model = joblib.load(MODEL_PATH)
    return model, prep


def predict_price(model, prep, inputs: dict) -> float:
    """Run a single-row prediction through the full feature pipeline."""
    row = pd.DataFrame([inputs])
    row["sale_price"] = 1.0          # dummy target (non-zero so cleaner doesn't drop it)
    row_eng = engineer_features(row)
    X, _   = prep.transform(row_eng)
    return float(model.predict(X)[0])


# ── Load model ───────────────────────────────────────────────────────────────

model, prep = load_or_train()

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2.5rem 2rem 2rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;">
    <h1 style="color:#e94560; margin:0; font-size:2.6rem; font-weight:800; letter-spacing:-1px;">
        🎨 ValuArt
    </h1>
    <p style="color:#a8b2d8; margin:0.4rem 0 0 0; font-size:1.15rem;">
        NFT Sale Price Estimator &nbsp;·&nbsp; Machine Learning Regression Pipeline
    </p>
    <p style="color:#64748b; margin:0.6rem 0 0 0; font-size:0.85rem;">
        Built by <strong style="color:#94a3b8;">Parv Arora</strong>
        &nbsp;·&nbsp; Random Forest · Gradient Boosting · Linear Regression
        &nbsp;·&nbsp; 5-Fold Cross-Validation
    </p>
</div>
""", unsafe_allow_html=True)

# ── Layout: sidebar inputs + main prediction ─────────────────────────────────

st.sidebar.markdown("## 🔧 NFT Listing Details")
st.sidebar.markdown("Adjust the sliders to match your NFT and click **Predict**.")

with st.sidebar:
    st.markdown("#### Collection Info")
    collection_name = st.selectbox("Collection", [
        "Bored Ape YC", "Azuki", "CryptoPunks", "Doodles", "CloneX",
        "Moonbirds", "Pudgy Penguins", "Cool Cats", "World of Women",
        "Chromie Squiggle", "Art Blocks", "Meebits", "Mutant Ape YC",
    ])
    creator = st.selectbox("Creator", [
        "YugaLabs", "Azuki_Team", "LarvaLabs", "Evan_Conceicao",
        "KevinRose_Team", "CoolCats_Dev", "WoW_NFT", "SnowFro",
        "Meebits_Dev", "RTFKTstudios",
    ])
    blockchain = st.selectbox("Blockchain", ["ethereum", "polygon", "solana"])

    st.markdown("#### Token Stats")
    rarity_score  = st.slider("Rarity Score", 1.0, 500.0, 120.0, step=1.0)
    trait_count   = st.slider("Trait Count", 1, 15, 7)
    floor_price   = st.slider("Floor Price (ETH)", 0.01, 150.0, 10.0, step=0.1)
    past_avg_price = st.slider("Past Avg Sale Price (ETH)", 0.01, 300.0,
                                round(floor_price * 1.5, 1), step=0.1)

    st.markdown("#### Market Activity")
    sales_last_7d  = st.slider("Sales – Last 7 Days",  0, 200, 40)
    sales_last_30d = st.slider("Sales – Last 30 Days", 0, 800, 160)
    listing_count  = st.slider("Active Listings", 10, 5000, 320)
    days_since_mint = st.slider("Days Since Mint", 1, 1200, 365)

    st.markdown("#### Social Presence")
    twitter_followers = st.number_input("Twitter Followers", 500, 2_000_000, 250_000, step=1000)
    discord_members   = st.number_input("Discord Members",   100,   500_000,  80_000, step=1000)
    engagement_rate   = st.slider("Engagement Rate", 0.001, 0.15, 0.035, step=0.001,
                                   format="%.3f")

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Sale Price", use_container_width=True, type="primary")

# ── Main area ────────────────────────────────────────────────────────────────

col_pred, col_info = st.columns([1.1, 1], gap="large")

with col_pred:
    st.markdown("### Predicted Sale Price")

    if predict_btn:
        inputs = {
            "collection_name":   collection_name,
            "creator":           creator,
            "blockchain":        blockchain,
            "rarity_score":      rarity_score,
            "trait_count":       trait_count,
            "floor_price":       floor_price,
            "past_avg_price":    past_avg_price,
            "sales_last_7d":     sales_last_7d,
            "sales_last_30d":    sales_last_30d,
            "listing_count":     listing_count,
            "days_since_mint":   days_since_mint,
            "twitter_followers": twitter_followers,
            "discord_members":   discord_members,
            "engagement_rate":   engagement_rate,
        }

        price = predict_price(model, prep, inputs)
        low   = round(price * 0.82, 3)
        high  = round(price * 1.18, 3)

        st.markdown(f"""
        <div style="background:#0f3460; border:2px solid #e94560; border-radius:12px;
                    padding:2rem; text-align:center; margin-top:0.5rem;">
            <p style="color:#a8b2d8; margin:0; font-size:1rem;">Estimated Sale Price</p>
            <h1 style="color:#e94560; font-size:3.2rem; margin:0.3rem 0; font-weight:900;">
                {price:.3f} ETH
            </h1>
            <p style="color:#64748b; margin:0; font-size:0.9rem;">
                Confidence range &nbsp; <strong style="color:#94a3b8;">{low} – {high} ETH</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # quick feature breakdown
        st.markdown("#### Key signals used")
        ratio = round(past_avg_price / (floor_price + 1e-6), 2)
        velocity = round(sales_last_7d / (sales_last_30d + 1), 2)
        social = round(np.log1p(twitter_followers) + np.log1p(discord_members), 2)

        m1, m2, m3 = st.columns(3)
        m1.metric("Price / Floor Ratio", f"{ratio}×")
        m2.metric("Sales Velocity", f"{velocity:.2f}")
        m3.metric("Social Score", f"{social:.1f}")

    else:
        st.markdown("""
        <div style="background:#1e293b; border:1px dashed #334155; border-radius:12px;
                    padding:2.5rem; text-align:center; color:#64748b; margin-top:0.5rem;">
            <p style="font-size:2rem; margin:0;">🔮</p>
            <p style="margin:0.5rem 0 0 0;">
                Adjust the sliders on the left and click <strong>Predict Sale Price</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    st.markdown("### Model Performance")

    perf_data = {
        "Model": ["Random Forest", "Gradient Boosting", "Linear Regression"],
        "Test R²": [0.9595, 0.9066, 0.8995],
        "Test MAE (ETH)": [1.70, 1.93, 3.18],
    }
    perf_df = pd.DataFrame(perf_data)

    # colour the best row
    def highlight_best(row):
        if row["Model"] == "Random Forest":
            return ["background-color: #0f3460; color: #e94560; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        perf_df.style.apply(highlight_best, axis=1).format({"Test R²": "{:.4f}", "Test MAE (ETH)": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    st.caption("Evaluated on a held-out 20% test split (600 rows). Random Forest selected as best model.")

    # feature importance plot
    st.markdown("### Feature Importances")
    if os.path.exists(FI_PATH):
        st.image(FI_PATH, use_container_width=True)
    else:
        st.info("Run `python main.py` once to generate the feature importance chart.")

# ── How it works ─────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("📖  How it works", expanded=False):
    st.markdown("""
    **ValuArt** is a regression pipeline trained on NFT sales data.
    It learns the relationship between a listing's features and its final sale price.

    **Pipeline steps:**
    1. **Data** – 3,000 synthetic NFT sales records with realistic price distributions
    2. **Feature engineering** – price-to-floor ratio, sales velocity, log-scaled social counts, age buckets
    3. **Preprocessing** – median imputation, ordinal encoding (blockchain), target encoding (collection/creator)
    4. **Training** – three regressors compared via 5-fold cross-validation
    5. **Selection** – best model saved and loaded here for inference

    **Models trained:** Linear Regression · Random Forest · Gradient Boosting

    **Metrics:** R² · MAE · RMSE on a held-out 20% test set

    > *Note: trained on synthetic data. Predictions are illustrative — not financial advice.*
    """)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.82rem; margin-top:3rem; padding-top:1rem;
            border-top: 1px solid #1e293b;">
    Built by <strong style="color:#64748b;">Parv Arora</strong>
    &nbsp;·&nbsp; ValuArt NFT Pricing Model
    &nbsp;·&nbsp; scikit-learn · pandas · Streamlit
</div>
""", unsafe_allow_html=True)
