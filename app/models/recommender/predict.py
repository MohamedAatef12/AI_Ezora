import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List

# Paths
ENCODERS_DIR = "app/saved_models/encoders/encoders"
MODEL_PATH = "app/saved_models/models/model.keras"
DATA_PATH = "data/female_fashion_interactions.csv"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# Load encoders
def load_encoder(name: str):
    with open(os.path.join(ENCODERS_DIR, f"{name}_encoder.pkl"), "rb") as f:
        return pickle.load(f)

mood_encoder = load_encoder("mood")
occasion_encoder = load_encoder("occasion")
category_encoder = load_encoder("categories_str")
product_encoder = load_encoder("product_id")

# Load product data once
df_raw = pd.read_csv(DATA_PATH)
df_raw.drop_duplicates(subset="product_id", inplace=True)

# Recommendation function
def recommend_products(
    mood: str,
    occasion: str,
    preferred_categories: List[str],
    top_k: int = 5
) -> List[dict]:
    """
    Recommend top_k products based on user mood, occasion, and preferences.

    Args:
        mood (str): User mood (e.g., "happy")
        occasion (str): Occasion (e.g., "party")
        preferred_categories (List[str]): List of categories (e.g., ['bags', 'heels'])
        top_k (int): Number of products to return

    Returns:
        List[dict]: List of product details
    """
    try:
        mood_id = mood_encoder.transform([mood])[0]
        occasion_id = occasion_encoder.transform([occasion])[0]
        category_str = " ".join(sorted(set(preferred_categories)))
        category_id = category_encoder.transform([category_str])[0]

        preds = model.predict({
            "mood_input": np.array([mood_id]),
            "occasion_input": np.array([occasion_id]),
            "category_input": np.array([category_id]),
        }, verbose=0)

        top_indices = preds[0].argsort()[-top_k:][::-1]
        top_product_ids = product_encoder.inverse_transform(top_indices)

        recommended = df_raw[df_raw["product_id"].isin(top_product_ids)]

        return recommended.to_dict(orient="records")

    except Exception as e:
        print("Error during prediction:", e)
        raise e
