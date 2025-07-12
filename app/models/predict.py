import tensorflow as tf
import pickle
import numpy as np
import os
import pandas as pd

ENCODERS_DIR = "app/saved_models/encoders/encoders"
MODEL_PATH = "app/saved_models/models/model.keras"
model = tf.keras.models.load_model(MODEL_PATH)


# تحميل الـ encoders
def load_encoder(name):
    with open(f"{ENCODERS_DIR}/{name}_encoder.pkl", "rb") as f:
        return pickle.load(f)

mood_encoder = load_encoder("mood")
occasion_encoder = load_encoder("occasion")
category_encoder = load_encoder("categories_str")
product_encoder = load_encoder("product_id")

# توصية
# حمّل الداتا الأصلية مرة واحدة
df_raw = pd.read_csv("female_fashion_interactions.csv")
df_raw.drop_duplicates(subset="product_id", inplace=True)  # في حالة تكرار المنتج

def recommend_products(mood: str, occasion: str, preferred_categories: list[str], top_k=5) -> list[dict]:
    try:
        mood_id = mood_encoder.transform([mood])[0]
        occasion_id = occasion_encoder.transform([occasion])[0]
        category_str = " ".join(sorted(set(preferred_categories)))
        category_id = category_encoder.transform([category_str])[0]

        preds = model.predict({
            "mood_input": np.array([mood_id]),
            "occasion_input": np.array([occasion_id]),
            "category_input": np.array([category_id])
        }, verbose=0)

        top_k_indices = preds[0].argsort()[-top_k:][::-1]
        top_product_ids = product_encoder.inverse_transform(top_k_indices)

        # إرجاع تفاصيل المنتجات بدلًا من فقط الـ IDs
        recommended_products = df_raw[df_raw["product_id"].isin(top_product_ids)]
        return recommended_products.to_dict(orient="records")

    except Exception as e:
        print("Error during prediction:", e)
        return []