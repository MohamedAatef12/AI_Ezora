import pandas as pd
import tensorflow as tf
import ast
from sklearn.preprocessing import LabelEncoder
import os
import pickle

RAW_DATA_PATH = "female_fashion_interactions.csv"
df_raw = pd.read_csv(RAW_DATA_PATH)
ENCODERS_DIR = "app/saved_models/encoders/encoders"
os.makedirs(ENCODERS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df["preferred_categories"] = df["preferred_categories"].apply(ast.literal_eval)

    # Flatten categories into one string
    df["categories_str"] = df["preferred_categories"].apply(lambda cats: " ".join(sorted(set(cats))))

    # Encode each feature
    encoders = {}

    def encode_column(col):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        with open(f"{ENCODERS_DIR}/{col}_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    for col in ["mood", "occasion", "categories_str", "product_id"]:
        encode_column(col)

    X = df[["mood", "occasion", "categories_str"]].copy()
    y = df["product_id"].copy()

    return X, y, encoders

def create_dataset(X, y, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
