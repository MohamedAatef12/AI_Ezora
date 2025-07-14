import pandas as pd
import tensorflow as tf
import os
import ast
import pickle
from sklearn.preprocessing import LabelEncoder
from app.core.config import AppConfig

cfg = AppConfig()
RAW_DATA_PATH = cfg.get("data", "raw_data_path")
ENCODERS_PATH = cfg.get("encoders", "path")
os.makedirs(ENCODERS_PATH, exist_ok=True)

def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df["preferred_categories"] = df["preferred_categories"].apply(ast.literal_eval)
    df["categories_str"] = df["preferred_categories"].apply(lambda cats: " ".join(sorted(set(cats))))
    return df

def encode_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    encoders = {}

    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

        with open(f"{ENCODERS_PATH}/{col}_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    return df, encoders

def load_data():
    df = load_raw_data()
    df, encoders = encode_columns(df, ["mood", "occasion", "categories_str", "product_id"])
    
    X = df[["mood", "occasion", "categories_str"]].copy()
    y = df["product_id"].copy()
    return X, y, encoders

def create_dataset(X, y, batch_size=None, shuffle=True):
    batch_size = batch_size or cfg.get("model", "batch_size") or 64
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
