import os
from app.models.recommender.data_loader import load_data, create_dataset
from app.models.recommender.recommender_model import build_recommender_model
import tensorflow as tf

MODEL_DIR = "app/saved_models/models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")

def train_model(epochs: int = 5):
    # Load and prepare data
    X, y, encoders = load_data()
    X = X.rename(columns={
        "mood": "mood_input",
        "occasion": "occasion_input",
        "categories_str": "category_input",
    })
    train_ds = create_dataset(X, y)

    # Get feature dimensions
    num_moods = len(encoders["mood"].classes_)
    num_occasions = len(encoders["occasion"].classes_)
    num_categories = len(encoders["categories_str"].classes_)
    num_products = len(encoders["product_id"].classes_)

    # Build and train model
    model = build_recommender_model(
        num_moods=num_moods,
        num_occasions=num_occasions,
        num_categories=num_categories,
        num_products=num_products,
    )

    model.fit(train_ds, epochs=epochs)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
