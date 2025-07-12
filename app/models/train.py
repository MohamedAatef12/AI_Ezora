
import os
import tensorflow as tf
from data_loader import load_data, create_dataset
from recommender_model import build_recommender_model

# تحميل البيانات وتشفيرها
X, y, encoders = load_data()

# عدد الفئات لكل عمود (مطلوب للـ Embedding layers)
num_moods = len(encoders["mood"].classes_)
num_occasions = len(encoders["occasion"].classes_)
num_categories = len(encoders["categories_str"].classes_)
num_products = len(encoders["product_id"].classes_)

X = X.rename(columns={
    "mood": "mood_input",
    "occasion": "occasion_input",
    "categories_str": "category_input",
})
# تجهيز الداتا
train_ds = create_dataset(X, y)

# بناء الموديل
model = build_recommender_model(
    num_moods=num_moods,
    num_occasions=num_occasions,
    num_categories=num_categories,
    num_products=num_products,
)

# تدريب
model.fit(train_ds, epochs=5)

# حفظ الموديل
SAVE_DIR = "app/saved_models/models"
SAVE_PATH = os.path.join(SAVE_DIR, "model.keras")

os.makedirs(SAVE_DIR, exist_ok=True)
model.save(SAVE_PATH)

print(f"✅ Model trained and saved to {SAVE_PATH}")
