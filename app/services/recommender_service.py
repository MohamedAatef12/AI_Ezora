from typing import List, Dict
from app.models.recommender.predict import (
    recommend_products,
    mood_encoder,
    occasion_encoder,
    category_encoder
)

def get_recommendations_from_user_context(
    mood: str,
    occasion: str,
    preferred_categories: List[str],
    top_k: int = 5
) -> List[Dict]:
    """
    Wrapper to get product recommendations based on user context.
    """
    return recommend_products(
        mood=mood,
        occasion=occasion,
        preferred_categories=preferred_categories,
        top_k=top_k
    )

def get_encoders():
    return mood_encoder, occasion_encoder, category_encoder