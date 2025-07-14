# services/pipeline.py

from typing import List, Dict, Optional
from app.services.recommender_service import get_recommendations_from_user_context, get_encoders

# from app.models.nlp.extract import extract_user_context  ← هنضيفها لما نجهز الـ NLP

def run_recommendation_pipeline(
    mood: Optional[str] = None,
    occasion: Optional[str] = None,
    preferred_categories: Optional[List[str]] = None,
    message_text: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Main pipeline to generate recommendations based on user behavior or message.

    Args:
        mood (Optional[str]): Explicit mood input.
        occasion (Optional[str]): Explicit occasion input.
        preferred_categories (Optional[List[str]]): Categories user prefers.
        message_text (Optional[str]): Optional raw message for NLP extraction.
        top_k (int): Number of results to return.

    Returns:
        List[Dict]: Recommended products.
    """

    # 1. If text is provided, extract mood/occasion from it (TODO later)
    # if message_text:
    #     mood, occasion = extract_user_context(message_text)

    # 2. Call recommender logic
    # داخل recommend_products أو run_recommendation_pipeline
    mood_encoder, occasion_encoder, category_encoder = get_encoders()

    if mood not in mood_encoder.classes_:
        raise ValueError(f"Unknown mood: {mood}")
    if occasion not in occasion_encoder.classes_:
        raise ValueError(f"Unknown occasion: {occasion}")

    category_str = " ".join(sorted(set(preferred_categories)))
    if category_str not in category_encoder.classes_:
        return [] 
    if not (mood and occasion and preferred_categories):
        raise ValueError("Insufficient data provided for recommendation.")

    if not (mood and occasion and preferred_categories):
        raise ValueError("Insufficient data provided for recommendation.")

    return get_recommendations_from_user_context(
        mood=mood,
        occasion=occasion,
        preferred_categories=preferred_categories,
        top_k=top_k
    )
    
