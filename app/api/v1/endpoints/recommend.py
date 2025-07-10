from fastapi import APIRouter
from app.schemas.recommend import RecommendRequest, RecommendResponse
from app.services.recommender_service import recommend_products

router = APIRouter()

@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    result = recommend_products(
        mood=request.mood,
        occasion=request.occasion,
        preferred_categories=request.preferred_categories
    )
    return RecommendResponse(recommended_products=result)
