from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.models.predict import recommend_products

router = APIRouter()

# ✅ بيانات المنتج في الريسبونس
class ProductDetail(BaseModel):
    product_id: str
    product_type: str
    style: str
    color: str
    season: str
    interaction: Optional[str]
    timestamp: Optional[str]

# ✅ الريسبونس الكامل
class RecommendResponse(BaseModel):
    recommended_products: List[ProductDetail]

# ✅ الريكوست
class RecommendRequest(BaseModel):
    mood: str
    occasion: str
    preferred_categories: List[str]

# ✅ API endpoint
@router.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest):
    try:
        recommendations = recommend_products(
            mood=request.mood,
            occasion=request.occasion,
            preferred_categories=request.preferred_categories
        )
        return RecommendResponse(recommended_products=recommendations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
