from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.pipeline import run_recommendation_pipeline

router = APIRouter(prefix='/recommendation', tags=["recommendation"])


# ✅ شكل المنتج في الريسبونس
class ProductDetail(BaseModel):
    product_id: str
    product_type: str
    style: str
    color: str
    season: str
    interaction: Optional[str]
    timestamp: Optional[str]


# ✅ الريكوست
class RecommendRequest(BaseModel):
    mood: str
    occasion: str
    preferred_categories: List[str]


# ✅ الريسبونس
class RecommendResponse(BaseModel):
    recommended_products: List[ProductDetail]


# ✅ Endpoint
@router.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest):
    try:
        results = run_recommendation_pipeline(
            mood=request.mood,
            occasion=request.occasion,
            preferred_categories=request.preferred_categories
        )

        # ❌ لا ترمي Error لو مفيش نتائج
        return RecommendResponse(recommended_products=results)

    except ValueError as ve:
        # ⛔️ هندل الأخطاء المعروفة مثل invalid mood/category
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

