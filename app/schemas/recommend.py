from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    mood: str
    occasion: str
    preferred_categories: List[str]

class RecommendResponse(BaseModel):
    recommended_products: List[str]
