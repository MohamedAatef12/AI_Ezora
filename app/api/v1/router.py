from fastapi import APIRouter
from app.api.v1.endpoints import recommend

router = APIRouter()
router.include_router(recommend.router,tags=["recommendation"])
