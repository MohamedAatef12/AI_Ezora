from fastapi import Request
from fastapi.responses import JSONResponse
from app.log.logger import get_logger

logger = get_logger("AI_Ezora")

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )