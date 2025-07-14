from fastapi import FastAPI
from app.api.v1.router import router as api_router
from log.middleware import ErrorHandlingMiddleware

app = FastAPI(title="AI_Ezora")

app.add_middleware(ErrorHandlingMiddleware)

app.include_router(api_router, prefix="/api/v1")
