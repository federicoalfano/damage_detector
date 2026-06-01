import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables, async_session
from app.dependencies import verify_api_key
from app.seed import seed_data
from app.services.ai_service import recover_pending_analyses
from app.routers.auth import router as auth_router
from app.routers.vehicles import router as vehicles_router
from app.routers.sessions import router as sessions_router
from app.utils.exceptions import register_exception_handlers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    async with async_session() as session:
        await seed_data(session)
    logger.info(
        "Active VLM: model=%s base_url=%s",
        settings.openai_model, settings.openai_base_url or "(OpenAI default)",
    )
    # Loud, non-silent guard: an empty api_key disables ALL auth (including the
    # DELETE / reanalyze routes that spend paid VLM quota). Surface a forgotten
    # env var instead of silently shipping an open instance.
    if not settings.api_key:
        msg = ("API_KEY is empty — the API is OPEN (no auth) and cost/destructive "
               "endpoints are exposed. Set API_KEY in production.")
        if settings.require_auth:
            logger.error("REFUSING INSECURE DEFAULTS: %s (REQUIRE_AUTH=true)", msg)
            raise RuntimeError(msg)
        logger.error("SECURITY: %s", msg)
    # Resume any analyses orphaned by a previous restart (Render free tier).
    await recover_pending_analyses()
    yield


app = FastAPI(
    title="DamageDetection API",
    description="Backend API per documentazione danni veicoli Poste Italiane",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Auth is a header (X-API-Key), not a cookie, so credentialed CORS is not
    # needed and pairing it with permissive origins is a footgun.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

_api_key_dep = [Depends(verify_api_key)]

app.include_router(auth_router, prefix="/api/v1", dependencies=_api_key_dep)
app.include_router(vehicles_router, prefix="/api/v1", dependencies=_api_key_dep)
app.include_router(sessions_router, prefix="/api/v1", dependencies=_api_key_dep)


@app.get("/health")
async def health_check():
    return {"status": "success", "data": {"service": "damage-detection-api", "version": "0.1.0"}, "message": None}
