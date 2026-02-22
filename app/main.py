from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables, async_session
from app.dependencies import verify_api_key
from app.seed import seed_data
from app.routers.auth import router as auth_router
from app.routers.vehicles import router as vehicles_router
from app.routers.sessions import router as sessions_router
from app.utils.exceptions import register_exception_handlers


async def _run_migrations():
    """Add new columns to existing tables (SQLite doesn't auto-add via create_all)."""
    from sqlalchemy import text
    async with async_session() as session:
        # Add image_data BLOB column to photos if missing
        result = await session.execute(text("PRAGMA table_info(photos)"))
        columns = {row[1] for row in result.fetchall()}
        if "image_data" not in columns:
            await session.execute(text("ALTER TABLE photos ADD COLUMN image_data BLOB"))
            await session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    await _run_migrations()
    async with async_session() as session:
        await seed_data(session)
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
    allow_credentials=True,
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
