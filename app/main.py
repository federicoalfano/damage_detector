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
    """Add new columns to existing tables."""
    import logging
    from sqlalchemy import text, inspect
    from app.database import engine

    logger = logging.getLogger(__name__)

    async with engine.connect() as conn:
        # Check if image_data column exists using SQLAlchemy inspector (works on all DBs)
        columns = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("photos")}
        )
        if "image_data" not in columns:
            logger.info("Adding image_data column to photos table")
            # BYTEA for PostgreSQL, BLOB for SQLite — use BYTEA which both understand
            col_type = "BYTEA" if not settings.database_url.startswith("sqlite") else "BLOB"
            await conn.execute(text(f"ALTER TABLE photos ADD COLUMN image_data {col_type}"))
            await conn.commit()
            logger.info("Migration complete: image_data column added")


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
