from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


def _get_database_url() -> str:
    url = settings.database_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _is_sqlite() -> bool:
    return settings.database_url.startswith("sqlite")


_database_url = _get_database_url()

_engine_kwargs: dict = {"echo": False}
if not _is_sqlite():
    _engine_kwargs.update(pool_size=5, max_overflow=10, pool_pre_ping=True)

engine = create_async_engine(_database_url, **_engine_kwargs)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

if _is_sqlite():
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(DeclarativeBase):
    pass


async def create_tables():
    async with engine.begin() as conn:
        from app.models import vehicle, session, photo, analysis, user  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with async_session() as session:
        yield session
