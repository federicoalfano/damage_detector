import os
import tempfile
from pathlib import Path

import pytest


_TEST_DB_PATH = Path(tempfile.gettempdir()) / f"damage_detection_api_tests_{os.getpid()}.sqlite3"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH}"


@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    import asyncio

    # Disable API key auth for tests
    from app.config import settings

    settings.api_key = ""
    settings.openai_api_key = ""

    from app.database import Base, async_session, create_tables, engine
    from app.seed import seed_data

    async def _setup():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await create_tables()
        async with async_session() as session:
            await seed_data(session, reset_existing=True)

    asyncio.run(_setup())

    yield

    async def _teardown():
        await engine.dispose()

    asyncio.run(_teardown())
    _TEST_DB_PATH.unlink(missing_ok=True)
