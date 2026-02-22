import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    import asyncio
    from app.database import create_tables, async_session
    from app.seed import seed_data

    async def _setup():
        await create_tables()
        async with async_session() as session:
            await seed_data(session)

    asyncio.run(_setup())
