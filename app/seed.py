import uuid

import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.vehicle import Vehicle
from app.models.user import User


SEED_VEHICLES = [
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-piaggio-001")), "model": "Piaggio Liberty", "plate": "AB12345", "type": "piaggio"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-ligier-001")), "model": "Ligier", "plate": "EF11223", "type": "ligier"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-mymoover-001")), "model": "My Moover", "plate": "IJ77889", "type": "my_moover"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-scudo-001")), "model": "Fiat Scudo", "plate": "MN22334", "type": "scudo"},
]

SEED_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-operatore"))
SEED_USER_USERNAME = "operatore"
SEED_USER_PASSWORD = "operatore123"

SEED_TEST_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-test"))
SEED_TEST_USER_USERNAME = "test"
SEED_TEST_USER_PASSWORD = "test123"


async def seed_data(session: AsyncSession) -> None:
    # Check if seed is up-to-date (exactly 4 vehicles = 1 per type)
    from sqlalchemy import func
    count_result = await session.execute(select(func.count()).select_from(Vehicle))
    vehicle_count = count_result.scalar() or 0
    if vehicle_count == len(SEED_VEHICLES):
        # Verify it's the right set
        result = await session.execute(
            select(Vehicle).where(Vehicle.model == "Piaggio Liberty").limit(1)
        )
        if result.scalars().first() is not None:
            return

    # Remove all old data (respecting FK order: damages -> analyses -> photos -> sessions -> vehicles/users)
    from sqlalchemy import delete
    from app.models.analysis import Damage, AnalysisResult
    from app.models.photo import Photo
    from app.models.session import Session
    await session.execute(delete(Damage))
    await session.execute(delete(AnalysisResult))
    await session.execute(delete(Photo))
    await session.execute(delete(Session))
    await session.execute(delete(Vehicle))
    await session.execute(delete(User))
    await session.flush()

    for v in SEED_VEHICLES:
        session.add(Vehicle(**v))

    password_hash = bcrypt.hashpw(SEED_USER_PASSWORD.encode(), bcrypt.gensalt()).decode()
    session.add(User(
        id=SEED_USER_ID,
        username=SEED_USER_USERNAME,
        password_hash=password_hash,
        enabled_until="2026-12-31",
        remaining_calls=50,
    ))

    test_password_hash = bcrypt.hashpw(SEED_TEST_USER_PASSWORD.encode(), bcrypt.gensalt()).decode()
    session.add(User(
        id=SEED_TEST_USER_ID,
        username=SEED_TEST_USER_USERNAME,
        password_hash=test_password_hash,
        enabled_until="2026-12-31",
        remaining_calls=50,
    ))

    await session.commit()
