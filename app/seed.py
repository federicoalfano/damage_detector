import uuid

import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.vehicle import Vehicle
from app.models.user import User


SEED_VEHICLES = [
    # Piaggio (ex "pulse" - Piaggio Ape)
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-piaggio-001")), "model": "Piaggio Ape", "plate": "AB12345", "type": "piaggio"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-piaggio-002")), "model": "Piaggio Ape", "plate": "CD67890", "type": "piaggio"},
    # Ligier
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-ligier-001")), "model": "Ligier", "plate": "EF11223", "type": "ligier"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-ligier-002")), "model": "Ligier", "plate": "GH44556", "type": "ligier"},
    # My Moover
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-mymoover-001")), "model": "My Moover", "plate": "IJ77889", "type": "my_moover"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-mymoover-002")), "model": "My Moover", "plate": "KL99001", "type": "my_moover"},
    # Scudo
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-scudo-001")), "model": "Fiat Scudo", "plate": "MN22334", "type": "scudo"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-scudo-002")), "model": "Fiat Scudo", "plate": "OP55667", "type": "scudo"},
]

SEED_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-operatore"))
SEED_USER_USERNAME = "operatore"
SEED_USER_PASSWORD = "operatore123"

SEED_TEST_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-test"))
SEED_TEST_USER_USERNAME = "test"
SEED_TEST_USER_PASSWORD = "test123"


async def seed_data(session: AsyncSession) -> None:
    # Check if new vehicle types already exist
    result = await session.execute(
        select(Vehicle).where(Vehicle.type == "piaggio").limit(1)
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
