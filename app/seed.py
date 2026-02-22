import uuid

import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.vehicle import Vehicle
from app.models.user import User


SEED_VEHICLES = [
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-pulse-001")), "model": "Piaggio Ape Pulse", "plate": "AB12345", "type": "pulse"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-pulse-002")), "model": "Piaggio Ape Pulse", "plate": "CD67890", "type": "pulse"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-hurba-001")), "model": "Hurba", "plate": "EF11223", "type": "hurba"},
    {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-hurba-002")), "model": "Hurba", "plate": "GH44556", "type": "hurba"},
]

SEED_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-operatore"))
SEED_USER_USERNAME = "operatore"
SEED_USER_PASSWORD = "operatore123"


async def seed_data(session: AsyncSession) -> None:
    result = await session.execute(select(Vehicle).limit(1))
    if result.scalars().first() is not None:
        return

    for v in SEED_VEHICLES:
        session.add(Vehicle(**v))

    password_hash = bcrypt.hashpw(SEED_USER_PASSWORD.encode(), bcrypt.gensalt()).decode()
    session.add(User(
        id=SEED_USER_ID,
        username=SEED_USER_USERNAME,
        password_hash=password_hash,
    ))

    await session.commit()
