import uuid

import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.vehicle import Vehicle
from app.models.user import User


SEED_VEHICLES = [
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-piaggio-001")),
        "model": "Piaggio Liberty",
        "plate": "AB12345",
        "type": "piaggio",
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-ligier-001")),
        "model": "Ligier",
        "plate": "EF11223",
        "type": "ligier",
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-mymoover-001")),
        "model": "My Moover",
        "plate": "IJ77889",
        "type": "my_moover",
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-scudo-001")),
        "model": "Fiat Scudo",
        "plate": "MN22334",
        "type": "scudo",
    },
]

SEED_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-operatore"))
SEED_USER_USERNAME = "operatore"
SEED_USER_PASSWORD = "operatore123"

SEED_TEST_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "user-test"))
SEED_TEST_USER_USERNAME = "test"
SEED_TEST_USER_PASSWORD = "test123"

_SEED_USERS = [
    {
        "id": SEED_USER_ID,
        "username": SEED_USER_USERNAME,
        "password": SEED_USER_PASSWORD,
    },
    {
        "id": SEED_TEST_USER_ID,
        "username": SEED_TEST_USER_USERNAME,
        "password": SEED_TEST_USER_PASSWORD,
    },
]


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


async def _upsert_vehicle(session: AsyncSession, payload: dict[str, str]) -> None:
    vehicle = await session.get(Vehicle, payload["id"])
    if vehicle is None:
        session.add(Vehicle(**payload))
        return

    vehicle.model = payload["model"]
    vehicle.plate = payload["plate"]
    vehicle.type = payload["type"]


async def _upsert_user(
    session: AsyncSession,
    payload: dict[str, str],
    *,
    reset_existing: bool,
) -> None:
    user = await session.get(User, payload["id"])
    if user is None:
        user = User(
            id=payload["id"],
            username=payload["username"],
            password_hash=_hash_password(payload["password"]),
            enabled_until="2026-12-31",
            remaining_calls=50,
        )
        session.add(user)
        return

    user.username = payload["username"]
    if reset_existing:
        user.password_hash = _hash_password(payload["password"])
        user.enabled_until = "2026-12-31"
        user.remaining_calls = 50


async def seed_data(session: AsyncSession, *, reset_existing: bool = False) -> None:
    """Create or update deterministic seed rows without deleting user data.

    Production startup should not wipe sessions or reset quotas. Tests can pass
    reset_existing=True to force deterministic credentials and counters.
    """
    for vehicle in SEED_VEHICLES:
        await _upsert_vehicle(session, vehicle)

    for user in _SEED_USERS:
        await _upsert_user(session, user, reset_existing=reset_existing)

    await session.commit()
