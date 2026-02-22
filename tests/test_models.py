import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.database import Base
from app.models.vehicle import Vehicle
from app.models.user import User
from app.models.session import Session
from app.models.photo import Photo
from app.models.analysis import AnalysisResult, Damage


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_vehicle(db_session):
    vehicle = Vehicle(id="v-001", model="Piaggio Liberty 125", plate="AB12345", type="motorino")
    db_session.add(vehicle)
    await db_session.commit()

    result = await db_session.get(Vehicle, "v-001")
    assert result is not None
    assert result.model == "Piaggio Liberty 125"
    assert result.plate == "AB12345"
    assert result.type == "motorino"


@pytest.mark.asyncio
async def test_create_user(db_session):
    user = User(id="u-001", username="operatore", password_hash="hashed")
    db_session.add(user)
    await db_session.commit()

    result = await db_session.get(User, "u-001")
    assert result is not None
    assert result.username == "operatore"


@pytest.mark.asyncio
async def test_create_session(db_session):
    session = Session(
        id="s-001", vehicle_id="v-001", user_id="u-001",
        started_at="2026-02-21T10:00:00Z", status="in_progress",
        total_photos=4, valid_photos=0,
    )
    db_session.add(session)
    await db_session.commit()

    result = await db_session.get(Session, "s-001")
    assert result is not None
    assert result.vehicle_id == "v-001"
    assert result.total_photos == 4


@pytest.mark.asyncio
async def test_create_photo(db_session):
    photo = Photo(
        id="p-001", session_id="s-001", angle_index=0,
        angle_label="fronte", file_path="/data/p-001.jpg",
        captured_at="2026-02-21T10:05:00Z",
    )
    db_session.add(photo)
    await db_session.commit()

    result = await db_session.get(Photo, "p-001")
    assert result is not None
    assert result.angle_label == "fronte"
    assert result.upload_status == "pending"


@pytest.mark.asyncio
async def test_create_analysis_and_damage(db_session):
    analysis = AnalysisResult(
        id="a-001", session_id="s-001", status="completed",
        completed_at="2026-02-21T11:00:00Z",
    )
    db_session.add(analysis)
    await db_session.commit()

    damage = Damage(
        id="d-001", analysis_id="a-001", damage_type="graffio",
        severity="G2", zone="lato_sinistro",
        description="Graffio profondo",
    )
    db_session.add(damage)
    await db_session.commit()

    result = await db_session.get(Damage, "d-001")
    assert result is not None
    assert result.damage_type == "graffio"
    assert result.severity == "G2"
