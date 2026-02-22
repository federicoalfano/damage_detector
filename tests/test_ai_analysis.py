import io
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.seed import SEED_VEHICLES, SEED_USER_ID
from app.services.ai_service import analyze_session, _mock_analysis

from sqlalchemy import select


async def _create_session_with_photos(client, angle_labels=None):
    """Helper to create a session, upload photos, and complete it."""
    if angle_labels is None:
        angle_labels = ["fronte", "lato_sinistro"]

    response = await client.post(
        "/api/v1/sessions",
        json={
            "vehicle_id": SEED_VEHICLES[0]["id"],
            "user_id": SEED_USER_ID,
        },
    )
    session_id = response.json()["data"]["id"]

    for i, label in enumerate(angle_labels):
        fake_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        await client.post(
            f"/api/v1/sessions/{session_id}/photos",
            files={"file": ("test.jpg", fake_image, "image/jpeg")},
            data={"angle_index": str(i), "angle_label": label},
        )

    return session_id


@pytest.mark.asyncio
async def test_analyze_session_creates_results():
    """AI analysis creates AnalysisResult and Damage records."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session_with_photos(client)

    # Run analysis directly (not via complete endpoint to avoid race)
    await analyze_session(session_id)

    async with async_session() as db:
        result = await db.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        analysis = result.scalars().first()
        assert analysis is not None
        assert analysis.status == "completed"

        result = await db.execute(
            select(Damage).where(Damage.analysis_id == analysis.id)
        )
        damages = result.scalars().all()
        assert len(damages) == 2  # fronte + lato_sinistro mock


@pytest.mark.asyncio
async def test_analyze_session_no_photos():
    """Analysis with no photos returns empty damages."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": SEED_USER_ID,
            },
        )
        session_id = response.json()["data"]["id"]

    await analyze_session(session_id)

    async with async_session() as db:
        result = await db.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        analysis = result.scalars().first()
        assert analysis is not None
        assert analysis.status == "completed"
        assert '"damages": []' in analysis.raw_response


@pytest.mark.asyncio
async def test_mock_analysis_fronte_only():
    """Mock analysis returns graffio for fronte photo."""

    class FakePhoto:
        angle_label = "fronte"

    damages = _mock_analysis([FakePhoto()])
    assert len(damages) == 1
    assert damages[0]["damage_type"] == "graffio"
    assert damages[0]["zone"] == "frontale"


@pytest.mark.asyncio
async def test_mock_analysis_no_matching_angles():
    """Mock analysis returns no damages for unknown angles."""

    class FakePhoto:
        angle_label = "retro"

    damages = _mock_analysis([FakePhoto()])
    assert len(damages) == 0
