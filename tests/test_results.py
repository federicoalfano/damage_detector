import asyncio
import io
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.seed import SEED_VEHICLES, SEED_USER_ID
from app.services.ai_service import analyze_session


async def _create_and_complete_session(client, angle_labels=None):
    """Create session, upload photos, complete it, and wait for analysis."""
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
async def test_results_analysis_without_api_key():
    """GET results returns error status when no API key is configured."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_and_complete_session(client)

        # Run analysis directly (no API key = error)
        await analyze_session(session_id)

        response = await client.get(f"/api/v1/sessions/{session_id}/results")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["analysis_status"] == "error"


@pytest.mark.asyncio
async def test_results_no_analysis():
    """GET results returns pending when no analysis exists."""
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

        response = await client.get(f"/api/v1/sessions/{session_id}/results")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["analysis_status"] == "pending"
    assert body["data"]["damages"] == []


@pytest.mark.asyncio
async def test_results_nonexistent_session():
    """GET results returns 404 for nonexistent session."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/sessions/nonexistent/results")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_sessions():
    """GET /sessions returns list of all sessions."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a session first
        await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": SEED_USER_ID,
            },
        )

        response = await client.get("/api/v1/sessions")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert isinstance(body["data"], list)
    assert len(body["data"]) >= 1
