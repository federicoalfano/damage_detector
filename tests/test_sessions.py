import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.seed import SEED_VEHICLES, SEED_USER_ID


@pytest.mark.asyncio
async def test_create_session_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": SEED_USER_ID,
            },
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert data["vehicle_id"] == SEED_VEHICLES[0]["id"]
    assert data["user_id"] == SEED_USER_ID
    assert data["status"] == "in_progress"
    assert data["total_photos"] == 4
    assert data["valid_photos"] == 0


@pytest.mark.asyncio
async def test_create_session_invalid_vehicle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": "nonexistent-id",
                "user_id": SEED_USER_ID,
            },
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_session_invalid_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": "nonexistent-user",
            },
        )

    assert response.status_code == 404
