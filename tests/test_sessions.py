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


@pytest.mark.asyncio
async def test_create_session_with_name_persists_and_is_returned():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": SEED_USER_ID,
                "name": "AB123CD",
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()["data"]
        assert created["name"] == "AB123CD"
        session_id = created["id"]

        details_resp = await client.get(f"/api/v1/sessions/{session_id}/details")
        assert details_resp.status_code == 200
        details = details_resp.json()["data"]
        assert details["session"]["name"] == "AB123CD"

        list_resp = await client.get("/api/v1/sessions")
        assert list_resp.status_code == 200
        listed = list_resp.json()["data"]
        match = next((s for s in listed if s["id"] == session_id), None)
        assert match is not None
        assert match["name"] == "AB123CD"


@pytest.mark.asyncio
async def test_create_session_without_name_defaults_to_none():
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
    data = response.json()["data"]
    assert data.get("name") is None
