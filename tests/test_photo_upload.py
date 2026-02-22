import io
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.seed import SEED_VEHICLES, SEED_USER_ID


async def _create_session(client):
    """Helper to create a session and return its ID."""
    response = await client.post(
        "/api/v1/sessions",
        json={
            "vehicle_id": SEED_VEHICLES[0]["id"],
            "user_id": SEED_USER_ID,
        },
    )
    return response.json()["data"]["id"]


@pytest.mark.asyncio
async def test_upload_photo_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

        fake_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        response = await client.post(
            f"/api/v1/sessions/{session_id}/photos",
            files={"file": ("test.jpg", fake_image, "image/jpeg")},
            data={"angle_index": "0", "angle_label": "fronte"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert "photo_id" in body["data"]


@pytest.mark.asyncio
async def test_upload_photo_invalid_session():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        response = await client.post(
            "/api/v1/sessions/nonexistent/photos",
            files={"file": ("test.jpg", fake_image, "image/jpeg")},
            data={"angle_index": "0", "angle_label": "fronte"},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_complete_session():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

        # Upload a photo first
        fake_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        await client.post(
            f"/api/v1/sessions/{session_id}/photos",
            files={"file": ("test.jpg", fake_image, "image/jpeg")},
            data={"angle_index": "0", "angle_label": "fronte"},
        )

        response = await client.post(f"/api/v1/sessions/{session_id}/complete")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["status"] == "uploaded"
    assert body["data"]["valid_photos"] == 1


@pytest.mark.asyncio
async def test_complete_nonexistent_session():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/sessions/nonexistent/complete")

    assert response.status_code == 404
