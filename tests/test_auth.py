import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_login_valid_credentials():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "operatore", "password": "operatore123"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["username"] == "operatore"
    assert "user_id" in data["data"]
    assert data["message"] is None


@pytest.mark.asyncio
async def test_login_invalid_password():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "operatore", "password": "wrongpassword"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Credenziali non valide"


@pytest.mark.asyncio
async def test_login_nonexistent_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "inesistente", "password": "qualcosa"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Credenziali non valide"
