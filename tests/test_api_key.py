import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health_no_api_key_required():
    """Health endpoint is always accessible, even with API_KEY configured."""
    with patch("app.dependencies.settings") as mock_settings:
        mock_settings.api_key = "secret123"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_protected_route_no_key_configured():
    """When API_KEY is empty, routes are accessible without header."""
    with patch("app.dependencies.settings") as mock_settings:
        mock_settings.api_key = ""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/vehicles")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_protected_route_missing_key():
    """When API_KEY is set, request without header gets 403."""
    with patch("app.dependencies.settings") as mock_settings:
        mock_settings.api_key = "secret123"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/vehicles")
        assert response.status_code == 403
        assert "API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_protected_route_wrong_key():
    """When API_KEY is set, request with wrong header gets 403."""
    with patch("app.dependencies.settings") as mock_settings:
        mock_settings.api_key = "secret123"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/vehicles",
                headers={"X-API-Key": "wrong-key"},
            )
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_protected_route_correct_key():
    """When API_KEY is set, request with correct header succeeds."""
    with patch("app.dependencies.settings") as mock_settings:
        mock_settings.api_key = "secret123"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/vehicles",
                headers={"X-API-Key": "secret123"},
            )
        assert response.status_code == 200
