import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_get_vehicles_returns_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/vehicles")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert isinstance(body["data"], list)
    assert len(body["data"]) == 4


@pytest.mark.asyncio
async def test_get_vehicles_item_format():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/vehicles")

    body = response.json()
    vehicle = body["data"][0]
    assert "id" in vehicle
    assert "model" in vehicle
    assert "plate" in vehicle
    assert "type" in vehicle
