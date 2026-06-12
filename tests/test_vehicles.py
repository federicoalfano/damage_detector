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
    # At least the 4 seed vehicles are present. Sessions created with a new
    # captured plate mint extra vehicles into the shared test DB, so this is a
    # lower bound rather than an exact count.
    seed_plates = {"AB12345", "EF11223", "IJ77889", "MN22334"}
    listed_plates = {v["plate"] for v in body["data"]}
    assert seed_plates.issubset(listed_plates)


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
