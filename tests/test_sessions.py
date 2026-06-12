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


@pytest.mark.asyncio
async def test_create_session_with_new_plate_creates_vehicle_and_links():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # New plate, not in the seed set. Normalization should uppercase and
        # strip whitespace: " zz 999 xx " -> "ZZ999XX".
        create_resp = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": SEED_USER_ID,
                "plate": " zz 999 xx ",
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()["data"]
        session_id = created["id"]
        # Session must NOT link to the referenced seed vehicle anymore.
        assert created["vehicle_id"] != SEED_VEHICLES[0]["id"]

        # A new vehicle with the normalized plate must now exist.
        vehicles_resp = await client.get("/api/v1/vehicles")
        plates = {v["plate"]: v for v in vehicles_resp.json()["data"]}
        assert "ZZ999XX" in plates
        new_vehicle = plates["ZZ999XX"]
        assert new_vehicle["id"] == created["vehicle_id"]
        # New vehicle inherits type/model from the referenced seed vehicle.
        assert new_vehicle["type"] == SEED_VEHICLES[0]["type"]
        assert new_vehicle["model"] == SEED_VEHICLES[0]["model"]

        # The sessions list must surface the captured plate.
        list_resp = await client.get("/api/v1/sessions")
        match = next(
            (s for s in list_resp.json()["data"] if s["id"] == session_id), None
        )
        assert match is not None
        assert match["vehicle_plate"] == "ZZ999XX"


@pytest.mark.asyncio
async def test_create_session_with_existing_plate_reuses_vehicle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        vehicles_before = (await client.get("/api/v1/vehicles")).json()["data"]
        count_before = len(vehicles_before)

        # Send a seed plate (with messy casing/whitespace) under a DIFFERENT
        # referenced vehicle_id. It should resolve to the existing vehicle.
        seed_plate = SEED_VEHICLES[2]["plate"]  # e.g. "IJ77889"
        messy_plate = f"  {seed_plate.lower()}  "
        create_resp = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[0]["id"],
                "user_id": SEED_USER_ID,
                "plate": messy_plate,
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()["data"]
        # Linked to the existing vehicle owning that plate, not a new one.
        assert created["vehicle_id"] == SEED_VEHICLES[2]["id"]

        # No duplicate vehicle was created.
        vehicles_after = (await client.get("/api/v1/vehicles")).json()["data"]
        assert len(vehicles_after) == count_before

        list_resp = await client.get("/api/v1/sessions")
        match = next(
            (s for s in list_resp.json()["data"] if s["id"] == created["id"]), None
        )
        assert match is not None
        assert match["vehicle_plate"] == seed_plate


@pytest.mark.asyncio
async def test_create_session_without_plate_keeps_given_vehicle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        vehicles_before = (await client.get("/api/v1/vehicles")).json()["data"]
        count_before = len(vehicles_before)

        create_resp = await client.post(
            "/api/v1/sessions",
            json={
                "vehicle_id": SEED_VEHICLES[1]["id"],
                "user_id": SEED_USER_ID,
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()["data"]
        # Old behavior: session keeps the referenced vehicle_id verbatim.
        assert created["vehicle_id"] == SEED_VEHICLES[1]["id"]

        # No vehicle was created when no plate is supplied.
        vehicles_after = (await client.get("/api/v1/vehicles")).json()["data"]
        assert len(vehicles_after) == count_before
