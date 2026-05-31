import io
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.models.photo import Photo
from app.seed import SEED_VEHICLES, SEED_USER_ID
from app.services import ai_service
from app.services.ai_service import _call_openai, analyze_session

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
async def test_analyze_session_no_api_key():
    """Analysis without API key sets status to error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session_with_photos(client)

    await analyze_session(session_id)

    async with async_session() as db:
        result = await db.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        analysis = result.scalars().first()
        assert analysis is not None
        # Without OPENAI_API_KEY, analysis should error
        assert analysis.status == "error"


def _make_fake_photo(tmpdir: str, angle_label: str, idx: int) -> Photo:
    """Build a Photo row pointing at a real (tiny) JPEG file on disk."""
    file_path = os.path.join(tmpdir, f"{angle_label}.jpg")
    with open(file_path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return Photo(
        id=f"photo-{idx}",
        session_id="sess-x",
        angle_index=idx,
        angle_label=angle_label,
        file_path=file_path,
        captured_at="2026-04-20T00:00:00Z",
        is_valid=1,
        upload_status="completed",
    )


def _fake_openai_response(content: str):
    """Minimal stand-in for an OpenAI ChatCompletion response object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        model_dump_json=lambda: "{}",
    )


@pytest.mark.asyncio
async def test_call_openai_issues_one_request_per_photo(monkeypatch):
    """_call_openai must perform one chat completion per photo and aggregate results."""
    monkeypatch.setattr(ai_service.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(ai_service.settings, "openai_base_url", "")
    monkeypatch.setattr(ai_service.settings, "openai_model", "gpt-4o-mini")
    # This test asserts the per-photo routing invariant; pin to single pass.
    monkeypatch.setattr(ai_service.settings, "vlm_passes", 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        photos = [
            _make_fake_photo(tmpdir, "fronte", 0),
            _make_fake_photo(tmpdir, "lato_destro", 1),
            _make_fake_photo(tmpdir, "lato_sinistro", 2),
            _make_fake_photo(tmpdir, "retro", 3),
        ]

        per_angle_payload = {
            "fronte": '{"damages": [{"damage_type": "graffio", "severity": "lieve", "zone": "frontale", "description": "fronte-d"}]}',
            "lato_destro": '{"damages": [{"damage_type": "ammaccatura", "severity": "moderato", "zone": "laterale_destro", "description": "dx-d"}]}',
            "lato_sinistro": '{"damages": []}',
            "retro": '{"damages": [{"damage_type": "rottura", "severity": "grave", "zone": "posteriore", "description": "retro-d"}]}',
        }

        call_log: list[str] = []

        class FakeCompletions:
            def create(self, **kwargs):
                # Inspect the messages to figure out which angle was sent.
                content = kwargs["messages"][0]["content"]
                prompt_text = content[0]["text"]
                # Each per-angle prompt mentions its zone literal uniquely.
                if 'zone": "frontale"' in prompt_text:
                    angle = "fronte"
                elif 'zone": "laterale_destro"' in prompt_text:
                    angle = "lato_destro"
                elif 'zone": "laterale_sinistro"' in prompt_text:
                    angle = "lato_sinistro"
                elif 'zone": "posteriore"' in prompt_text:
                    angle = "retro"
                else:
                    angle = "unknown"
                call_log.append(angle)
                return _fake_openai_response(per_angle_payload[angle])

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        with patch("openai.OpenAI", FakeClient):
            damages, raw = await _call_openai(photos, vehicle_type="piaggio")

        assert sorted(call_log) == ["fronte", "lato_destro", "lato_sinistro", "retro"]
        assert len(damages) == 3
        zones = {d["zone"] for d in damages}
        assert zones == {"frontale", "laterale_destro", "posteriore"}
        for angle in ("fronte", "lato_destro", "lato_sinistro", "retro"):
            assert f"=== {angle} ===" in raw


@pytest.mark.asyncio
async def test_call_openai_multipass_unions_and_scores(monkeypatch):
    """Multi-pass keeps singleton findings (recall) and scores by agreement."""
    monkeypatch.setattr(ai_service.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(ai_service.settings, "openai_base_url", "")
    monkeypatch.setattr(ai_service.settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(ai_service.settings, "vlm_passes", 3)

    # 3 passes for one photo: graffio in all 3 (severity rising), a critical
    # crepa in only 1 — the nondeterministic case multi-pass exists to catch.
    pass_payloads = [
        '{"damages": [{"damage_type": "graffio", "severity": "lieve", "zone": "frontale", "description": "graffio cofano"}]}',
        '{"damages": ['
        '{"damage_type": "graffio", "severity": "lieve", "zone": "frontale", "description": "graffio cofano"},'
        '{"damage_type": "crepa", "severity": "moderato", "zone": "frontale", "description": "faro anteriore crepa"}]}',
        '{"damages": [{"damage_type": "graffio", "severity": "moderato", "zone": "frontale", "description": "graffio cofano"}]}',
    ]
    calls = {"n": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            i = calls["n"]
            calls["n"] += 1
            return _fake_openai_response(pass_payloads[i % len(pass_payloads)])

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = FakeChat()

    with tempfile.TemporaryDirectory() as tmpdir:
        photos = [_make_fake_photo(tmpdir, "fronte", 0)]
        with patch("openai.OpenAI", FakeClient):
            damages, _ = await _call_openai(photos, vehicle_type="piaggio")

    assert calls["n"] == 3  # 3 passes for the single photo
    by_type = {d["damage_type"]: d for d in damages}
    # Singleton critical finding must survive (recall-first, never consensus-filtered).
    assert set(by_type) == {"graffio", "crepa"}
    # graffio in 3/3 -> confidence 1.0; crepa in 1/3 -> 0.33.
    assert by_type["graffio"]["confidence"] == 1.0
    assert by_type["crepa"]["confidence"] == 0.33
    # severity escalates to the most severe reading across passes.
    assert by_type["graffio"]["severity"] == "moderato"


@pytest.mark.asyncio
async def test_call_openai_survives_single_photo_failure(monkeypatch):
    """One failing photo-call must not abort the whole session; others still aggregate."""
    monkeypatch.setattr(ai_service.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(ai_service.settings, "openai_base_url", "")
    monkeypatch.setattr(ai_service.settings, "openai_model", "gpt-4o-mini")

    with tempfile.TemporaryDirectory() as tmpdir:
        photos = [
            _make_fake_photo(tmpdir, "fronte", 0),
            _make_fake_photo(tmpdir, "retro", 1),
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                prompt_text = kwargs["messages"][0]["content"][0]["text"]
                if 'zone": "frontale"' in prompt_text:
                    raise RuntimeError("boom-fronte")
                return _fake_openai_response(
                    '{"damages": [{"damage_type": "graffio", "severity": "lieve", "zone": "posteriore", "description": "retro-d"}]}'
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        with patch("openai.OpenAI", FakeClient):
            damages, raw = await _call_openai(photos, vehicle_type="scudo")

        assert len(damages) == 1
        assert damages[0]["zone"] == "posteriore"
        assert "=== fronte ===" in raw
        assert "[ERROR] boom-fronte" in raw
        assert "=== retro ===" in raw


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
