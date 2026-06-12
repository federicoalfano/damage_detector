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

    consolidate_calls = {"n": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            text = kwargs["messages"][0]["content"][0]["text"]
            if "Raggruppa" in text:  # final text-only consolidate pass
                consolidate_calls["n"] += 1
                return _fake_openai_response('{"gruppi": []}')
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
    assert consolidate_calls["n"] == 1  # 2+ findings -> one dedup call
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
async def test_call_openai_raises_when_all_photos_fail(monkeypatch):
    """If EVERY photo-call fails (e.g. revoked API key -> 401 on all calls) the
    session must surface as an analysis ERROR, not 'completed with 0 damages':
    a dead key would otherwise render as a falsely intact vehicle."""
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
                raise RuntimeError("Error code: 401 - User not found.")

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        with patch("openai.OpenAI", FakeClient):
            with pytest.raises(RuntimeError, match="all 2 photo analyses failed"):
                await _call_openai(photos, vehicle_type="piaggio")


def test_build_api_kwargs_pass_temperature_schedule():
    """Pass 0 is the stable 0.2 baseline; passes >=1 run hotter to decorrelate.
    Reasoning models never receive a temperature, whatever the pass index."""
    content = [{"type": "text", "text": "x"}]
    assert ai_service._build_api_kwargs("google/gemini-2.5-flash", content, 0)["temperature"] == 0.2
    assert ai_service._build_api_kwargs("google/gemini-2.5-flash", content, 1)["temperature"] == 0.5
    assert ai_service._build_api_kwargs("google/gemini-2.5-flash", content, 2)["temperature"] == 0.5
    kr = ai_service._build_api_kwargs("openai/o4-mini", content, 1)
    assert "temperature" not in kr
    assert kr["max_completion_tokens"] == 8192


@pytest.mark.asyncio
async def test_scooter_checklist_backstop(monkeypatch):
    """Scooter checklist: a 'mancante' component becomes a pezzo_mancante flagged
    needs_review; 'non_visibile' (off-frame) is NOT a finding. This is the
    structural missing-part recall path, now enabled for scooters too."""
    monkeypatch.setattr(ai_service.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(ai_service.settings, "openai_base_url", "")
    monkeypatch.setattr(ai_service.settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(ai_service.settings, "vlm_passes", 1)

    payload = (
        '{"checklist": {"faro_anteriore": "ok", "specchietto_sinistro": "mancante", '
        '"specchietto_destro": "non_visibile", "ruota_anteriore": "ok"}, "damages": []}'
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_openai_response(payload)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = FakeChat()

    with tempfile.TemporaryDirectory() as tmpdir:
        photos = [_make_fake_photo(tmpdir, "fronte", 0)]
        with patch("openai.OpenAI", FakeClient):
            damages, _ = await _call_openai(photos, vehicle_type="piaggio")

    assert len(damages) == 1  # only the 'mancante'; non_visibile dropped
    d = damages[0]
    assert d["damage_type"] == "pezzo_mancante"
    assert d["zone"] == "frontale"
    assert d["needs_review"] is True
    assert d["severity"] == "grave"  # specchietto = safety component
    assert "specchietto sinistro" in (d["description"] or "").lower()


@pytest.mark.asyncio
async def test_focus_suffix_applied_on_later_passes(monkeypatch):
    """Pass 0 sends the bare prompt at temp 0.2; pass >=1 appends the focus
    suffix at temp 0.5 — decorrelating the union without adding any call."""
    monkeypatch.setattr(ai_service.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(ai_service.settings, "openai_base_url", "")
    monkeypatch.setattr(ai_service.settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(ai_service.settings, "vlm_passes", 2)

    seen: list[tuple[bool, float]] = []

    class FakeCompletions:
        def create(self, **kwargs):
            text = kwargs["messages"][0]["content"][0]["text"]
            seen.append(("SECONDA LETTURA" in text, kwargs.get("temperature")))
            return _fake_openai_response('{"damages": []}')

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = FakeChat()

    with tempfile.TemporaryDirectory() as tmpdir:
        photos = [_make_fake_photo(tmpdir, "fronte", 0)]
        with patch("openai.OpenAI", FakeClient):
            await _call_openai(photos, vehicle_type="piaggio")

    assert len(seen) == 2
    assert (False, 0.2) in seen  # baseline pass
    assert (True, 0.5) in seen   # focus pass


def test_tile_passes_union(monkeypatch):
    """The scudo tile pass runs settings.tile_passes times and UNIONS findings,
    so a critical found in only one nondeterministic run is still kept."""
    monkeypatch.setattr(ai_service.settings, "tile_passes", 2)
    # Bypass real image decode + reference lookup; we only test the union loop.
    monkeypatch.setattr(ai_service, "_pil_from_source", lambda *a, **k: object())
    monkeypatch.setattr(ai_service, "_reference_image_path", lambda *a, **k: None)

    runs = [
        [{"damage_type": "graffio", "severity": "lieve", "zone": "frontale",
          "description": "graffio cofano"}],
        [{"damage_type": "graffio", "severity": "lieve", "zone": "frontale",
          "description": "graffio cofano"},
         {"damage_type": "crepa", "severity": "grave", "zone": "frontale",
          "description": "faro anteriore destro crepa"}],
    ]
    calls = {"n": 0}

    def fake_pass(client, model, insp, ref, angle, pass_index=0):
        i = calls["n"]
        calls["n"] += 1
        return [dict(d) for d in runs[i % len(runs)]]

    monkeypatch.setattr(ai_service, "_tiled_detail_pass", fake_pass)

    photo = SimpleNamespace(file_path="x.jpg", image_data=None, angle_label="fronte")
    out = ai_service._run_tiled_for_photo(None, "m", photo, "scudo")

    assert calls["n"] == 2  # tile pass ran twice
    descs = {d["description"] for d in out}
    # the crepa seen only in the 2nd run survives the union (recall-first)
    assert "faro anteriore destro crepa" in descs
    assert "graffio cofano" in descs
    assert len(out) == 2  # deduped to the 2 unique findings


def test_tiling_skipped_for_non_reference_vehicle(monkeypatch):
    """tile_passes only affects scudo: scooters never enter the tile loop."""
    monkeypatch.setattr(ai_service.settings, "tile_passes", 3)
    called = {"n": 0}
    monkeypatch.setattr(ai_service, "_tiled_detail_pass",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])
    photo = SimpleNamespace(file_path="x.jpg", image_data=None, angle_label="fronte")
    out = ai_service._run_tiled_for_photo(None, "m", photo, "piaggio")
    assert out == []
    assert called["n"] == 0  # never tiled a scooter


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


def test_salient_nouns_canonicalization():
    """Paraphrases of the same physical damage produce the SAME dedup key:
    damage-kind words and relational filler drop, component synonyms collapse."""
    a = {"damage_type": "graffio", "severity": "moderato", "zone": "laterale_destro",
         "description": "graffi e rigature sulla parte inferiore della porta scorrevole destra, vicino al passaruota posteriore"}
    b = {"damage_type": "graffio", "severity": "moderato", "zone": "laterale_destro",
         "description": "Graffi e abrasioni sulla parte posteriore della porta scorrevole destra, vicino al passaruota posteriore."}
    assert ai_service._dedup_key(a) == ai_service._dedup_key(b)
    assert len(ai_service._merge_damages([a], [b])) == 1

    # L/R twins must NOT merge: side word survives canonicalization
    left = {"damage_type": "rottura", "severity": "grave", "zone": "frontale",
            "description": "faro anteriore sinistro rotto"}
    right = {"damage_type": "rottura", "severity": "grave", "zone": "frontale",
             "description": "faro anteriore destro rotto"}
    assert ai_service._dedup_key(left) != ai_service._dedup_key(right)


def _consolidate_fake_client(reply_json: str):
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _fake_openai_response(reply_json)

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    return FakeClient()


def test_consolidate_damages_merges_groups():
    """LLM groups duplicate paraphrases; code merges keeping max severity/conf,
    review flag only if EVERY sighting asked for it."""
    damages = [
        {"damage_type": "ammaccatura", "severity": "moderato", "zone": "laterale_destro",
         "description": "ammaccatura sul parafango posteriore destro, sopra il passaruota",
         "confidence": 1.0, "needs_review": False, "bounding_box": None},
        {"damage_type": "ammaccatura", "severity": "lieve", "zone": "laterale_destro",
         "description": "[DA VERIFICARE] fiancata posteriore ammaccata",
         "confidence": 0.4, "needs_review": True, "bounding_box": "0,0,10,10"},
        {"damage_type": "graffio", "severity": "moderato", "zone": "laterale_destro",
         "description": "graffi sulla portiera posteriore destra",
         "confidence": 0.5, "needs_review": False, "bounding_box": None},
    ]
    client = _consolidate_fake_client('{"gruppi": [[0, 1]]}')
    out = ai_service._consolidate_damages(client, "gpt-4o-mini", "lato_destro", damages)
    assert len(out) == 2
    merged = next(d for d in out if d["damage_type"] == "ammaccatura")
    assert merged["severity"] == "moderato"
    assert merged["confidence"] == 1.0
    assert merged["needs_review"] is False
    assert merged["bounding_box"] == "0,0,10,10"
    # the graffio not named in any group is untouched
    assert any(d["damage_type"] == "graffio" for d in out)


def test_consolidate_damages_kind_guard_and_fallback():
    """graffio+ammaccatura merge as ONE cosmetic damage (-> ammaccatura); a
    structural crepa never merges into a cosmetic group; a garbage LLM reply
    leaves the findings unchanged."""
    damages = [
        {"damage_type": "graffio", "severity": "moderato", "zone": "laterale_destro",
         "description": "graffi sopra la ruota", "confidence": 0.5, "needs_review": False},
        {"damage_type": "ammaccatura", "severity": "moderato", "zone": "laterale_destro",
         "description": "ammaccatura sopra la ruota posteriore", "confidence": 0.5, "needs_review": False},
        {"damage_type": "crepa", "severity": "grave", "zone": "laterale_destro",
         "description": "crepa sul vetro laterale", "confidence": 0.5, "needs_review": False},
    ]
    client = _consolidate_fake_client('{"gruppi": [[0, 1, 2]]}')
    out = ai_service._consolidate_damages(client, "gpt-4o-mini", "lato_destro", damages)
    assert len(out) == 2
    merged = next(d for d in out if d["damage_type"] == "ammaccatura")
    assert "ruota" in merged["description"]  # cosmetic pair folded into the dent
    assert any(d["damage_type"] == "crepa" for d in out)  # structural untouched

    client = _consolidate_fake_client("non-json garbage")
    out = ai_service._consolidate_damages(client, "gpt-4o-mini", "lato_destro", damages)
    assert out == damages  # best-effort: failure never loses findings


def test_cell_rect_str_mapping():
    """1-based grid cell -> overlapping _grid_boxes rect string ("x0,y0,x1,y1"),
    row-major, for both wide (3x2) and tall (2x3) photos; garbage -> None."""
    wide = (600, 300)   # 3 cols x 2 rows, nominal tile 200x150, 20% overlap
    assert ai_service._cell_rect_str(wide, 1) == "0,0,240,180"
    assert ai_service._cell_rect_str(wide, 6) == "360,120,600,300"
    tall = (300, 600)   # 2 cols x 3 rows, nominal tile 150x200
    assert ai_service._cell_rect_str(tall, 1) == "0,0,180,240"
    assert ai_service._cell_rect_str(tall, 6) == "120,360,300,600"
    # every cell matches the _grid_boxes rect at the same (row-major) index,
    # including int-like strings the model may emit as JSON map values
    for size in (wide, tall, (1920, 1080)):
        boxes = ai_service._grid_boxes(size)
        assert len(boxes) == 6
        for n, box in enumerate(boxes, start=1):
            expected = ",".join(str(v) for v in box)
            assert ai_service._cell_rect_str(size, n) == expected
            assert ai_service._cell_rect_str(size, str(n)) == expected
    # garbage in -> None out (out of range, bool, non-numeric, missing)
    for bad in (0, 7, -1, "0", "7", "x", "", None, True, False, [], {}):
        assert ai_service._cell_rect_str(wide, bad) is None


def _photo_stub(file_path="/nonexistent/photo.jpg"):
    """Photo-like object for _consolidate_damages; path does not exist so
    _pil_from_source returns None unless monkeypatched."""
    return SimpleNamespace(file_path=file_path, image_data=None)


def test_consolidate_grid_fills_null_boxes_only(monkeypatch):
    """A valid 'cella' fills a NULL bounding_box with the matching _grid_boxes
    rect of the ORIGINAL photo size; an existing tile-pass box is NEVER
    overwritten; entries the model gives no cell for stay null. The gridded
    image rides on the SAME single consolidate call (zero extra calls)."""
    from PIL import Image
    monkeypatch.setattr(ai_service, "_pil_from_source",
                        lambda *a, **k: Image.new("RGB", (600, 300), "gray"))
    damages = [
        {"damage_type": "graffio", "severity": "moderato", "zone": "frontale",
         "description": "graffio sul cofano", "confidence": 1.0,
         "needs_review": False, "bounding_box": None},
        {"damage_type": "crepa", "severity": "grave", "zone": "frontale",
         "description": "crepa sul faro destro", "confidence": 0.5,
         "needs_review": False, "bounding_box": "10,10,50,50"},
        {"damage_type": "ammaccatura", "severity": "lieve", "zone": "frontale",
         "description": "ammaccatura sul paraurti", "confidence": 0.4,
         "needs_review": True, "bounding_box": None},
    ]
    client = _consolidate_fake_client('{"gruppi": [], "celle": {"0": 2, "1": 5}}')
    out = ai_service._consolidate_damages(
        client, "gpt-4o-mini", "fronte", damages, photo=_photo_stub())
    assert len(out) == 3
    by_type = {d["damage_type"]: d for d in out}
    assert by_type["graffio"]["bounding_box"] == ai_service._cell_rect_str((600, 300), 2)
    assert by_type["crepa"]["bounding_box"] == "10,10,50,50"   # tile box untouched
    assert by_type["ammaccatura"]["bounding_box"] is None      # no cell -> stays null
    # piggyback contract: exactly ONE call, carrying the gridded image + the
    # extended prompt; the caller's input dicts are not mutated.
    calls = client.chat.completions.calls
    assert len(calls) == 1
    content = calls[0]["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert '"celle"' in content[0]["text"]
    assert damages[0]["bounding_box"] is None


def test_consolidate_grid_cell_localizes_merged_group(monkeypatch):
    """A merged group with all-null boxes takes its box from ANY member index
    the model assigned a cell to."""
    from PIL import Image
    monkeypatch.setattr(ai_service, "_pil_from_source",
                        lambda *a, **k: Image.new("RGB", (300, 600), "gray"))
    damages = [
        {"damage_type": "graffio", "severity": "moderato", "zone": "laterale_destro",
         "description": "graffi sopra la ruota", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
        {"damage_type": "ammaccatura", "severity": "moderato", "zone": "laterale_destro",
         "description": "ammaccatura sopra la ruota posteriore", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
    ]
    # cell only on member 1 — the merged entry must still get localized (tall photo)
    client = _consolidate_fake_client('{"gruppi": [[0, 1]], "celle": {"1": 6}}')
    out = ai_service._consolidate_damages(
        client, "gpt-4o-mini", "lato_destro", damages, photo=_photo_stub())
    assert len(out) == 1
    assert out[0]["bounding_box"] == "120,360,300,600"  # _grid_boxes((300,600))[5]


def test_consolidate_garbage_cella_leaves_boxes_null(monkeypatch):
    """Out-of-range, non-numeric or missing 'cella' values must never produce a
    bounding box; the consolidated findings are otherwise unchanged."""
    from PIL import Image
    monkeypatch.setattr(ai_service, "_pil_from_source",
                        lambda *a, **k: Image.new("RGB", (600, 300), "gray"))
    damages = [
        {"damage_type": "graffio", "severity": "lieve", "zone": "frontale",
         "description": f"graffio numero {i}", "confidence": 0.5,
         "needs_review": False, "bounding_box": None}
        for i in range(4)
    ]
    client = _consolidate_fake_client(
        '{"gruppi": [], "celle": {"0": 0, "1": 7, "2": "x"}}')  # index 3 missing
    out = ai_service._consolidate_damages(
        client, "gpt-4o-mini", "fronte", damages, photo=_photo_stub())
    assert out == damages  # nothing merged, no box materialized
    assert all(d["bounding_box"] is None for d in out)

    # "celle" that is not even a map is ignored wholesale
    client = _consolidate_fake_client('{"gruppi": [], "celle": [1, 2, 3, 4]}')
    out = ai_service._consolidate_damages(
        client, "gpt-4o-mini", "fronte", damages, photo=_photo_stub())
    assert all(d["bounding_box"] is None for d in out)


def test_consolidate_never_yields_more_than_input():
    """Adversarial reply referencing indices that do not exist: extras are
    dropped, the output can never exceed the input, and every output entry is
    one of (or a merge of) the given findings — nothing invented."""
    damages = [
        {"damage_type": "graffio", "severity": "moderato", "zone": "laterale_destro",
         "description": "graffi sopra la ruota", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
        {"damage_type": "ammaccatura", "severity": "moderato", "zone": "laterale_destro",
         "description": "ammaccatura sopra la ruota posteriore", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
        {"damage_type": "crepa", "severity": "grave", "zone": "laterale_destro",
         "description": "crepa sul vetro laterale", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
    ]
    client = _consolidate_fake_client(
        '{"gruppi": [[0, 1], [5, 6, 7], [2, 99], [1, 0]]}')
    out = ai_service._consolidate_damages(client, "gpt-4o-mini", "lato_destro", damages)
    assert len(out) <= len(damages)
    in_desc = {d["description"] for d in damages}
    assert all(d["description"] in in_desc for d in out)  # no invented findings
    # phantom group [5,6,7] contributed nothing; [0,1] merged; crepa survived
    assert len(out) == 2
    assert any(d["damage_type"] == "crepa" for d in out)


def test_consolidate_without_grid_stays_text_only():
    """No photo (or an undecodable one) -> exactly today's behavior: text-only
    prompt, no image part, no 'celle' request, boxes untouched."""
    damages = [
        {"damage_type": "graffio", "severity": "lieve", "zone": "frontale",
         "description": "graffio cofano", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
        {"damage_type": "crepa", "severity": "grave", "zone": "frontale",
         "description": "crepa faro", "confidence": 0.5,
         "needs_review": False, "bounding_box": None},
    ]
    for photo in (None, _photo_stub()):  # stub path does not exist -> PIL None
        client = _consolidate_fake_client('{"gruppi": []}')
        out = ai_service._consolidate_damages(
            client, "gpt-4o-mini", "fronte", damages, photo=photo)
        assert out == damages
        content = client.chat.completions.calls[0]["messages"][0]["content"]
        assert all(part.get("type") != "image_url" for part in content)
        assert '"celle"' not in content[0]["text"]
        assert content[0]["text"].rstrip().endswith("voci separate.")
        # historical token cap preserved on the text-only path
        assert client.chat.completions.calls[0].get("max_tokens") == 500
