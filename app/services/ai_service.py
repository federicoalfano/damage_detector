import asyncio
import base64
import json
import logging
import os
import re
import uuid

from sqlalchemy import select

from app.config import settings
from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.models.photo import Photo
from app.models.session import Session
from app.models.user import User
from app.models.vehicle import Vehicle

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "prompts",
)

# Map vehicle type -> subdirectory under prompts/ holding per-angle files.
PROMPT_SUBDIR_BY_VEHICLE_TYPE = {
    "piaggio": "scooter",
    "ligier": "scooter",
    "my_moover": "scooter",
    "scudo": "scudo",
}

# Legacy generic prompts used as fallback if a per-angle file is missing.
FALLBACK_PROMPT_BY_VEHICLE_TYPE = {
    "scudo": "damage_analysis_scudo.txt",
}
DEFAULT_FALLBACK_PROMPT_FILE = "damage_analysis.txt"

ANGLE_LABELS = {
    "fronte": "FOTO FRONTALE",
    "lato_destro": "FOTO LATO DESTRO",
    "lato_sinistro": "FOTO LATO SINISTRO",
    "retro": "FOTO POSTERIORE",
}


def _load_prompt(vehicle_type: str | None = None, angle_label: str | None = None) -> str:
    """Load the per-angle prompt for the given vehicle type.

    Falls back to the generic prompt if the per-angle file is not available.
    """
    subdir = PROMPT_SUBDIR_BY_VEHICLE_TYPE.get(vehicle_type or "")
    if subdir and angle_label:
        per_angle_path = os.path.join(PROMPTS_DIR, subdir, f"{angle_label}.txt")
        if os.path.exists(per_angle_path):
            logger.info(
                "Loading per-angle prompt vehicle_type=%s angle=%s -> %s/%s.txt",
                vehicle_type, angle_label, subdir, angle_label,
            )
            with open(per_angle_path) as f:
                return f.read()
        logger.warning(
            "Per-angle prompt missing for vehicle_type=%s angle=%s, falling back to generic",
            vehicle_type, angle_label,
        )

    fallback = FALLBACK_PROMPT_BY_VEHICLE_TYPE.get(vehicle_type or "", DEFAULT_FALLBACK_PROMPT_FILE)
    path = os.path.join(PROMPTS_DIR, fallback)
    logger.info("Loading fallback prompt vehicle_type=%s -> %s", vehicle_type, fallback)
    with open(path) as f:
        return f.read()


def _encode_image_base64(file_path: str, fallback_bytes: bytes | None = None) -> str | None:
    """Read an image file, apply EXIF orientation, return base64-encoded JPEG.

    Falls back to [fallback_bytes] (DB blob) when the disk file is missing —
    Render free tier wipes data/sessions on cold restart.
    """
    raw: bytes | None = None
    if file_path and os.path.exists(file_path):
        size = os.path.getsize(file_path)
        logger.info("Reading photo: %s (%d bytes / %.1f KB)", file_path, size, size / 1024)
        if size > 0:
            with open(file_path, "rb") as f:
                raw = f.read()
        else:
            logger.warning("Photo file is EMPTY: %s", file_path)

    if raw is None and fallback_bytes:
        logger.info("Photo file unavailable on disk — using DB blob (%d bytes)", len(fallback_bytes))
        raw = bytes(fallback_bytes)

    if raw is None:
        logger.warning("Photo data not available (path=%s, blob=False)", file_path)
        return None

    # OpenAI/OpenRouter ignores EXIF orientation. Phone cameras store images
    # rotated with an orientation tag — physically transpose so the model
    # sees them upright.
    try:
        from io import BytesIO
        from PIL import Image, ImageOps
        with Image.open(BytesIO(raw)) as im:
            ori = im.getexif().get(274)
            transposed = ImageOps.exif_transpose(im)
            buf = BytesIO()
            transposed.convert("RGB").save(buf, format="JPEG", quality=90)
            data = buf.getvalue()
            if ori and ori != 1:
                logger.info("Photo EXIF orientation=%s — physically rotated before encoding", ori)
    except Exception as e:
        logger.warning("PIL orient/encode failed (%s) — falling back to raw bytes", e)
        data = raw
    b64 = base64.b64encode(data).decode("utf-8")
    logger.info("Encoded photo: %d bytes base64", len(b64))
    return b64


def _extract_damages(text: str) -> list:
    """Parse JSON damages from model output, tolerating truncation and extra text.

    Strategy:
      1. Try strict json.loads of the whole text.
      2. If that fails, extract the first balanced { ... } object and parse it.
      3. If that also fails (e.g. truncation), recover by parsing the inner array
         entry-by-entry up to the last complete object.
    """
    try:
        parsed = json.loads(text)
        return parsed.get("damages", parsed.get("danni", []))
    except json.JSONDecodeError:
        pass

    # Find balanced outer object
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:i + 1])
                            return parsed.get("damages", parsed.get("danni", []))
                        except json.JSONDecodeError:
                            break

    # Truncation recovery: extract individual {...} damage entries
    damages: list = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            entry = json.loads(m.group(0))
            if isinstance(entry, dict) and "damage_type" in entry:
                damages.append(entry)
        except json.JSONDecodeError:
            continue
    if damages:
        logger.warning("Recovered %d damages from malformed JSON via fallback parser", len(damages))
        return damages

    raise ValueError(f"Could not parse damages from response: {text[:200]}")


_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is an OpenAI reasoning model (o1/o3/o4 series).

    Handles provider-prefixed IDs like 'openai/o4-mini'.
    """
    name = model.rsplit("/", 1)[-1]
    return name.startswith(_REASONING_PREFIXES)


def _build_api_kwargs(model: str, content: list[dict]) -> dict:
    """Build OpenAI API kwargs based on model type."""
    api_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }

    if _is_reasoning_model(model):
        api_kwargs["max_completion_tokens"] = 8192
    else:
        api_kwargs["max_tokens"] = 8192
        api_kwargs["temperature"] = 0.2

    return api_kwargs


_VALID_DAMAGE_TYPES = {"graffio", "ammaccatura", "crepa", "rottura", "pezzo_mancante", "usura", "sporcizia"}
_VALID_SEVERITIES = {"lieve", "moderato", "grave"}
_VALID_ZONES = {"frontale", "laterale_sinistro", "posteriore", "laterale_destro", "superiore"}
# usura / sporcizia are reported only when severity is "grave".
_GRAVE_ONLY_TYPES = {"usura", "sporcizia"}


def _validate_damages(damages: list) -> list:
    """Drop entries that don't match the allowed enums or violate grave-only rules."""
    validated: list = []
    for d in damages:
        if (
            d.get("damage_type") in _VALID_DAMAGE_TYPES
            and d.get("severity") in _VALID_SEVERITIES
            and d.get("zone") in _VALID_ZONES
        ):
            if d["damage_type"] in _GRAVE_ONLY_TYPES and d["severity"] != "grave":
                logger.info("Dropping non-grave %s entry: %s", d["damage_type"], d)
                continue
            validated.append(d)
        else:
            logger.warning("Skipping invalid damage entry: %s", d)
    return validated


def _call_openai_single(client, model: str, photo: Photo, vehicle_type: str | None) -> tuple[list, str]:
    """Synchronous: run ONE OpenAI call for ONE photo. Returns (validated_damages, raw_text).

    Raises on transport/API failures; callers should catch and log per-photo.
    """
    b64 = _encode_image_base64(photo.file_path, getattr(photo, "image_data", None))
    if b64 is None:
        return [], ""

    prompt = _load_prompt(vehicle_type, photo.angle_label)
    label = ANGLE_LABELS.get(photo.angle_label, photo.angle_label)

    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"--- {label} ---"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]

    api_kwargs = _build_api_kwargs(model, content)
    logger.info(
        "OpenAI request (per-photo): model=%s, angle=%s, vehicle_type=%s",
        model, photo.angle_label, vehicle_type,
    )

    response = client.chat.completions.create(**api_kwargs)

    if not response.choices:
        logger.error(
            "API response has no choices (angle=%s). Response: %s",
            photo.angle_label, response.model_dump_json()[:1000],
        )
        raise RuntimeError(f"API returned no choices (model={model}, angle={photo.angle_label})")

    raw_text = response.choices[0].message.content or ""
    logger.info(
        "OpenAI raw response angle=%s (%d chars): %s",
        photo.angle_label, len(raw_text), raw_text[:500],
    )

    # Strip <think>...</think> blocks (Qwen3, DeepSeek, etc.)
    json_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines).strip()

    damages = _extract_damages(json_text)
    validated = _validate_damages(damages)
    logger.info(
        "Per-photo analysis angle=%s: %d damages validated out of %d returned",
        photo.angle_label, len(validated), len(damages),
    )
    return validated, raw_text


async def _call_openai(photos: list, vehicle_type: str | None = None) -> tuple[list, str]:
    """Call OpenAI Vision API once PER PHOTO, concurrently, and aggregate results.

    Returns (aggregated_validated_damages, concatenated_raw_text).
    Per-photo failures are logged and included as error markers in raw text but do not
    abort the whole session.
    """
    from openai import OpenAI

    model = settings.openai_model
    logger.info("Calling OpenAI model=%s with %d photos (one call per photo)", model, len(photos))

    if not photos:
        return [], ""

    kwargs = {"api_key": settings.openai_api_key, "timeout": 120.0}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    client = OpenAI(**kwargs)

    async def _run_one(photo: Photo) -> tuple[str, list, str, str | None]:
        """Wrap _call_openai_single so we can parallelize and capture per-photo errors."""
        try:
            damages, raw = await asyncio.to_thread(
                _call_openai_single, client, model, photo, vehicle_type
            )
            return photo.angle_label, damages, raw, None
        except Exception as exc:
            logger.exception(
                "Per-photo OpenAI call FAILED for angle=%s: %s", photo.angle_label, exc,
            )
            return photo.angle_label, [], "", str(exc)

    results = await asyncio.gather(*(_run_one(p) for p in photos))

    aggregated_damages: list = []
    raw_parts: list[str] = []
    for angle, damages, raw, error in results:
        header = f"=== {angle} ==="
        if error:
            raw_parts.append(f"{header}\n[ERROR] {error}")
            continue
        aggregated_damages.extend(damages)
        raw_parts.append(f"{header}\n{raw}" if raw else f"{header}\n[EMPTY_RESPONSE]")

    combined_raw = "\n\n".join(raw_parts)
    logger.info(
        "AI analysis aggregated: %d damages across %d photo-calls",
        len(aggregated_damages), len(photos),
    )
    return aggregated_damages, combined_raw


async def analyze_session(session_id: str) -> None:
    """Analyze all photos for a session using AI."""
    async with async_session() as db_session:
        # Create analysis result record
        analysis_id = str(uuid.uuid4())
        analysis = AnalysisResult(
            id=analysis_id,
            session_id=session_id,
            status="processing",
        )
        db_session.add(analysis)
        await db_session.commit()

        try:
            # Get photos for this session (skip invalid ones — e.g. no vehicle visible)
            result = await db_session.execute(
                select(Photo).where(Photo.session_id == session_id, Photo.is_valid == 1)
            )
            photos = result.scalars().all()

            if not photos:
                analysis.status = "completed"
                analysis.raw_response = json.dumps({"damages": []})
                await db_session.commit()
                return

            if not settings.openai_api_key:
                # No API key = error, not silent mock
                logger.error("OPENAI_API_KEY not configured — cannot analyze session %s", session_id)
                analysis.status = "error"
                analysis.raw_response = json.dumps({"error": "OPENAI_API_KEY not configured"})
                await db_session.commit()

                sess = await db_session.get(Session, session_id)
                if sess and sess.status == "uploaded":
                    sess.status = "completed"
                    await db_session.commit()
                return

            # Decrement remaining calls for the user
            sess = await db_session.get(Session, session_id)
            if sess:
                user = await db_session.get(User, sess.user_id)
                if user and user.remaining_calls is not None:
                    if user.remaining_calls <= 0:
                        analysis.status = "error"
                        analysis.raw_response = json.dumps({"error": "Chiamate esaurite"})
                        await db_session.commit()
                        return
                    user.remaining_calls -= 1
                    await db_session.commit()

            # Resolve vehicle type to pick the right prompt
            vehicle_type: str | None = None
            if sess:
                vehicle = await db_session.get(Vehicle, sess.vehicle_id)
                if vehicle:
                    vehicle_type = vehicle.type

            # Call OpenAI once per photo (concurrently)
            damage_list, raw_model_text = await _call_openai(photos, vehicle_type)

            # Save damages
            for damage_data in damage_list:
                damage = Damage(
                    id=str(uuid.uuid4()),
                    analysis_id=analysis_id,
                    damage_type=damage_data["damage_type"],
                    severity=damage_data["severity"],
                    zone=damage_data["zone"],
                    description=damage_data.get("description"),
                )
                db_session.add(damage)

            analysis.status = "completed"
            analysis.raw_response = raw_model_text

            # Update session status
            sess = await db_session.get(Session, session_id)
            if sess and sess.status == "uploaded":
                sess.status = "completed"

            await db_session.commit()
            logger.info("Analysis completed for session %s: %d damages", session_id, len(damage_list))

        except Exception as e:
            logger.exception("AI analysis FAILED for session %s: %s", session_id, e)
            analysis.status = "error"
            # Mask sensitive info (API keys, tokens) from error message
            error_msg = str(e)
            error_msg = re.sub(r'sk-[A-Za-z0-9_-]+', 'sk-***', error_msg)
            analysis.raw_response = json.dumps({"error": error_msg})
            await db_session.commit()
