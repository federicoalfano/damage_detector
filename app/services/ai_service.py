import base64
import json
import logging
import os
import uuid

from sqlalchemy import select

from app.config import settings
from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.models.photo import Photo
from app.models.session import Session

logger = logging.getLogger(__name__)

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "prompts",
    "damage_analysis.txt",
)


def _load_prompt() -> str:
    with open(PROMPT_PATH) as f:
        return f.read()


def _encode_image_base64(file_path: str) -> str | None:
    """Read an image file and return its base64 encoding."""
    if not os.path.exists(file_path):
        logger.warning("Photo file not found: %s", file_path)
        return None
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def _call_openai(photos: list) -> list:
    """Call OpenAI Vision API with all session photos."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = _load_prompt()

    # Build content array with all photos
    content: list[dict] = [{"type": "text", "text": prompt}]

    for photo in photos:
        b64 = _encode_image_base64(photo.file_path)
        if b64 is None:
            continue
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            },
        })

    if len(content) <= 1:
        # No valid photos to analyze
        return []

    # o4-mini uses max_completion_tokens instead of max_tokens,
    # and does not support the temperature parameter
    api_kwargs: dict = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    if settings.openai_model.startswith("o"):
        api_kwargs["max_completion_tokens"] = 2048
    else:
        api_kwargs["max_tokens"] = 1024
        api_kwargs["temperature"] = 0.1

    response = client.chat.completions.create(**api_kwargs)

    raw_text = response.choices[0].message.content or ""
    logger.info("OpenAI raw response: %s", raw_text)

    # Parse JSON from response (handle markdown code blocks)
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        # Remove ```json ... ``` wrapper
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines)

    parsed = json.loads(json_text)
    damages = parsed.get("damages", [])

    # Validate structure
    valid_types = {"graffio", "ammaccatura", "crepa", "rottura", "pezzo_mancante"}
    valid_severities = {"lieve", "moderato", "grave"}
    valid_zones = {"frontale", "laterale_sinistro", "posteriore", "laterale_destro", "superiore"}

    validated = []
    for d in damages:
        if (
            d.get("damage_type") in valid_types
            and d.get("severity") in valid_severities
            and d.get("zone") in valid_zones
        ):
            validated.append(d)
        else:
            logger.warning("Skipping invalid damage entry: %s", d)

    return validated


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
            # Get photos for this session
            result = await db_session.execute(
                select(Photo).where(Photo.session_id == session_id)
            )
            photos = result.scalars().all()

            if not photos:
                analysis.status = "completed"
                analysis.raw_response = json.dumps({"damages": []})
                await db_session.commit()
                return

            # Use real OpenAI if API key is configured, otherwise mock
            if settings.openai_api_key:
                try:
                    damage_list = await _call_openai(photos)
                except Exception as e:
                    logger.exception("OpenAI call failed, falling back to mock: %s", e)
                    damage_list = _mock_analysis(photos)
            else:
                logger.info("No OpenAI API key configured, using mock analysis")
                damage_list = _mock_analysis(photos)

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
            analysis.raw_response = json.dumps({"damages": damage_list})

            # Update session status so the list shows "Completata"
            sess = await db_session.get(Session, session_id)
            if sess and sess.status == "uploaded":
                sess.status = "completed"

            await db_session.commit()

        except Exception as e:
            logger.exception("AI analysis failed for session %s", session_id)
            analysis.status = "error"
            await db_session.commit()


def _mock_analysis(photos: list) -> list:
    """MVP mock: simulate AI damage detection results."""
    damages = []

    # Simulate finding a graffio on the front
    if any(p.angle_label == "fronte" for p in photos):
        damages.append({
            "damage_type": "graffio",
            "severity": "lieve",
            "zone": "frontale",
            "description": "Graffio superficiale sulla carenatura frontale",
        })

    # Simulate finding an ammaccatura on the left
    if any(p.angle_label == "lato_sinistro" for p in photos):
        damages.append({
            "damage_type": "ammaccatura",
            "severity": "moderato",
            "zone": "laterale_sinistro",
            "description": "Ammaccatura sulla fiancata sinistra",
        })

    return damages
