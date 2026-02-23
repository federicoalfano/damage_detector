"""Validate uploaded photos against expected vehicle type using GPT-4o-mini."""
import base64
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Descriptions used in the validation prompt
VEHICLE_DESCRIPTIONS: dict[str, str] = {
    "pulse": (
        "Piaggio Ape Pulse 3: uno scooter elettrico a 3 ruote con un vano cargo "
        "posteriore (cassone/box). Ha due ruote dietro e una davanti, design compatto "
        "e utilitario per consegne urbane."
    ),
    "hurba": (
        "Hurba: uno scooter elettrico a due ruote in stile maxi-scooter (simile al "
        "Yamaha T-Max). Design sportivo con carenatura aerodinamica, sella lunga, "
        "e ruote grandi."
    ),
}

VALIDATION_PROMPT = """\
Sei un sistema di controllo qualità fotografica per una flotta di veicoli.

Il veicolo atteso in questa foto è: {vehicle_description}

Analizza l'immagine e rispondi SOLO con un oggetto JSON (nessun testo aggiuntivo):
{{
  "valid": true/false,
  "reason": "breve spiegazione in italiano"
}}

Regole:
- "valid": true se nell'immagine è chiaramente visibile il tipo di veicolo descritto (anche parzialmente, da qualsiasi angolazione)
- "valid": false se l'immagine mostra un veicolo completamente diverso, nessun veicolo, o è troppo sfocata/scura per identificare il soggetto
- Sii ragionevole: non serve una corrispondenza perfetta, basta che sia riconoscibile come quel tipo di veicolo
"""


async def validate_photo(file_path: str, vehicle_type: str) -> dict:
    """Validate a photo against the expected vehicle type.

    Returns {"valid": True/False, "reason": "..."}.
    If no API key is configured or vehicle type unknown, skips validation.
    """
    description = VEHICLE_DESCRIPTIONS.get(vehicle_type)
    if not description:
        logger.info("No description for vehicle type '%s', skipping validation", vehicle_type)
        return {"valid": True, "reason": "Tipo veicolo senza validazione"}

    if not settings.openai_api_key:
        logger.info("No OpenAI API key, skipping photo validation")
        return {"valid": True, "reason": "Validazione disabilitata (no API key)"}

    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        from openai import OpenAI
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        client = OpenAI(**kwargs)

        prompt = VALIDATION_PROMPT.format(vehicle_description=description)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            max_tokens=150,
            temperature=0.1,
        )

        raw = response.choices[0].message.content or ""
        logger.info("Photo validation raw response: %s", raw)

        # Parse JSON (handle markdown code blocks)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        result = json.loads(text)
        return {
            "valid": bool(result.get("valid", True)),
            "reason": result.get("reason", ""),
        }

    except Exception as e:
        logger.exception("Photo validation failed: %s", e)
        # On error, don't block the upload
        return {"valid": True, "reason": f"Errore validazione: {e}"}
