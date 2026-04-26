"""Validate uploaded photos against expected vehicle type using GPT-4o-mini."""
import base64
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Descriptions used in the validation prompt
VEHICLE_DESCRIPTIONS: dict[str, str] = {
    "piaggio": (
        "Piaggio Liberty: uno scooter elettrico a due ruote. "
        "Design classico con ruote alte, sella comoda e vano sottosella, "
        "utilizzato per mobilità urbana."
    ),
    "ligier": (
        "Ligier: un quadriciclo leggero elettrico, simile a una microcar. "
        "Design compatto con abitacolo chiuso, 4 ruote, utilizzato per "
        "mobilità urbana."
    ),
    "my_moover": (
        "My Moover: un veicolo elettrico compatto per consegne urbane. "
        "Design moderno, utilizzato per la logistica dell'ultimo miglio."
    ),
    "scudo": (
        "Fiat Scudo: un furgone commerciale di medie dimensioni. "
        "Design da veicolo commerciale con vano di carico posteriore, "
        "utilizzato per trasporto merci e consegne."
    ),
}

VALIDATION_PROMPT = """\
Sei un sistema di controllo qualità fotografica per ispezione flotta.

Categoria veicolo atteso: {category}
Angolazione dichiarata della foto: {angle}

Analizza l'immagine e rispondi SOLO con un oggetto JSON (nessun testo aggiuntivo):
{{
  "valid": true/false,
  "reason": "breve spiegazione in italiano"
}}

Regole (sii PERMISSIVO):
- "valid": true se l'immagine mostra il tipo di veicolo giusto (es. un furgone) DA QUALSIASI MARCA/MODELLO e l'angolazione corrisponde a quella dichiarata
- NON rifiutare perché il modello specifico non combacia (es. uno Scudo può sembrare un Peugeot Expert - è ok, condividono piattaforma)
- L'angolazione deve corrispondere:
  - "fronte": si vede chiaramente il muso anteriore del veicolo
  - "retro": si vede chiaramente la parte posteriore / portellone
  - "lato_destro": si vede il fianco destro del veicolo (porte/fiancata lato destro, orientato verso la nostra destra)
  - "lato_sinistro": si vede il fianco sinistro del veicolo (porte/fiancata lato sinistro, orientato verso la nostra sinistra)
- "valid": false SOLO SE:
  - Non c'è alcun veicolo nella foto (solo asfalto, persone, oggetti non veicolari)
  - Il tipo di veicolo è completamente sbagliato (es. atteso furgone, c'è un aereo o una moto)
  - L'angolazione è chiaramente sbagliata (dichiarato "fronte" ma la foto mostra il retro)
  - La foto è troppo sfocata/scura/inquadrata male per capire cosa mostra

Esempi:
- Foto di scarpe su asfalto → valid: false, reason: "Nessun veicolo visibile"
- Foto di un Citroën Berlingo dichiarato Scudo → valid: true, reason: "Furgone commerciale, angolazione corretta"
- Foto del retro dichiarata come "lato_sinistro" → valid: false, reason: "Angolazione non corrisponde: foto mostra il retro"
"""

# Categoria per ogni vehicle type (più permissiva del modello specifico)
VEHICLE_CATEGORIES: dict[str, str] = {
    "piaggio": "scooter / motociclo a 3 ruote (Piaggio Liberty o simile)",
    "ligier": "quadriciclo leggero / microcar a 4 ruote (Ligier o simile)",
    "my_moover": "veicolo elettrico compatto urbano per consegne",
    "scudo": "furgone commerciale di medie dimensioni",
}

# Italian angle labels
ANGLE_IT = {
    "fronte": "fronte (vista anteriore)",
    "lato_destro": "lato destro (fiancata destra)",
    "lato_sinistro": "lato sinistro (fiancata sinistra)",
    "retro": "retro (vista posteriore)",
}


async def validate_photo(file_path: str, vehicle_type: str, angle_label: str = "") -> dict:
    """Validate a photo against the expected vehicle category + angle.

    Returns {"valid": True/False, "reason": "..."}.
    If no API key is configured or vehicle type unknown, skips validation.
    """
    category = VEHICLE_CATEGORIES.get(vehicle_type)
    if not category:
        logger.info("No category for vehicle type '%s', skipping validation", vehicle_type)
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

        prompt = VALIDATION_PROMPT.format(
            category=category,
            angle=ANGLE_IT.get(angle_label, angle_label or "qualsiasi"),
        )

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
