import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid

from sqlalchemy import select, update

from app.config import settings
from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.models.photo import Photo
from app.models.session import Session
from app.models.user import User
from app.models.vehicle import Vehicle

logger = logging.getLogger(__name__)

# Pillow is load-bearing: every VLM call goes through EXIF orientation + JPEG
# re-encode, and the tiled detail pass crops/upscales with it. If it is missing
# the code silently falls back to raw (often sideways) bytes and recall collapses
# — so fail LOUD at import instead of degrading invisibly.
try:  # pragma: no cover - environment guard
    import PIL  # noqa: F401
except Exception:  # pragma: no cover
    logger.error(
        "Pillow (PIL) is NOT installed — photos will be sent WITHOUT EXIF rotation "
        "and the tiled detail pass is disabled. Detection quality will be severely "
        "degraded. Add 'Pillow>=10' to requirements.txt."
    )

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "prompts",
)

# Per-angle "OK"/integro reference images used for visual comparison.
# Only vehicle types present here send a reference alongside the inspection photo.
REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reference_images",
)
REFERENCE_SUBDIR_BY_VEHICLE_TYPE = {
    "scudo": "scudo",
}

# Longest side sent on the MAIN per-photo call (inspection photo + integro
# reference). Today's app uploads 720p so this is a no-op; it is a guard so a
# future higher-res capture path can't silently multiply main-pass image tokens.
_MAIN_IMAGE_MAX_SIDE = 1536

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


def _reference_image_path(vehicle_type: str | None, angle_label: str | None) -> str | None:
    """Return the path to the integro reference image for this vehicle type + angle, if any."""
    subdir = REFERENCE_SUBDIR_BY_VEHICLE_TYPE.get(vehicle_type or "")
    if not subdir or not angle_label:
        return None
    path = os.path.join(REFERENCE_DIR, subdir, f"{angle_label}.jpg")
    return path if os.path.exists(path) else None


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
    # sees them upright. Also cap resolution: the main pass only needs
    # context-level detail (the tile pass covers fine detail at zoom), so
    # anything beyond _MAIN_IMAGE_MAX_SIDE just buys extra Gemini crops.
    try:
        from io import BytesIO
        from PIL import Image, ImageOps
        with Image.open(BytesIO(raw)) as im:
            ori = im.getexif().get(274)
            transposed = _downscale(ImageOps.exif_transpose(im), _MAIN_IMAGE_MAX_SIDE)
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


def _damages_from_obj(obj) -> list:
    """Pull the damages array out of a parsed object, tolerating key variance."""
    if not isinstance(obj, dict):
        return []
    for key in ("damages", "danni"):
        v = obj.get(key)
        if isinstance(v, list):
            return v
    return []


def _balanced_objects(text: str, start: int = 0):
    """Yield every top-level balanced {...} substring from `text` (handles strings/escapes)."""
    depth = 0
    in_str = False
    esc = False
    obj_start = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    yield text[obj_start:i + 1]
                    obj_start = -1


def _parse_top_object(text: str):
    """Return the first parseable top-level JSON object (dict), or None.

    Used to recover the full response (damages + `checklist`) — not just damages.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for blob in _balanced_objects(text):
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _extract_damages(text: str) -> list:
    """Parse JSON damages from model output, tolerating truncation and extra text.

    Strategy:
      1. Strict json.loads of the whole text.
      2. First balanced {...} object that parses.
      3. Truncation recovery: brace-balanced scan for individual damage entries
         (handles nested objects, unlike the old single-level regex), accepting
         either an English ('damage_type') or Italian ('tipo'/'tipo_danno') key.
    """
    try:
        parsed = json.loads(text)
        dmgs = _damages_from_obj(parsed)
        if dmgs or (isinstance(parsed, dict) and ("damages" in parsed or "danni" in parsed)):
            return dmgs
    except json.JSONDecodeError:
        pass

    # First balanced outer object that parses cleanly.
    for blob in _balanced_objects(text):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            break  # truncated mid-object — fall through to entry recovery
        if isinstance(parsed, dict) and ("damages" in parsed or "danni" in parsed):
            return _damages_from_obj(parsed)

    # Truncation recovery: brace-balanced scan for individual damage entries.
    damages: list = []
    for blob in _balanced_objects(text):
        try:
            entry = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and any(k in entry for k in ("damage_type", "tipo", "tipo_danno")):
            damages.append(entry)
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


def _build_api_kwargs(model: str, content: list[dict], pass_index: int = 0) -> dict:
    """Build OpenAI API kwargs based on model type.

    `pass_index` decorrelates the independent main passes (recall-first union):
    pass 0 runs at temperature 0.2 (stable baseline); passes >=1 run hotter so the
    union surfaces findings a single deterministic read would miss. Reasoning
    models take no temperature, so the schedule only applies to the else branch.
    """
    api_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }

    if _is_reasoning_model(model):
        api_kwargs["max_completion_tokens"] = 8192
    else:
        api_kwargs["max_tokens"] = 8192
        api_kwargs["temperature"] = _PASS0_TEMP if pass_index <= 0 else _PASSN_TEMP

    return api_kwargs


# Per-pass temperature schedule for the recall-first union (see _build_api_kwargs).
_PASS0_TEMP = 0.2  # stable deterministic baseline read
_PASSN_TEMP = 0.5  # hotter on passes >=1 to decorrelate the union


# Appended to the prompt on passes >=1 so a second/third read of the SAME photo
# looks where a single pass loses recall: functional/safety components (broken or
# MISSING) and the low-contrast lower body. It must NOT make the model more
# trigger-happy — only re-direct attention — to keep precision flat.
_FOCUS_SUFFIX = (
    "\n\n=== SECONDA LETTURA (stessa foto) ===\n"
    "Ricontrolla con occhio fresco le zone che si perdono facilmente: i componenti "
    "funzionali/di sicurezza (fari, fanali, frecce, specchietti, vetri, ruote, targa, "
    "paraurti) per capire se sono rotti o MANCANTI, e la fascia bassa del veicolo "
    "(paraurti inferiore, modanature, passaruota, plastiche scure) dove i danni sono "
    "poco contrastati. NON essere più aggressivo: segnala SOLO danni reali e, nel "
    "dubbio, NON segnalare. Mantieni esattamente lo stesso formato JSON di risposta."
)


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_EXC_NAMES = {
    "APITimeoutError", "APIConnectionError", "RateLimitError",
    "InternalServerError", "APIError", "Timeout", "ServiceUnavailableError",
}


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True
    return type(exc).__name__ in _RETRYABLE_EXC_NAMES


def _chat_with_retry(client, *, max_attempts: int = 3, base_delay: float = 1.5, **kwargs):
    """client.chat.completions.create with exponential backoff on transient errors.

    The tiled detail pass fires ~6 calls per photo (24+ per scudo session); a
    single un-retried 429/5xx silently drops a tile and can lose a cracked-lens
    finding — the exact critical-damage recall this pipeline exists for.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # narrowed by _is_retryable
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "VLM call transient failure (attempt %d/%d): %s — retry in %.1fs",
                attempt, max_attempts, type(exc).__name__, delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _log_usage(resp, model: str, label: str) -> None:
    """Log token usage per call so spend is greppable (roadmap needs per-session
    cost tracking; this is the per-call building block, tiles included)."""
    usage = getattr(resp, "usage", None)
    if not usage:
        return
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    logger.info("VLM usage [%s] model=%s in=%d out=%d", label, model, pt, ct)


_VALID_DAMAGE_TYPES = {"graffio", "ammaccatura", "crepa", "rottura", "pezzo_mancante", "usura", "sporcizia"}
_VALID_SEVERITIES = {"lieve", "moderato", "grave"}
_VALID_ZONES = {"frontale", "laterale_sinistro", "posteriore", "laterale_destro", "superiore"}
# usura / sporcizia are reported only when severity is "grave".
_GRAVE_ONLY_TYPES = {"usura", "sporcizia"}


# Loose model phrasings -> canonical zone enum. The angle is known server-side,
# so zone must NEVER be a drop criterion (it was silently discarding valid
# pezzo_mancante findings whose zone the model phrased freely).
_ZONE_SYNONYMS = {
    "anteriore": "frontale", "frontale": "frontale", "fronte": "frontale",
    "davanti": "frontale", "muso": "frontale",
    "posteriore": "posteriore", "retro": "posteriore", "dietro": "posteriore",
    "coda": "posteriore",
    "laterale_destro": "laterale_destro", "laterale destra": "laterale_destro",
    "destro": "laterale_destro", "destra": "laterale_destro", "lato_destro": "laterale_destro",
    "laterale_sinistro": "laterale_sinistro", "laterale sinistra": "laterale_sinistro",
    "sinistro": "laterale_sinistro", "sinistra": "laterale_sinistro", "lato_sinistro": "laterale_sinistro",
    "superiore": "superiore", "tetto": "superiore",
}

# Functional / safety-relevant components: a missing or broken one is never
# "lieve" — enforce a severity floor so the costliest findings can't hide among
# cosmetic noise. Lighting/visibility components escalate to "grave".
_SAFETY_KW = (
    "faro", "fari", "fanale", "fanali", "fanal", "luce", "stop", "lente",
    "specchiett", "retrovisor", "vetro", "parabrezza", "lunotto", "ruota",
    "pneumatic", "gomma", "targa",
)
_FUNCTIONAL_KW = _SAFETY_KW + ("paraurti", "portellone", "porta", "portiera", "maniglia", "cofano")


def _normalize_zone(zone, angle_label: str | None) -> str:
    """Map a model-supplied zone to a canonical enum, defaulting from the known angle."""
    if zone in _VALID_ZONES:
        return zone
    if isinstance(zone, str):
        mapped = _ZONE_SYNONYMS.get(zone.strip().lower())
        if mapped:
            return mapped
    return ZONE_BY_ANGLE.get(angle_label or "", "frontale")


def _apply_severity_floor(d: dict) -> None:
    """Raise severity in-place for missing/broken functional components."""
    dt = d.get("damage_type")
    if dt not in ("pezzo_mancante", "rottura"):
        return
    text = ((d.get("description") or "") + " " + str(d.get("componente") or "")).lower()
    is_safety = any(k in text for k in _SAFETY_KW)
    floor = "grave" if is_safety else "moderato"
    if _SEV_RANK.get(d.get("severity"), 0) < _SEV_RANK[floor]:
        d["severity"] = floor


def _validate_damages(damages: list, angle_label: str | None = None) -> list:
    """Validate enums and grave-only rules. Zone is DEFAULTED from the known angle
    (never a drop criterion) so freely-phrased pezzo_mancante findings survive."""
    validated: list = []
    for d in damages:
        if d.get("damage_type") not in _VALID_DAMAGE_TYPES or d.get("severity") not in _VALID_SEVERITIES:
            logger.warning("Skipping invalid damage entry (type/severity): %s", d)
            continue
        if d["damage_type"] in _GRAVE_ONLY_TYPES and d["severity"] != "grave":
            logger.info("Dropping non-grave %s entry: %s", d["damage_type"], d)
            continue
        d["zone"] = _normalize_zone(d.get("zone"), angle_label)
        _apply_severity_floor(d)
        validated.append(d)
    return validated


def _damages_from_checklist(obj, angle_label: str | None) -> list:
    """Synthesize pezzo_mancante findings from the model's component `checklist`.

    The per-angle scudo prompts force the model to declare each component
    ok|danno|mancante|non_visibile|non_presente, but production code never read
    it — every `mancante` was silently discarded. This is the structural backstop
    for missing-part recall: turn each `mancante` into a pezzo_mancante damage.
    """
    if not isinstance(obj, dict):
        return []
    checklist = obj.get("checklist")
    if not isinstance(checklist, dict):
        return []
    zone = ZONE_BY_ANGLE.get(angle_label or "", "frontale")
    out: list = []
    for comp, status in checklist.items():
        if not isinstance(status, str) or status.strip().lower() != "mancante":
            continue
        comp_name = str(comp).replace("_", " ")
        is_safety = any(k in comp_name.lower() for k in _SAFETY_KW)
        out.append({
            "damage_type": "pezzo_mancante",
            "severity": "grave" if is_safety else "moderato",
            "zone": zone,
            "description": f"{comp_name} mancante (da checklist)",
            "confidence": 0.6,
            "needs_review": True,
        })
    if out:
        logger.info("Checklist backstop angle=%s: +%d pezzo_mancante from `mancante` statuses", angle_label, len(out))
    return out


# --- Stage-2 zoom verification (bbox-guided crop) -----------------------
ZONE_BY_ANGLE = {
    "fronte": "frontale", "lato_destro": "laterale_destro",
    "lato_sinistro": "laterale_sinistro", "retro": "posteriore",
}
# Components whose damage is resolution-sensitive -> flagged "da verificare".
_LIGHT_KW = ("faro", "fari", "fanale", "fanali", "fanal", "luce", "stop", "lente")

# Tile-pass guardrail: regions that are NOT the van. Unambiguous non-vehicle
# terms only — deliberately EXCLUDES livrea/scritta/logo/sporco, which legitimately
# appear as damage *landmarks* ("ammaccatura sotto la livrea gialla").
_BACKGROUND_KW = (
    "asfalt", "strada", "suolo", "terreno", "marciapied", "carreggiata", "parcheggio",
    "muro", "parete", "edificio", "palazzo", "capannone", "sfondo", "cielo", "nuvol",
    "erba", "prato", "vegetazion", "albero", "siepe",
    "altro veicolo", "altra auto", "altra vettura", "veicolo accanto", "auto accanto",
    "linea bianca", "strisce a terra", "segnaletica",
)

# Dedup stop-words. NOTE: side words (destro/sinistro/anteriore/posteriore) are
# deliberately NOT here — they discriminate symmetric L/R components and dropping
# them merged distinct findings. Generic surfaces are added so they never become
# the dedup key.
_NOUN_STOP = {
    "sulla", "della", "delle", "dello", "parte", "lato", "zona", "area",
    "inferiore", "superiore", "centrale", "furgone", "veicolo", "presenta",
    "visibile", "evidente", "componente", "carrozzeria", "pannello", "plastica",
    "fiancata", "porzione", "regione", "verificare", "danno", "danni",
    # relational words: "vicino al passaruota" and "sopra il passaruota" are the
    # same place for dedup purposes
    "vicino", "sopra", "sotto", "presso", "lungo", "accanto", "dietro",
    "davanti", "altezza", "associati", "associata", "corrispondenza",
}

# Damage-kind words add no information beyond the entry's damage_type (already
# the first element of _dedup_key) but their wording varies per pass
# ("graffi"/"rigature"/"abrasioni" for the same scratch) and used to break the
# noun-set equality. Prefix match to cover inflections (graffio/graffi/graffiata).
_DAMAGE_WORD_PREFIXES = (
    "ammacc", "graff", "rigat", "abras", "crep", "rott", "fratt",
    "scheggiat", "spacc", "deformaz", "segn",
)

# Synonyms different passes use for the SAME component, mapped to one canonical
# token (the union of 2 main + 2 tile passes re-describes each damage up to 4
# ways). Gender variants of side words collapse too — the L/R distinction
# survives because destro/destra and sinistro/sinistra map to different tokens.
_NOUN_CANON = {
    "parafango": "passaruota", "ruota": "passaruota", "ruote": "passaruota",
    "portiera": "porta", "portiere": "porta", "sportello": "porta",
    "destra": "destro", "sinistra": "sinistro",
}

# crepa and rottura describe the same physical break -> one dedup class so the
# same cracked lens reported as both is collapsed (keep the more severe label).
_TYPE_CLASS = {"crepa": "frattura", "rottura": "frattura"}

# Surface-damage pair the consolidate pass may merge ACROSS type: the same
# physical "ammaccatura con graffi" is reported as graffio by one pass and as
# ammaccatura by another, producing a phantom twin for nearly every dent.
_COSMETIC_TYPES = {"graffio", "ammaccatura"}

# Map a described component to its true zone, so a wrap-around tail light seen in
# a SIDE photo is recorded as 'posteriore', not 'laterale_*'.
_COMP_ZONE = (
    (("fanale", "fanali", "lunotto", "portellone", "terzo stop", "tergilunotto",
      "targa posteriore", "paraspruzzi", "catarifrangent"), "posteriore"),
    (("faro", "fari", "cofano", "griglia", "mascherina", "parabrezza", "fendinebbia",
      "presa aria", "targa anteriore"), "frontale"),
)
_SIDE_COMP_KW = ("portiera", "porta", "fiancata", "parafango", "passaruota",
                 "modanatura", "sottoporta", "pannello laterale", "specchiett", "maniglia")


def _zone_from_component(description: str, angle_label: str | None) -> str:
    """Infer zone from the named component; fall back to the photo angle."""
    t = (description or "").lower()
    for kws, z in _COMP_ZONE:
        if any(k in t for k in kws):
            return z
    if any(k in t for k in _SIDE_COMP_KW):
        if "sinistr" in t:
            return "laterale_sinistro"
        if "destr" in t:
            return "laterale_destro"
    return ZONE_BY_ANGLE.get(angle_label or "", "frontale")

# Deterministic detail pass: VLM bbox grounding proved unreliable on 720p van
# photos (boxes landed on blank panels), so instead we tile the photo into an
# overlapping grid, upscale each tile, and inspect it on its own. This reliably
# puts small damage (cracked light lens, broken lower bumper) in front of the
# model at usable scale.
# Reference-aware tile prompt. The detail pass now ships THREE images per tile:
# A = whole-vehicle context thumbnail, B = the SAME region on the integro
# reference (when available), C = the upscaled region to inspect. Diffing C
# against B on-the-van is what kills the 'asfalto con linea bianca' class of
# false positive AND makes a missing component (present in B, gone in C)
# detectable at usable resolution.
_TILE_PROMPT = (
    "Sei un ispettore di furgoni commerciali. Stai esaminando UNA REGIONE del {ang}. "
    "IMMAGINE A = contesto dell'intero veicolo (bassa risoluzione). "
    "IMMAGINE B = la STESSA regione su un esemplare INTEGRO di riferimento (può essere assente). "
    "IMMAGINE C = la regione INGRANDITA da ispezionare ORA. "
    "Segnala un danno SOLO se in C c'è un peggioramento REALE rispetto a B ED è SULLA CARROZZERIA DEL FURGONE. "
    "Se un componente è presente e integro in B ma in C è ASSENTE o frammentato => pezzo_mancante. "
    "Un BORDO FRASTAGLIATO/IRREGOLARE o una CAVITÀ SCURA con struttura interna esposta (traversa, "
    "assorbitore, staffe, lamiera, viti) dove B mostra un profilo LISCIO e CONTINUO (paraurti, spigolo "
    "del paraurti, plastica) => pezzo_mancante o rottura, NON 'graffio' né 'segni neri' (la plastica nera "
    "di fabbrica è liscia e uniforme; una rottura ha contorno netto e lascia vedere dentro). "
    "Cerca: fari/fanali con lente crepata/spaccata o frammento mancante; paraurti o plastica inferiore "
    "rotta/strappata/MANCANTE, spigolo/terminale del paraurti staccato; ammaccature evidenti; graffi/rigature profonde; vetri crepati; "
    "specchietti/maniglie rotti o mancanti. "
    "Restituisci [] se la regione è prevalentemente: asfalto/strada/suolo/marciapiede, cielo, vegetazione, "
    "ALTRI veicoli, muro/edificio di sfondo, ombre, riflessi, sporco/polvere, oppure solo livrea/scritte/loghi senza danno. "
    "NON segnalare difetti estetici minimi o dubbi. "
    "Rispondi SOLO JSON {{\"damages\":[{{\"damage_type\":\"graffio|ammaccatura|crepa|rottura|pezzo_mancante\","
    "\"severity\":\"lieve|moderato|grave\",\"componente\":\"<nome>\",\"description\":\"componente + cosa\"}}]}}. "
    "Se nulla: {{\"damages\":[]}}."
)
_TILE_OVERLAP = 0.20
# 2x upscale is COST-FREE on OpenRouter/gemini-2.5-flash: images are billed
# ~flat (~258 tok each, measured 2026-06-10 — tile calls cost ~1184 tok input
# at both 768px and 1194px tiles), and the A/B on a real dented van showed the
# 768px variant downgrading a clearly 'moderato' dent to 'lieve'. Keep 2x.
_TILE_UPSCALE = 2.0
_CONTEXT_THUMB_MAX = 512
# Concurrent tile calls per photo-run. Bounds PEAK MEMORY, not cost (each
# worker briefly holds 2 upscaled crops ~8MB each while encoding): 3 workers
# ~= 50MB/photo transient vs ~100MB with all 6 tiles in flight. Latency of a
# tile run is HTTP-dominated, so 2 batches of 3 ~ doubles tile wall-time only.
_TILE_WORKERS = 3  # longest side of the whole-frame context thumbnail


def _pil_from_source(file_path: str, fallback_bytes: bytes | None = None):
    """Open the inspection photo as an EXIF-corrected RGB PIL image (disk or DB blob)."""
    raw: bytes | None = None
    if file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f:
            raw = f.read()
    if raw is None and fallback_bytes:
        raw = bytes(fallback_bytes)
    if raw is None:
        return None
    try:
        from io import BytesIO
        from PIL import Image, ImageOps
        return ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
    except Exception as e:
        logger.warning("PIL open failed for zoom: %s", e)
        return None


def _pil_to_b64(im, quality: int = 90) -> str:
    import base64
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _salient_nouns(description: str) -> frozenset:
    """Discriminating words (>=4 chars), canonicalized. Side words are KEPT so
    symmetric L/R components ('fanale ... sinistro' vs '... destro') stay
    distinct; damage-kind words and relational filler are dropped, component
    synonyms collapse via _NOUN_CANON."""
    out = set()
    for w in re.findall(r"[a-zàèéìòù]{4,}", (description or "").lower()):
        if w in _NOUN_STOP or w.startswith(_DAMAGE_WORD_PREFIXES):
            continue
        out.add(_NOUN_CANON.get(w, w))
    return frozenset(out)


def _type_class(damage_type) -> str:
    return _TYPE_CLASS.get(damage_type, damage_type)


def _dedup_key(d: dict) -> tuple:
    """(type-class, zone, salient-noun-set). crepa/rottura collapse to one class;
    the noun set keeps left/right twins and distinct components apart. In a
    laterale_* zone the side is already encoded by the zone itself, so the side
    word is redundant there and only splits paraphrases ('parafango posteriore
    destro' vs 'passaruota posteriore' in the same lato_destro photo); for
    frontale/posteriore zones it stays — it is what separates L/R twins."""
    zone = d.get("zone")
    nouns = _salient_nouns(d.get("description", ""))
    if isinstance(zone, str) and zone.startswith("laterale"):
        nouns = nouns - {"destro", "sinistro"}
    return (_type_class(d.get("damage_type")), zone, nouns)


def _merge_damages(base: list, extra: list) -> list:
    """Append extra damages, dropping near-duplicates by _dedup_key."""
    out = list(base)
    seen = {_dedup_key(d) for d in out}
    for d in extra:
        k = _dedup_key(d)
        if k not in seen:
            out.append(d)
            seen.add(k)
    return out


_SEV_RANK = {"lieve": 1, "moderato": 2, "grave": 3}


def _merge_passes(pass_results: list[list[dict]], n_passes: int) -> list[dict]:
    """Union damages across N independent passes; dedup by _dedup_key.

    Recall-first: every distinct finding is KEPT even if it appeared in only one
    pass (critical damage is often found nondeterministically in just 1/N). The
    number of passes that agreed becomes a confidence signal (votes / N), and
    severity is escalated to the most severe reading across passes. Nothing is
    dropped — the agreement score lets a downstream reviewer/dashboard rank or
    filter without sacrificing recall.
    """
    agg: dict[tuple, dict] = {}
    for damages in pass_results:
        seen_this_pass: set = set()
        for d in damages:
            k = _dedup_key(d)
            if k in seen_this_pass:
                continue  # count a finding once per pass
            seen_this_pass.add(k)
            cur = agg.get(k)
            if cur is None:
                agg[k] = {**d, "_votes": 1}
            else:
                cur["_votes"] += 1
                # escalate to the most severe reading seen across passes
                if _SEV_RANK.get(d.get("severity"), 0) > _SEV_RANK.get(cur.get("severity"), 0):
                    cur["severity"] = d.get("severity")
                    cur["description"] = d.get("description", cur.get("description"))
                if d.get("needs_review"):
                    cur["needs_review"] = True

    out: list = []
    for d in agg.values():
        votes = d.pop("_votes")
        agree_conf = round(votes / n_passes, 2) if n_passes else 1.0
        base_conf = d.get("confidence")
        # Blend: never let multi-pass agreement lower a tile-pass confidence.
        d["confidence"] = round(max(agree_conf, base_conf or 0.0), 2)
        out.append(d)
    return out


_CONSOLIDATE_PROMPT = """Sei un perito assicurativo. Qui sotto c'è l'elenco dei danni rilevati da più letture indipendenti della STESSA foto (angolo: {ang}). Letture diverse descrivono spesso lo STESSO danno fisico con parole diverse (es. "parafango sopra il passaruota" = "carrozzeria sopra la ruota posteriore" = "fiancata posteriore").

Raggruppa SOLO le voci che sono chiaramente lo stesso danno fisico sullo stesso componente. NEL DUBBIO lascia separato. Non eliminare voci, non aggiungerne, non riscriverle.

DANNI:
{items}

Rispondi SOLO con JSON, senza testo extra:
{{"gruppi": [[0, 3], [1, 4]]}}
dove ogni lista contiene gli indici delle voci che sono lo stesso danno fisico. Gli indici non presenti in nessun gruppo restano voci separate."""

# Appended (verbatim, AFTER .format of the main prompt) only when the gridded
# photo could be rendered and attached. Asks for one extra JSON field — a map
# from input index to grid cell — and restates the no-add/no-drop contract.
_CONSOLIDATE_GRID_SUFFIX = """

IMMAGINE ALLEGATA: la STESSA foto con una griglia numerata sovraimpressa (celle 1-6). Per ogni voce dell'elenco indica anche in quale cella della griglia il danno è visibile, aggiungendo al JSON la mappa "celle" (chiave = indice della voce, valore = numero di cella 1-6, oppure null se non determinabile):
{"gruppi": [[0, 3]], "celle": {"0": 2, "3": 2, "1": 5}}
NON aggiungere, NON eliminare e NON inventare danni: limitati a raggruppare le voci dell'elenco dato e ad assegnare le celle."""


def _consolidate_damages(client, model, angle_label: str | None, damages: list,
                         photo=None) -> list:
    """Final LLM dedup of one photo's merged findings (1 cheap call).

    The union of 2 main + 2 tile passes re-describes the same physical damage
    with different wording, and _dedup_key's noun-set equality can't catch every
    paraphrase ("portiera posteriore" vs "porta scorrevole vicino al
    passaruota"). The model only returns GROUPS OF INDICES — the merge itself
    happens here in code, so it cannot drop, invent or rewrite findings.
    Guards: a group may only merge indices sharing (merge-class, zone), where
    graffio+ammaccatura form ONE cosmetic class (passes routinely read the same
    "ammaccatura con graffi" as either type, doubling every dent — the merged
    entry becomes ammaccatura, the more substantive reading) while structural
    classes (frattura, pezzo_mancante, ...) never merge across type. Any
    parse/API failure returns the input unchanged (best-effort, recall never
    loses).

    Localization piggyback (2026-06-12): when `photo` decodes, this SAME call
    also carries a downscaled copy of the photo with a numbered 1-6 grid drawn
    on it (image cost on OpenRouter is flat per call, so this is free) and the
    model returns, per INPUT INDEX, the grid cell where that damage is visible
    ("celle"). Cells only FILL bounding boxes that are still null after the
    merge — tile-pass boxes are ground truth and are never overwritten — by
    mapping cell N to _grid_boxes(original_size)[N-1], the same rect-string
    format the tile pass emits. Everything about the cells is best-effort: a
    failed overlay/encode, a missing or garbled "celle", or any exception in
    the mapping degrades to exactly the pre-grid behavior (consolidated list,
    boxes untouched)."""
    if len(damages) < 2:
        return damages
    # Render the numbered-grid copy of the photo (optional, never blocking).
    grid_b64 = None
    insp_size = None
    if photo is not None:
        try:
            insp_im = _pil_from_source(photo.file_path, getattr(photo, "image_data", None))
            if insp_im is not None:
                insp_size = insp_im.size
                grid_b64 = _grid_overlay_b64(insp_im)
        except Exception as e:
            logger.warning("consolidate grid overlay failed angle=%s: %s", angle_label, e)
    lines = "\n".join(
        f"{i}. {d.get('damage_type')} {d.get('severity')} [{d.get('zone')}]: {d.get('description')}"
        for i, d in enumerate(damages)
    )
    prompt = _CONSOLIDATE_PROMPT.format(
        ang=(angle_label or "").replace("_", " "), items=lines)
    content = [{"type": "text", "text": prompt}]
    if grid_b64:
        content[0]["text"] = prompt + _CONSOLIDATE_GRID_SUFFIX
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{grid_b64}"}})
    kwargs = _build_api_kwargs(model, content, pass_index=0)
    # Headroom for the "celle" map when the grid rides along; text-only path
    # keeps the historical 500 so behavior there is bit-identical.
    limit = 700 if grid_b64 else 500
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = limit
    else:
        kwargs["max_tokens"] = limit
        kwargs["temperature"] = 0.0  # pure grouping, determinism over recall
    try:
        resp = _chat_with_retry(client, **kwargs)
        _log_usage(resp, model, "consolidate")
        txt = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.DOTALL)
        txt = re.sub(r"```\w*", "", txt).replace("```", "").strip()
        parsed = json.loads(txt)
        groups = parsed.get("gruppi", [])
        # Only trust "celle" if we actually asked for it (grid attached) and it
        # is the documented index->cell map; anything else is ignored.
        celle = parsed.get("celle") if grid_b64 else None
        if not isinstance(celle, dict):
            celle = {}
    except Exception as e:
        logger.warning("consolidate pass failed angle=%s (%s) — keeping %d findings",
                       angle_label, e, len(damages))
        return damages

    merged_idx: set[int] = set()
    out: list = []
    out_members: list[list[int]] = []  # input indices behind each output entry
    for group in groups:
        if not isinstance(group, list):
            continue
        idx = [i for i in group if isinstance(i, int) and 0 <= i < len(damages) and i not in merged_idx]
        # Same-kind guard: partition the group by (merge-class, zone) so an
        # over-eager grouping can't fold a crepa into a graffio. graffio and
        # ammaccatura share the cosmetic class on purpose: the same physical
        # "ammaccatura con graffi" is read as either type by different passes.
        by_kind: dict[tuple, list[int]] = {}
        for i in idx:
            dt = damages[i].get("damage_type")
            cls = "cosmetico" if dt in _COSMETIC_TYPES else _type_class(dt)
            by_kind.setdefault((cls, damages[i].get("zone")), []).append(i)
        for members in by_kind.values():
            if len(members) < 2:
                continue
            ds = [damages[i] for i in members]
            best = max(ds, key=lambda d: (_SEV_RANK.get(d.get("severity"), 0),
                                          len(d.get("description") or "")))
            merged = dict(best)
            # Mixed cosmetic group = one dent with scratches: report it as the
            # more substantive ammaccatura, not as a separate graffio twin.
            if any(d.get("damage_type") == "ammaccatura" for d in ds):
                merged["damage_type"] = "ammaccatura"
            merged["confidence"] = round(max((d.get("confidence") or 0.0) for d in ds), 2)
            # A duplicate independently re-seen without the review flag is the
            # stronger reading: review only if EVERY sighting asked for it.
            merged["needs_review"] = all(bool(d.get("needs_review")) for d in ds)
            if not merged["needs_review"] and (merged.get("description") or "").startswith("[DA VERIFICARE] "):
                merged["description"] = merged["description"][len("[DA VERIFICARE] "):]
            merged["bounding_box"] = next((d.get("bounding_box") for d in ds if d.get("bounding_box")), None)
            out.append(merged)
            out_members.append(members)
            merged_idx.update(members)
    for i, d in enumerate(damages):
        if i not in merged_idx:
            out.append(d)
            out_members.append([i])
    # HARD GUARD: consolidation may NEVER yield more findings than it was given.
    # Structurally impossible (the model only returns indices into the input and
    # each index is consumed at most once), but enforced in code regardless.
    if len(out) > len(damages):
        logger.warning("consolidate angle=%s produced %d > %d entries — keeping originals",
                       angle_label, len(out), len(damages))
        return damages
    # Grid localization: fill ONLY null bounding boxes from the model's cell
    # assignment (any constituent index of a merged entry may supply the cell).
    # Tile-pass boxes are ground truth and are never overwritten. Defensive:
    # no exception here may alter detection results.
    if grid_b64 and insp_size and celle:
        try:
            for j, members in enumerate(out_members):
                if out[j].get("bounding_box"):
                    continue
                for i in members:
                    rect = _cell_rect_str(insp_size, celle.get(str(i), celle.get(i)))
                    if rect:
                        out[j] = {**out[j], "bounding_box": rect}
                        break
        except Exception as e:
            logger.warning("consolidate cell localization failed angle=%s: %s", angle_label, e)
    if len(out) < len(damages):
        logger.info("consolidate angle=%s: %d -> %d findings", angle_label, len(damages), len(out))
    return out


def _downscale(im, max_side: int):
    """Return a copy of `im` whose longest side is <= max_side (no upscaling)."""
    from PIL import Image
    w, h = im.size
    longest = max(w, h)
    if longest <= max_side:
        return im
    s = max_side / longest
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def _grid_boxes(size) -> list:
    """Overlapping grid boxes (wide=3x2, tall=2x3) as (x0, y0, x1, y1) tuples.

    Boxes only — the actual crop/upscale/encode happens lazily inside the tile
    worker (_encode_tile) so at most _TILE_WORKERS upscaled tiles exist at once.
    Materializing all 12 PIL tiles (6 boxes x inspection+reference, 2x upscaled)
    up-front held ~100MB per photo and OOM-killed the 512MB Render instance once
    the app started uploading 1080p photos."""
    W, H = size
    cols, rows = (3, 2) if W >= H else (2, 3)
    tw, th = W / cols, H / rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tw - _TILE_OVERLAP * tw)); y0 = max(0, int(r * th - _TILE_OVERLAP * th))
            x1 = min(W, int((c + 1) * tw + _TILE_OVERLAP * tw)); y1 = min(H, int((r + 1) * th + _TILE_OVERLAP * th))
            boxes.append((x0, y0, x1, y1))
    return boxes


def _cell_rect_str(size, cell) -> str | None:
    """Map a 1-based grid cell number onto the OVERLAPPING _grid_boxes rect of
    the original photo, as the same "x0,y0,x1,y1" pixel string the tile pass
    emits. Returns None for anything that is not a valid cell (bool, 0, 7,
    "x", None, ...) so garbage from the model can never produce a box."""
    if isinstance(cell, bool):
        return None
    try:
        c = int(cell)
    except (TypeError, ValueError):
        return None
    boxes = _grid_boxes(size)
    if not 1 <= c <= len(boxes):
        return None
    return ",".join(str(v) for v in boxes[c - 1])


def _mirror_cell_index(size, idx: int) -> int:
    """Mirror a 0-based ROW-MAJOR _grid_boxes cell index horizontally:
    same row, mirrored column (wide=3x2 -> 0<->2, 1 self; tall=2x3 -> 0<->1)."""
    W, H = size
    cols = 3 if W >= H else 2
    r, c = divmod(idx, cols)
    return r * cols + (cols - 1 - c)


def _mirror_rect_str(rect, size) -> str | None:
    """Mirror an 'x0,y0,x1,y1' pixel rect horizontally on an image of `size`:
    x -> W - x with x0/x1 swapped, clamped to the frame. Returns None for any
    unparseable or degenerate rect so garbage can never become a crop box."""
    W, H = size
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in str(rect).split(","))
    except (TypeError, ValueError):
        return None
    mx0, mx1 = max(0, W - x1), min(W, W - x0)
    my0, my1 = max(0, y0), min(H, y1)
    if mx1 <= mx0 or my1 <= my0:
        return None
    return f"{mx0},{my0},{mx1},{my1}"


def _grid_overlay_b64(insp_im) -> str | None:
    """Downscaled copy of the inspection photo with the NOMINAL (non-overlapping)
    grid partition drawn on it: cell border lines + big numbered labels 1..6
    (yellow with black outline) at cell centers.

    Same wide=3x2 / tall=2x3 orientation and ROW-MAJOR numbering as _grid_boxes,
    so "cella N" maps back to _grid_boxes(original_size)[N-1] via _cell_rect_str.
    The visual uses the non-overlapping partition because overlapping rectangles
    are unreadable as an overlay; the ~20% slack of the real rect absorbs the
    coarseness. Returns None on ANY failure — callers degrade to the plain
    text-only consolidate."""
    try:
        from PIL import ImageDraw, ImageFont
        im = _downscale(insp_im, _CONTEXT_THUMB_MAX).copy()
        W, H = im.size
        cols, rows = (3, 2) if W >= H else (2, 3)
        draw = ImageDraw.Draw(im)
        for c in range(1, cols):
            x = round(W * c / cols)
            draw.line([(x, 0), (x, H)], fill="black", width=5)
            draw.line([(x, 0), (x, H)], fill="#FFC400", width=2)
        for r in range(1, rows):
            y = round(H * r / rows)
            draw.line([(0, y), (W, y)], fill="black", width=5)
            draw.line([(0, y), (W, y)], fill="#FFC400", width=2)
        try:
            font = ImageFont.load_default(size=max(24, min(W, H) // 6))
        except Exception:  # very old Pillow: fixed-size bitmap fallback
            font = ImageFont.load_default()
        for n in range(rows * cols):
            r, c = divmod(n, cols)
            cx, cy = W * (2 * c + 1) / (2 * cols), H * (2 * r + 1) / (2 * rows)
            draw.text((cx, cy), str(n + 1), font=font, anchor="mm",
                      fill="#FFC400", stroke_width=4, stroke_fill="black")
        return _pil_to_b64(im, quality=70)
    except Exception as e:
        logger.warning("grid overlay render failed: %s", e)
        return None


def _encode_tile(insp, ref, box) -> tuple:
    """Crop + 2x-upscale + JPEG/base64-encode one grid cell; PIL tiles are local
    so they are freed before the HTTP call. The reference is cropped by the SAME
    relative box (it may differ in size/pose, so this is an approximate region
    match — enough for component-level diffing). Returns (tile_b64, ref_b64)."""
    from PIL import Image
    x0, y0, x1, y1 = box
    W, H = insp.size
    rW, rH = (ref.size if ref is not None else (W, H))
    sx, sy = (rW / W if W else 1.0), (rH / H if H else 1.0)
    tw_px, th_px = max(1, int((x1 - x0) * _TILE_UPSCALE)), max(1, int((y1 - y0) * _TILE_UPSCALE))
    tile_b64 = _pil_to_b64(insp.crop((x0, y0, x1, y1)).resize((tw_px, th_px), Image.LANCZOS))
    ref_b64 = None
    if ref is not None:
        rx0, ry0 = int(x0 * sx), int(y0 * sy)
        rx1, ry1 = min(rW, int(x1 * sx)), min(rH, int(y1 * sy))
        if rx1 > rx0 and ry1 > ry0:
            ref_b64 = _pil_to_b64(ref.crop((rx0, ry0, rx1, ry1)).resize((tw_px, th_px), Image.LANCZOS))
    return tile_b64, ref_b64


def _inspect_tile(client, model, ang_human, box_str, tile_b64, ref_b64, context_b64, pass_index=0) -> list:
    """Inspect one region with context + reference; return normalized damage dicts.

    Takes pre-encoded base64 JPEGs (not PIL images) so the heavy upscaled tiles
    are already freed by the time we sit in the HTTP call.
    `pass_index` carries the tile-run temperature schedule (run 0 cool, run >=1
    hotter) so repeated tile runs decorrelate and the union finds complementary
    damage instead of the same finding twice."""
    content = [{"type": "text", "text": _TILE_PROMPT.format(ang=ang_human)}]
    if context_b64:
        content += [
            {"type": "text", "text": "IMMAGINE A — CONTESTO (intero veicolo):"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{context_b64}"}},
        ]
    if ref_b64:
        content += [
            {"type": "text", "text": "IMMAGINE B — RIFERIMENTO INTEGRO di questa regione:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
        ]
    content += [
        {"type": "text", "text": "IMMAGINE C — REGIONE DA ISPEZIONARE (ingrandita):"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tile_b64}"}},
    ]
    kwargs = _build_api_kwargs(model, content, pass_index)
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = 700
    else:
        kwargs["max_tokens"] = 700
    resp = _chat_with_retry(client, **kwargs)
    _log_usage(resp, model, "tile")
    txt = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.DOTALL)
    txt = re.sub(r"```\w*", "", txt).replace("```", "").strip()
    try:
        items = json.loads(txt).get("damages", [])
    except Exception:
        items = _extract_damages(txt)
    return [{"box": box_str, **it} for it in items if isinstance(it, dict)]


def _is_background(text: str) -> bool:
    return any(k in text for k in _BACKGROUND_KW)


def _tiled_detail_pass(client, model, insp_im, ref_im, angle_label, pass_index=0) -> list:
    """Reference-aware grid detail pass over one photo (run TILE_PASSES times).

    Guardrails applied server-side (the prompt's IGNORA list is advisory only):
      - drop findings whose component/description names a non-vehicle region
        (the only true-FP drop — everything else is kept for recall);
      - lone cosmetic graffio/ammaccatura 'lieve' is KEPT but flagged
        needs_review at low confidence (recall-first; was dropped before);
      - keep ALL structural findings (crepa/rottura/pezzo_mancante) at any severity;
      - zone is inferred from the named component, not blindly from the angle;
      - fari/fanali findings stay flagged needs_review ([DA VERIFICARE])."""
    from concurrent.futures import ThreadPoolExecutor
    ang_human = (angle_label or "").replace("_", " ")
    context_b64 = None
    try:
        context_b64 = _pil_to_b64(_downscale(insp_im, _CONTEXT_THUMB_MAX), quality=70)
    except Exception as e:
        logger.warning("context thumbnail failed: %s", e)

    def _one(box) -> list:
        # Crop/encode inside the worker: at most _TILE_WORKERS upscaled tiles
        # alive at once (vs all 12 up-front — the 512MB-instance OOM).
        try:
            tile_b64, ref_b64 = _encode_tile(insp_im, ref_im, box)
        except Exception as e:
            logger.warning("tile encode failed box=%s: %s", box, e)
            return []
        box_str = ",".join(str(v) for v in box)
        return _safe_inspect_tile(client, model, ang_human, box_str, tile_b64, ref_b64, context_b64, pass_index)

    raw_items: list = []
    with ThreadPoolExecutor(max_workers=_TILE_WORKERS) as ex:
        for items in ex.map(_one, _grid_boxes(insp_im.size)):
            raw_items.extend(items)

    out: list = []
    for it in raw_items:
        dt, sev = it.get("damage_type"), it.get("severity")
        if dt not in _VALID_DAMAGE_TYPES or sev not in _VALID_SEVERITIES:
            continue
        if dt in _GRAVE_ONLY_TYPES and sev != "grave":
            continue
        comp = str(it.get("componente") or "")
        desc = (it.get("description") or comp)[:160]
        blob = (comp + " " + desc).lower()
        # Guardrail: non-vehicle region. This is the ONE thing we still drop —
        # it is a true false positive (asphalt/sky/other vehicle), not a miss.
        if _is_background(blob):
            logger.info("Tile FP dropped (background): %s", desc)
            continue
        # Recall-first (scudo "rileva tutto"): a lone lieve scratch/dent from a
        # tile used to be DROPPED as noise. We now KEEP it but flag it
        # needs_review at low confidence, so it surfaces as "da verificare" and is
        # triaged by the inspector instead of being silently lost.
        cosmetic_minor = dt in ("graffio", "ammaccatura") and sev == "lieve"
        is_light = any(k in blob for k in _LIGHT_KW)
        needs_review = is_light or cosmetic_minor
        if is_light and not desc.startswith("[DA VERIFICARE]"):
            desc = "[DA VERIFICARE] " + desc
        d = {
            "damage_type": dt, "severity": sev,
            "zone": _zone_from_component(blob, angle_label),
            "description": desc,
            "bounding_box": it.get("box"),
            "confidence": 0.4 if cosmetic_minor else 0.5,
            "needs_review": needs_review,
        }
        _apply_severity_floor(d)
        out.append(d)
    return out


def _safe_inspect_tile(client, model, ang_human, box_str, tile_b64, ref_b64, context_b64, pass_index=0) -> list:
    try:
        return _inspect_tile(client, model, ang_human, box_str, tile_b64, ref_b64, context_b64, pass_index)
    except Exception as e:
        logger.warning("tile inspect failed box=%s: %s", box_str, e)
        return []


def _run_tiled_for_photo(client, model, photo, vehicle_type) -> list:
    """Load inspection + integro-reference images and run the tiled detail pass ONCE.

    Gated on vehicle_type membership (not on a reference encoding succeeding), so a
    corrupt reference JPEG degrades to reference-less tiling instead of silently
    disabling the entire detail pass. Best-effort: returns [] on any failure.
    """
    if not REFERENCE_SUBDIR_BY_VEHICLE_TYPE.get(vehicle_type or ""):
        return []
    insp_im = _pil_from_source(photo.file_path, getattr(photo, "image_data", None))
    if insp_im is None:
        return []
    ref_im = None
    ref_path = _reference_image_path(vehicle_type, photo.angle_label)
    if ref_path:
        try:
            from PIL import Image, ImageOps
            with Image.open(ref_path) as r:
                ref_im = ImageOps.exif_transpose(r).convert("RGB")
        except Exception as e:
            logger.warning("Reference image open failed (%s) — tiling WITHOUT reference", e)
    else:
        logger.warning(
            "No integro reference for vehicle_type=%s angle=%s — tiling without reference",
            vehicle_type, photo.angle_label,
        )
    # Run the (nondeterministic) tile pass N times and UNION the results. One run
    # randomly misses real criticals (same photo gave 4/10/31 across runs), so the
    # union is the highest-leverage recall lever for scudo. Each run is independent;
    # a failed run contributes nothing instead of aborting the rest.
    n = max(1, settings.tile_passes)
    merged: list = []
    runs_ok = 0
    for run in range(n):
        try:
            found = _tiled_detail_pass(client, model, insp_im, ref_im, photo.angle_label, run)
            runs_ok += 1
        except Exception as e:
            logger.warning("Tiled detail run %d/%d failed angle=%s: %s", run + 1, n, photo.angle_label, e)
            continue
        before = len(merged)
        merged = _merge_damages(merged, found)
        logger.info(
            "Tiled run %d/%d angle=%s: %d found, +%d new (%d total)",
            run + 1, n, photo.angle_label, len(found), len(merged) - before, len(merged),
        )
    if n > 1:
        logger.info(
            "Tiled union angle=%s: %d unique damages across %d/%d runs",
            photo.angle_label, len(merged), runs_ok, n,
        )
    return merged


# --- Symmetric-light twin recheck ----------------------------------------
# Real incident (2026-06-12): BOTH tail lights shattered, checklist said "ok"
# for both, the tile pass caught only ONE side. Lights break in pairs far more
# often than independent passes detect them, so a one-sided light finding is a
# strong prior that the mirrored region deserves one extra look.


def _light_side(text: str) -> str | None:
    """Return 'sinistr'/'destr' when the text names EXACTLY one side, else None."""
    has_sx, has_dx = "sinistr" in text, "destr" in text
    if has_sx == has_dx:  # neither, or both ("fanali sinistro e destro")
        return None
    return "sinistr" if has_sx else "destr"


def _twin_recheck_target(damages: list) -> tuple[dict, str] | None:
    """Pick the damage that triggers the twin recheck, if any.

    Trigger: at least one light finding (_LIGHT_KW) naming exactly ONE side,
    AND no light finding covering the OPPOSITE side in the same list. Returns
    (triggering_damage, opposite_side_word). The trigger must carry a
    bounding_box (tile box or consolidate grid-cell fill) — without one there
    is no region to mirror, so we skip rather than guess.
    """
    by_side: dict[str, list] = {}
    for d in damages:
        text = ((d.get("description") or "") + " " + str(d.get("componente") or "")).lower()
        if not any(k in text for k in _LIGHT_KW):
            continue
        side = _light_side(text)
        if side:
            by_side.setdefault(side, []).append(d)
    if len(by_side) != 1:  # no one-sided light damage, or both sides covered
        return None
    side, candidates = next(iter(by_side.items()))
    trigger = next((d for d in candidates if d.get("bounding_box")), None)
    if trigger is None:
        logger.info("twin recheck fanale: trigger has no bounding_box — skipping")
        return None
    return trigger, ("destr" if side == "sinistr" else "sinistr")


def _twin_light_recheck(client, model, photo, vehicle_type, damages: list) -> list:
    """ONE extra mirrored tile inspection when exactly one side's light is damaged.

    Mirrors the triggering finding's region horizontally (grid-cell boxes map to
    the same-row mirrored-column cell; arbitrary rects mirror arithmetically)
    and runs a single _inspect_tile call on it, reference crop attached when the
    integro reference exists. Contract:
      - at most ONE extra call per photo, only when triggered;
      - only LIGHT findings from the recheck are kept, FORCED needs_review=True
        at confidence <= 0.4 with the mirrored rect as bounding_box;
      - existing findings are NEVER altered or removed; any failure returns the
        input list unchanged.
    Gated to reference vehicle types (scudo): the tile prompt is van-specific
    and scooters never run the tile pass — the simpler safe option.
    """
    try:
        if not REFERENCE_SUBDIR_BY_VEHICLE_TYPE.get(vehicle_type or ""):
            return damages
        # Mirroring x only makes sense where both lights are in frame: a
        # wrap-around tail light seen in a SIDE photo mirrors onto the other
        # END of the vehicle, not onto its twin.
        if photo.angle_label not in ("fronte", "retro"):
            return damages
        target = _twin_recheck_target(damages)
        if target is None:
            return damages
        trigger, _opp_side = target
        insp_im = _pil_from_source(photo.file_path, getattr(photo, "image_data", None))
        if insp_im is None:
            return damages

        # Grid-cell boxes mirror via the cell index (exact even when W/cols
        # truncation makes the arithmetic mirror off-by-one); any other rect
        # mirrors arithmetically with clamping.
        boxes = _grid_boxes(insp_im.size)
        box_strs = [",".join(str(v) for v in b) for b in boxes]
        if trigger.get("bounding_box") in box_strs:
            midx = _mirror_cell_index(insp_im.size, box_strs.index(trigger["bounding_box"]))
            mbox = boxes[midx]
            mirror_str = box_strs[midx]
        else:
            mirror_str = _mirror_rect_str(trigger.get("bounding_box"), insp_im.size)
            if mirror_str is None:
                return damages
            mbox = tuple(int(v) for v in mirror_str.split(","))
        logger.info(
            "twin recheck fanale angle=%s: '%s' names one side only — inspecting mirrored region %s",
            photo.angle_label, (trigger.get("description") or "")[:80], mirror_str,
        )

        ref_im = None
        ref_path = _reference_image_path(vehicle_type, photo.angle_label)
        if ref_path:
            try:
                from PIL import Image, ImageOps
                with Image.open(ref_path) as r:
                    ref_im = ImageOps.exif_transpose(r).convert("RGB")
            except Exception as e:
                logger.warning("twin recheck: reference open failed (%s) — inspecting without reference", e)
        context_b64 = None
        try:
            context_b64 = _pil_to_b64(_downscale(insp_im, _CONTEXT_THUMB_MAX), quality=70)
        except Exception as e:
            logger.warning("twin recheck: context thumbnail failed: %s", e)
        tile_b64, ref_b64 = _encode_tile(insp_im, ref_im, mbox)
        ang_human = (photo.angle_label or "").replace("_", " ")
        items = _inspect_tile(client, model, ang_human, mirror_str, tile_b64, ref_b64,
                              context_b64, pass_index=0)

        extra: list = []
        for it in items:
            dt, sev = it.get("damage_type"), it.get("severity")
            if dt not in _VALID_DAMAGE_TYPES or sev not in _VALID_SEVERITIES:
                continue
            if dt in _GRAVE_ONLY_TYPES and sev != "grave":
                continue
            comp = str(it.get("componente") or "")
            desc = (it.get("description") or comp)[:160]
            blob = (comp + " " + desc).lower()
            if not any(k in blob for k in _LIGHT_KW):
                continue  # the recheck may ONLY add light findings
            if _is_background(blob):
                continue
            if not desc.startswith("[DA VERIFICARE]"):
                desc = "[DA VERIFICARE] " + desc
            d = {
                "damage_type": dt, "severity": sev,
                "zone": _zone_from_component(blob, photo.angle_label),
                "description": desc,
                "bounding_box": mirror_str,
                "confidence": 0.4,
                "needs_review": True,
            }
            _apply_severity_floor(d)
            extra.append(d)
        if not extra:
            logger.info("twin recheck fanale angle=%s: mirrored region clean", photo.angle_label)
            return damages
        merged = _merge_damages(damages, extra)
        logger.info(
            "twin recheck fanale angle=%s: +%d light finding(s) on mirrored side",
            photo.angle_label, len(merged) - len(damages),
        )
        return merged
    except Exception as e:
        logger.warning(
            "twin recheck failed angle=%s: %s — keeping findings unchanged",
            getattr(photo, "angle_label", "?"), e,
        )
        return damages


def _call_openai_single(
    client, model: str, photo: Photo, vehicle_type: str | None, pass_index: int = 0
) -> tuple[list, str]:
    """Synchronous: run ONE OpenAI call for ONE photo. Returns (validated_damages, raw_text).

    `pass_index` selects the per-pass decorrelation (temperature + focus suffix)
    used by the recall-first multipass union. Raises on transport/API failures;
    callers should catch and log per-photo.
    """
    b64 = _encode_image_base64(photo.file_path, getattr(photo, "image_data", None))
    if b64 is None:
        return [], ""

    prompt = _load_prompt(vehicle_type, photo.angle_label)
    if pass_index >= 1:
        prompt = prompt + _FOCUS_SUFFIX
    label = ANGLE_LABELS.get(photo.angle_label, photo.angle_label)

    content: list[dict] = [{"type": "text", "text": prompt}]

    # If an integro reference exists for this vehicle/angle, send it FIRST as
    # IMMAGINE 1 so the model can diff the inspection photo (IMMAGINE 2) against it.
    ref_path = _reference_image_path(vehicle_type, photo.angle_label)
    if ref_path:
        ref_b64 = _encode_image_base64(ref_path)
        if ref_b64:
            content += [
                {"type": "text", "text": f"--- IMMAGINE 1: RIFERIMENTO INTEGRO ({label}) ---"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
                {"type": "text", "text": f"--- IMMAGINE 2: VEICOLO DA ISPEZIONARE ({label}) ---"},
            ]
            logger.info("Attached integro reference for vehicle_type=%s angle=%s", vehicle_type, photo.angle_label)
        else:
            content.append({"type": "text", "text": f"--- {label} ---"})
    else:
        content.append({"type": "text", "text": f"--- {label} ---"})

    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    api_kwargs = _build_api_kwargs(model, content, pass_index)
    logger.info(
        "OpenAI request (per-photo): model=%s, angle=%s, vehicle_type=%s, pass=%d",
        model, photo.angle_label, vehicle_type, pass_index,
    )

    response = _chat_with_retry(client, **api_kwargs)
    _log_usage(response, model, f"main:{photo.angle_label}:p{pass_index}")

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

    top_obj = _parse_top_object(json_text)
    damages = _extract_damages(json_text)
    validated = _validate_damages(damages, photo.angle_label)

    # Checklist backstop: convert `mancante` component statuses the model declared
    # (but did not duplicate into `damages`) into pezzo_mancante findings. This is
    # the structural recall fix for missing parts — previously the checklist was
    # emitted by the prompt and then discarded by every code path.
    checklist_dmgs = _damages_from_checklist(top_obj, photo.angle_label)
    if checklist_dmgs:
        validated = _merge_damages(validated, checklist_dmgs)

    logger.info(
        "Per-photo analysis angle=%s: %d damages validated out of %d returned (+%d from checklist)",
        photo.angle_label, len(validated), len(damages), len(checklist_dmgs),
    )

    # The deterministic tiled detail pass is NOT run here: it is run ONCE per
    # photo in _run_one (it is deterministic, so repeating it per VLM pass tripled
    # tile cost for no recall gain).
    return validated, raw_text


async def _call_openai(photos: list, vehicle_type: str | None = None) -> tuple[list, str]:
    """Call OpenAI Vision API once PER PHOTO, concurrently, and aggregate results.

    Returns (aggregated_validated_damages, concatenated_raw_text).
    Per-photo failures are logged and included as error markers in raw text but do not
    abort the whole session.
    """
    from openai import OpenAI

    model = settings.openai_model
    logger.info(
        "Calling OpenAI model=%s with %d photos (%d pass(es) per photo)",
        model, len(photos), max(1, settings.vlm_passes),
    )

    if not photos:
        return [], ""

    kwargs = {"api_key": settings.openai_api_key, "timeout": 120.0}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    client = OpenAI(**kwargs)

    passes = max(1, settings.vlm_passes)
    # Photos go through a semaphore instead of all-at-once: each in-flight photo
    # holds its PIL images + tile crops, and 4 concurrent 1080p photos OOM-killed
    # the 512MB Render instance. Memory scales with this, wall-time inversely.
    photo_sem = asyncio.Semaphore(max(1, settings.photo_concurrency))

    async def _run_one_inner(photo: Photo) -> tuple[str, list, str, str | None]:
        """Run [passes] independent VLM passes for one photo, concurrently, and
        merge them union-wise (recall-first). A photo is only marked failed if
        EVERY pass failed — a partial failure still yields the surviving passes."""
        outcomes = await asyncio.gather(
            *(
                asyncio.to_thread(_call_openai_single, client, model, photo, vehicle_type, i)
                for i in range(passes)
            ),
            return_exceptions=True,
        )
        ok = [o for o in outcomes if not isinstance(o, BaseException)]
        errs = [o for o in outcomes if isinstance(o, BaseException)]

        if not ok:
            exc = errs[0] if errs else RuntimeError("all passes returned nothing")
            logger.exception(
                "All %d passes FAILED for angle=%s: %s", passes, photo.angle_label, exc,
            )
            return photo.angle_label, [], "", str(exc)

        if errs:
            logger.warning(
                "angle=%s: %d/%d passes failed, merging the %d that succeeded",
                photo.angle_label, len(errs), passes, len(ok),
            )

        pass_damages = [dmg for dmg, _ in ok]
        merged = _merge_passes(pass_damages, passes) if passes > 1 else list(pass_damages[0])

        # Deterministic reference-aware tiled detail pass — run ONCE per photo
        # (not once per VLM pass), then union into the multipass-merged findings.
        tile_extra = await asyncio.to_thread(_run_tiled_for_photo, client, model, photo, vehicle_type)
        if tile_extra:
            before = len(merged)
            merged = _merge_damages(merged, tile_extra)
            logger.info(
                "angle=%s tiled: +%d detail damages (%d unique after merge)",
                photo.angle_label, len(tile_extra), len(merged) - before,
            )

        # Final paraphrase-dedup + localization: the union above is recall-first
        # and re-lists the same physical damage under different wording. One call
        # per photo (skipped under 2 findings) groups true duplicates and — via
        # a numbered-grid copy of the photo riding on that same call — fills the
        # bounding boxes the full-photo passes leave null (tile boxes win).
        if len(merged) > 1:
            merged = await asyncio.to_thread(
                _consolidate_damages, client, model, photo.angle_label, merged, photo
            )

        # Symmetric-light twin recheck: lights shatter in PAIRS more often than
        # the passes detect them (real case: both tail lights gone, one found).
        # When exactly one side's light is flagged, ONE extra mirrored tile call
        # probes the opposite light. Add-only, needs_review-forced, best-effort.
        if merged:
            merged = await asyncio.to_thread(
                _twin_light_recheck, client, model, photo, vehicle_type, merged
            )

        raw = f"[multipass {len(ok)}/{passes}]\n" + (ok[0][1] or "")
        logger.info(
            "angle=%s multipass: %d unique damages from %d/%d passes",
            photo.angle_label, len(merged), len(ok), passes,
        )
        return photo.angle_label, merged, raw, None

    async def _run_one(photo: Photo) -> tuple[str, list, str, str | None]:
        async with photo_sem:
            return await _run_one_inner(photo)

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

    # If EVERY photo failed (e.g. revoked API key -> 401 on all calls) this is
    # an infrastructure failure, not a clean vehicle: raise so analyze_session
    # stores status="error" and the app shows the failure + retry instead of a
    # false "nessun danno rilevato".
    if photos and all(error for _, _, _, error in results):
        raise RuntimeError(
            f"all {len(photos)} photo analyses failed:\n{combined_raw}"
        )

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

            # Decrement remaining calls for the user. Atomic UPDATE so two
            # concurrent analyses for the same user can't both pass the check
            # and over-spend the quota. remaining_calls IS NULL == unlimited.
            sess = await db_session.get(Session, session_id)
            if sess:
                upd = await db_session.execute(
                    update(User)
                    .where(
                        User.id == sess.user_id,
                        User.remaining_calls.isnot(None),
                        User.remaining_calls > 0,
                    )
                    .values(remaining_calls=User.remaining_calls - 1)
                )
                await db_session.commit()
                if upd.rowcount == 0:
                    # No row decremented: either unlimited (NULL) or exhausted.
                    user = await db_session.get(User, sess.user_id)
                    if user and user.remaining_calls is not None and user.remaining_calls <= 0:
                        analysis.status = "error"
                        analysis.raw_response = json.dumps({"error": "Chiamate esaurite"})
                        await db_session.commit()
                        return

            # Resolve vehicle type to pick the right prompt
            vehicle_type: str | None = None
            if sess:
                vehicle = await db_session.get(Vehicle, sess.vehicle_id)
                if vehicle:
                    vehicle_type = vehicle.type

            # Call OpenAI once per photo (concurrently)
            damage_list, raw_model_text = await _call_openai(photos, vehicle_type)

            # Save damages. Use .get for the required fields and skip malformed
            # entries so one bad merge/tile dict can't 500 the whole session.
            for damage_data in damage_list:
                if not (damage_data.get("damage_type") and damage_data.get("severity") and damage_data.get("zone")):
                    logger.warning("Skipping malformed damage at persist: %s", damage_data)
                    continue
                damage = Damage(
                    id=str(uuid.uuid4()),
                    analysis_id=analysis_id,
                    damage_type=damage_data["damage_type"],
                    severity=damage_data["severity"],
                    zone=damage_data["zone"],
                    description=damage_data.get("description"),
                    bounding_box=damage_data.get("bounding_box"),
                    confidence=damage_data.get("confidence"),
                    needs_review=1 if damage_data.get("needs_review") else 0,
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


# Strong refs to in-flight analysis tasks. asyncio keeps only a weak reference
# to bare create_task() results, so without this set a task can be garbage-
# collected mid-run and vanish silently.
_background_tasks: set = set()


def spawn_analysis(session_id: str) -> None:
    """Fire-and-forget analyze_session, holding a strong ref until it finishes."""
    task = asyncio.create_task(analyze_session(session_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def recover_pending_analyses() -> None:
    """Re-trigger analyses orphaned by a process restart.

    Render's free tier kills in-flight background tasks on cold restart, leaving
    an AnalysisResult stuck in 'processing' (or an 'uploaded' session with no
    analysis at all) forever. Photos survive in the DB blob, so re-running is
    safe. Called once on startup.
    """
    try:
        async with async_session() as db_session:
            proc = await db_session.execute(
                select(AnalysisResult.session_id).where(AnalysisResult.status == "processing")
            )
            session_ids = list(dict.fromkeys(r[0] for r in proc.all()))
            uploaded = await db_session.execute(
                select(Session.id).where(Session.status == "uploaded")
            )
            for sid in uploaded.scalars().all():
                if sid not in session_ids:
                    session_ids.append(sid)
    except Exception as e:
        logger.warning("recover_pending_analyses: query failed: %s", e)
        return

    if not session_ids:
        return
    logger.info("Recovering %d stuck analysis session(s): %s", len(session_ids), session_ids)
    for sid in session_ids:
        # Drop the stale 'processing' row so analyze_session starts a fresh one.
        try:
            async with async_session() as db_session:
                stale = await db_session.execute(
                    select(AnalysisResult).where(
                        AnalysisResult.session_id == sid,
                        AnalysisResult.status == "processing",
                    )
                )
                for a in stale.scalars().all():
                    await db_session.delete(a)
                await db_session.commit()
        except Exception as e:
            logger.warning("recover: cleanup failed for %s: %s", sid, e)
        spawn_analysis(sid)
