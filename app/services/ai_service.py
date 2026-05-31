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


# --- Stage-2 zoom verification (bbox-guided crop) -----------------------
ZONE_BY_ANGLE = {
    "fronte": "frontale", "lato_destro": "laterale_destro",
    "lato_sinistro": "laterale_sinistro", "retro": "posteriore",
}
# Components whose damage is resolution-sensitive -> flagged "da verificare".
_LIGHT_KW = ("faro", "fari", "fanale", "fanali", "fanal", "luce", "stop", "lente")
_NOUN_STOP = {
    "sulla", "della", "parte", "lato", "destro", "sinistro", "anteriore", "posteriore",
    "inferiore", "superiore", "furgone", "veicolo", "presenta", "visibile", "componente",
}

# Deterministic detail pass: VLM bbox grounding proved unreliable on 720p van
# photos (boxes landed on blank panels), so instead we tile the photo into an
# overlapping grid, upscale each tile, and inspect it on its own. This reliably
# puts small damage (cracked light lens, broken lower bumper) in front of the
# model at usable scale.
_TILE_PROMPT = (
    "Vedi una PORZIONE INGRANDITA del {ang} di un furgone commerciale. Ispeziona SOLO ciò che è visibile "
    "in questa porzione. Cerca danni STRUTTURALI: fari/fanali con lente crepata/spaccata o frammento "
    "mancante; paraurti o plastica inferiore rotta/strappata/MANCANTE; ammaccature evidenti; graffi/rigature "
    "profonde; vetri crepati; specchietti/maniglie rotti o mancanti. "
    "IGNORA: sfondo, altri veicoli, ombre, sporco/polvere, riflessi di luce, livrea/scritte/loghi. "
    "NON segnalare danni dubbi o difetti estetici minimi. "
    "Rispondi SOLO JSON {{\"damages\":[{{\"damage_type\":\"graffio|ammaccatura|crepa|rottura|pezzo_mancante\","
    "\"severity\":\"lieve|moderato|grave\",\"componente\":\"<nome>\",\"description\":\"componente + cosa\"}}]}}. "
    "Se nulla: {{\"damages\":[]}}."
)
_TILE_OVERLAP = 0.20
_TILE_UPSCALE = 2.0


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


def _noun_key(description: str) -> str:
    for w in re.findall(r"[a-zàèéìòù]{4,}", (description or "").lower()):
        if w not in _NOUN_STOP:
            return w
    return ""


def _merge_damages(base: list, extra: list) -> list:
    """Append extra damages, dropping near-duplicates by (type, zone, first noun)."""
    out = list(base)
    seen = {(d.get("damage_type"), d.get("zone"), _noun_key(d.get("description", ""))) for d in out}
    for d in extra:
        k = (d.get("damage_type"), d.get("zone"), _noun_key(d.get("description", "")))
        if k not in seen:
            out.append(d)
            seen.add(k)
    return out


_SEV_RANK = {"lieve": 1, "moderato": 2, "grave": 3}


def _merge_passes(pass_results: list[list[dict]], n_passes: int) -> list[dict]:
    """Union damages across N independent passes; dedup by (type, zone, noun).

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
            k = (d.get("damage_type"), d.get("zone"), _noun_key(d.get("description", "")))
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

    out: list = []
    for d in agg.values():
        votes = d.pop("_votes")
        agree_conf = round(votes / n_passes, 2) if n_passes else 1.0
        base_conf = d.get("confidence")
        # Blend: never let multi-pass agreement lower a tile-pass confidence.
        d["confidence"] = round(max(agree_conf, base_conf or 0.0), 2)
        out.append(d)
    return out


def _make_grid(im) -> list[tuple[str, object]]:
    """Split the image into an overlapping grid (wide=3x2, tall=2x3), each tile
    upscaled. Returns [(pixel_box_str, PIL_tile), ...]."""
    from PIL import Image
    W, H = im.size
    cols, rows = (3, 2) if W >= H else (2, 3)
    tw, th = W / cols, H / rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tw - _TILE_OVERLAP * tw)); y0 = max(0, int(r * th - _TILE_OVERLAP * th))
            x1 = min(W, int((c + 1) * tw + _TILE_OVERLAP * tw)); y1 = min(H, int((r + 1) * th + _TILE_OVERLAP * th))
            t = im.crop((x0, y0, x1, y1))
            t = t.resize((int(t.width * _TILE_UPSCALE), int(t.height * _TILE_UPSCALE)), Image.LANCZOS)
            tiles.append((f"{x0},{y0},{x1},{y1}", t))
    return tiles


def _inspect_tile(client, model, ang_human, box_str, tile) -> list:
    """Inspect one upscaled tile; return list of normalized damage dicts."""
    content = [
        {"type": "text", "text": _TILE_PROMPT.format(ang=ang_human)},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(tile)}"}},
    ]
    kwargs = _build_api_kwargs(model, content)
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


def _tiled_detail_pass(client, model, im_full, angle_label) -> list:
    """Run the deterministic grid detail pass. Returns extra damage dicts.
    Fari/fanali findings are flagged for manual review ([DA VERIFICARE], low confidence)."""
    from concurrent.futures import ThreadPoolExecutor
    ang_human = (angle_label or "").replace("_", " ")
    zone = ZONE_BY_ANGLE.get(angle_label, "frontale")
    tiles = _make_grid(im_full)
    raw_items: list = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for items in ex.map(lambda t: _safe_inspect_tile(client, model, ang_human, t[0], t[1]), tiles):
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
        is_light = any(k in (comp + " " + desc).lower() for k in _LIGHT_KW)
        if is_light:
            desc = "[DA VERIFICARE] " + desc
        out.append({
            "damage_type": dt, "severity": sev, "zone": zone, "description": desc,
            "bounding_box": it.get("box"),
            "confidence": 0.4 if is_light else 0.5,
            "needs_review": is_light,
        })
    return out


def _safe_inspect_tile(client, model, ang_human, box_str, tile) -> list:
    try:
        return _inspect_tile(client, model, ang_human, box_str, tile)
    except Exception as e:
        logger.warning("tile inspect failed box=%s: %s", box_str, e)
        return []


def _call_openai_single(client, model: str, photo: Photo, vehicle_type: str | None) -> tuple[list, str]:
    """Synchronous: run ONE OpenAI call for ONE photo. Returns (validated_damages, raw_text).

    Raises on transport/API failures; callers should catch and log per-photo.
    """
    b64 = _encode_image_base64(photo.file_path, getattr(photo, "image_data", None))
    if b64 is None:
        return [], ""

    prompt = _load_prompt(vehicle_type, photo.angle_label)
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

    api_kwargs = _build_api_kwargs(model, content)
    logger.info(
        "OpenAI request (per-photo): model=%s, angle=%s, vehicle_type=%s",
        model, photo.angle_label, vehicle_type,
    )

    response = _chat_with_retry(client, **api_kwargs)
    _log_usage(response, model, f"main:{photo.angle_label}")

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

    # Stage 2 (scudo with reference): deterministic grid detail pass. Catches small
    # damage (cracked light lens, broken/missing lower bumper) that is below the
    # detection threshold at full-frame 720p. Fari findings are flagged DA VERIFICARE.
    if ref_path and ref_b64:
        im_full = _pil_from_source(photo.file_path, getattr(photo, "image_data", None))
        if im_full is not None:
            try:
                extra = _tiled_detail_pass(client, model, im_full, photo.angle_label)
            except Exception as e:
                logger.warning("Stage-2 tiled pass failed angle=%s: %s", photo.angle_label, e)
                extra = []
            if extra:
                before = len(validated)
                validated = _merge_damages(validated, extra)
                logger.info(
                    "Stage-2 tiled angle=%s: +%d detail damages (%d unique after merge)",
                    photo.angle_label, len(extra), len(validated) - before,
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

    async def _run_one(photo: Photo) -> tuple[str, list, str, str | None]:
        """Run [passes] independent VLM passes for one photo, concurrently, and
        merge them union-wise (recall-first). A photo is only marked failed if
        EVERY pass failed — a partial failure still yields the surviving passes."""
        outcomes = await asyncio.gather(
            *(
                asyncio.to_thread(_call_openai_single, client, model, photo, vehicle_type)
                for _ in range(passes)
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
        merged = _merge_passes(pass_damages, passes) if passes > 1 else pass_damages[0]
        raw = f"[multipass {len(ok)}/{passes}]\n" + (ok[0][1] or "")
        logger.info(
            "angle=%s multipass: %d unique damages from %d/%d passes",
            photo.angle_label, len(merged), len(ok), passes,
        )
        return photo.angle_label, merged, raw, None

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

            # Save damages
            for damage_data in damage_list:
                damage = Damage(
                    id=str(uuid.uuid4()),
                    analysis_id=analysis_id,
                    damage_type=damage_data["damage_type"],
                    severity=damage_data["severity"],
                    zone=damage_data["zone"],
                    description=damage_data.get("description"),
                    bounding_box=damage_data.get("bounding_box"),
                    confidence=damage_data.get("confidence"),
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
