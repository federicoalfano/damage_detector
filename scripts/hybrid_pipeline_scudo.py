"""Hybrid damage-detection pipeline for the scudo recovered sessions.

Per session:
  1. Photo validator (VLM call): mark photos without a vehicle/bad framing as invalid.
  2. YOLO ensemble on valid photos (two pretrained models).
  3. VLM pass (Gemma 3 27B via OpenRouter) on valid photos with the scudo prompt.
  4. Merge YOLO + VLM damages, dedupe on (type, zone).
  5. Persist into AnalysisResult / Damage rows.
"""
import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))
os.chdir(API_ROOT)

from sqlalchemy import delete, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session  # noqa: E402
from app.models.analysis import AnalysisResult, Damage  # noqa: E402
from app.models.photo import Photo  # noqa: E402
from app.models.session import Session  # noqa: E402
from app.services.ai_service import _extract_damages, _load_prompt  # noqa: E402
from app.services.photo_validator import validate_photo  # noqa: E402
from app.services.yolo_damage_service import detect_on_photo, yolo_available  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("hybrid_pipeline")

SCUDO_VEHICLE_ID = "5ca791b5-2664-5ce0-a7d7-bf467f113820"

ANGLE_LABELS = {
    "fronte": "FOTO FRONTALE",
    "lato_destro": "FOTO LATO DESTRO",
    "lato_sinistro": "FOTO LATO SINISTRO",
    "retro": "FOTO POSTERIORE",
}

# Severity order for merge conflicts
SEV_RANK = {"lieve": 1, "moderato": 2, "grave": 3}


def _vlm_client():
    from openai import OpenAI
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=120.0,
    )


VALID_TYPES = {"graffio", "ammaccatura", "crepa", "rottura", "pezzo_mancante", "usura", "sporcizia"}
VALID_SEV = {"lieve", "moderato", "grave"}
VALID_ZONES = {"frontale", "laterale_sinistro", "posteriore", "laterale_destro", "superiore"}
GRAVE_ONLY = {"usura", "sporcizia"}


def _encode_photo_oriented(path: Path) -> str | None:
    """Read a JPEG, applica EXIF orientation fisicamente, ritorna base64.

    Foto Pixel/Android sono spesso "ruotate via EXIF": se le mandi raw al
    modello vede il furgone ruotato di 90° → checklist sballata. Allinea
    questo encoder a quello di ai_service._encode_image_base64.
    """
    if not path.exists():
        return None
    raw = path.read_bytes()
    if not raw:
        return None
    try:
        from io import BytesIO
        from PIL import Image, ImageOps
        with Image.open(BytesIO(raw)) as im:
            transposed = ImageOps.exif_transpose(im)
            buf = BytesIO()
            transposed.convert("RGB").save(buf, format="JPEG", quality=92)
            data = buf.getvalue()
    except Exception as e:
        logger.warning("EXIF transpose failed for %s (%s) — using raw bytes", path, e)
        data = raw
    return base64.b64encode(data).decode("utf-8")


async def _vlm_call_per_angle(
    photo: Photo, temperature: float, seed: int | None
) -> tuple[list[dict], str]:
    """Una chiamata VLM PER FOTO con il prompt dedicato all'angolo.

    Cambia rispetto al vecchio batch:
      - prompt per-angolo (fronte/lato_destro/...): checklist mirate.
      - foto trasposta via EXIF prima dell'encode.
      - una call indipendente per foto → consensus per-foto, non globale.
    """
    path = API_ROOT / photo.file_path
    b64 = await asyncio.to_thread(_encode_photo_oriented, path)
    if b64 is None:
        return [], ""

    prompt = _load_prompt("scudo", photo.angle_label)
    label = ANGLE_LABELS.get(photo.angle_label, photo.angle_label)
    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"--- {label} ---"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]

    client = _vlm_client()
    t0 = time.time()
    kwargs: dict = {
        "model": settings.openai_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": temperature,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        resp = await asyncio.to_thread(lambda: client.chat.completions.create(**kwargs))
    except Exception as e:
        logger.warning("VLM call exception (angle=%s, seed=%s): %s", photo.angle_label, seed, e)
        return [], ""
    if not resp.choices:
        logger.warning("VLM no choices angle=%s seed=%s", photo.angle_label, seed)
        return [], ""
    raw = (resp.choices[0].message.content or "") if resp.choices[0].message else ""
    logger.info(
        "VLM angle=%s t=%.1f seed=%s: %d chars in %.1fs",
        photo.angle_label, temperature, seed, len(raw), time.time() - t0,
    )

    txt = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if txt.startswith("```"):
        txt = "\n".join(l for l in txt.split("\n") if not l.strip().startswith("```")).strip()
    try:
        damages = _extract_damages(txt)
    except Exception as e:
        logger.warning("VLM damage parsing failed: %s — raw: %s", e, raw[:200])
        damages = []

    cleaned = []
    for d in damages:
        if (
            d.get("damage_type") in VALID_TYPES
            and d.get("severity") in VALID_SEV
            and d.get("zone") in VALID_ZONES
        ):
            if d["damage_type"] in GRAVE_ONLY and d["severity"] != "grave":
                continue
            cleaned.append(d)
    return cleaned, raw


def _consensus(runs: list[list[dict]], min_votes: int = 2) -> list[dict]:
    """Keep damages that appear in ≥ min_votes of the runs.

    Two damages 'match' if they share (damage_type, zone). When matched,
    we take the highest severity across the matching occurrences.
    """
    SEV_RANK = {"lieve": 1, "moderato": 2, "grave": 3}
    # Bucket damages by (damage_type, zone), tracking which run each came from
    buckets: dict[tuple, list[tuple[int, dict]]] = {}
    for run_idx, run in enumerate(runs):
        seen_in_run: set = set()
        for d in run:
            key = (d["damage_type"], d["zone"])
            if key in seen_in_run:
                continue  # one vote per run
            seen_in_run.add(key)
            buckets.setdefault(key, []).append((run_idx, d))

    total_runs = len(runs)
    consensus = []
    for key, occurrences in buckets.items():
        votes = len({idx for idx, _ in occurrences})
        if votes < min_votes:
            continue
        # Pick the highest-severity version
        best = max((d for _, d in occurrences), key=lambda x: SEV_RANK.get(x["severity"], 0))
        best = dict(best)
        best["confidence"] = round(votes / total_runs, 2) if total_runs else 0.0
        consensus.append(best)
    return consensus


async def vlm_call(valid_photos: list[Photo], passes: int = 3) -> tuple[list[dict], str]:
    """Multi-pass VLM PER-FOTO con voto di consenso.

    Pipeline:
      1. Per ogni foto valida lancia `passes` chiamate VLM con seed diversi
         (in parallelo, asyncio.gather). Ogni call usa il prompt dell'angolo.
      2. Per ogni foto, combina i `passes` con consensus ≥ 2/3
         (un danno deve apparire in almeno 2 pass per essere accettato).
         Severity finale = mediana → riduce sia falsi positivi che inflazione.
      3. Aggrega i danni accettati di tutte le foto (zona è già implicita
         nell'angolo, niente cross-foto).
    Manteniamo budget invariato: stesso numero di chiamate (n_photos × passes)
    del vecchio approccio (1 call con n foto × passes).
    """
    if not valid_photos:
        return [], ""

    # Diverse sampling params: solo 2 pass servono se temperature è 0
    # con reasoning model (o4-mini); il deterministico già aiuta.
    configs = [
        {"temperature": 0.0, "seed": 7},
        {"temperature": 0.3, "seed": 42},
        {"temperature": 0.6, "seed": 1337},
    ][:passes]

    # Per ogni foto, lancia tutti i pass in parallelo
    async def _photo_runs(photo: Photo) -> tuple[Photo, list[list[dict]], list[str]]:
        results = await asyncio.gather(
            *(_vlm_call_per_angle(photo, **cfg) for cfg in configs)
        )
        runs = [r[0] for r in results]
        raws = [r[1] for r in results]
        return photo, runs, raws

    photo_results = await asyncio.gather(*(_photo_runs(p) for p in valid_photos))

    # Consensus per-foto, poi unione globale
    all_damages: list[dict] = []
    raws_by_angle: dict[str, list[str]] = {}
    counts_by_angle: dict[str, list[int]] = {}
    min_votes = max(2, (len(configs) + 1) // 2)  # 2/3, 2/2, 1/1
    for photo, runs, raws in photo_results:
        per_run = [len(r) for r in runs]
        counts_by_angle[photo.angle_label] = per_run
        raws_by_angle[photo.angle_label] = raws
        accepted = _consensus(runs, min_votes=min_votes)
        logger.info(
            "Angle %s consensus: per_run=%s -> accepted=%d (min_votes=%d)",
            photo.angle_label, per_run, len(accepted), min_votes,
        )
        all_damages.extend(accepted)

    combined = json.dumps(
        {
            "per_angle_runs": raws_by_angle,
            "per_angle_counts": counts_by_angle,
            "min_votes": min_votes,
            "consensus": all_damages,
        },
        ensure_ascii=False,
    )[:80000]
    return all_damages, combined


def merge_damages(yolo_list: list[dict], vlm_list: list[dict]) -> list[dict]:
    """Merge YOLO + VLM damages, dedupe on (damage_type, zone), prefer higher severity."""
    out: dict[tuple, dict] = {}
    for d in yolo_list + vlm_list:
        key = (d["damage_type"], d["zone"])
        if key not in out:
            out[key] = d.copy()
            continue
        # Keep the one with higher severity
        existing = out[key]
        if SEV_RANK.get(d["severity"], 0) > SEV_RANK.get(existing["severity"], 0):
            out[key] = d.copy()
        elif d["severity"] == existing["severity"]:
            # Accumulate descriptions
            existing["description"] = (
                existing["description"] + " | " + d["description"]
            )[:480]
    return list(out.values())


async def process_session(sid: str) -> dict:
    stats = {"session_id": sid, "invalid_photos": 0, "yolo_damages": 0, "vlm_damages": 0, "final": 0}

    async with async_session() as db:
        # Fetch photos
        photos_result = await db.execute(select(Photo).where(Photo.session_id == sid))
        photos = photos_result.scalars().all()

        # 1. Validate each photo (passes angle_label for angle-match check)
        valid_photos: list[Photo] = []
        for p in photos:
            abs_path = API_ROOT / p.file_path
            v = await validate_photo(str(abs_path), "scudo", p.angle_label)
            p.is_valid = 1 if v.get("valid") else 0
            p.validation_message = (v.get("reason", "") or "")[:200]
            if not v.get("valid"):
                stats["invalid_photos"] += 1
                logger.warning("Session %s photo %s (%s) INVALID: %s",
                               sid, p.id, p.angle_label, v.get("reason"))
            else:
                valid_photos.append(p)
        await db.commit()

        # 2. YOLO disabled per user decision
        yolo_damages: list[dict] = []
        stats["yolo_damages"] = 0

        # 3. VLM multi-pass voting (3 runs, consensus ≥ 2/3)
        vlm_damages, vlm_raw = await vlm_call(valid_photos, passes=3)
        stats["vlm_damages"] = len(vlm_damages)

        # 4. Final = VLM consensus damages (no YOLO, no post-filter needed)
        final = vlm_damages
        stats["final"] = len(final)

        # 5. Persist: wipe old analysis, insert new
        ar = await db.execute(select(AnalysisResult).where(AnalysisResult.session_id == sid))
        for row in ar.scalars().all():
            await db.execute(delete(Damage).where(Damage.analysis_id == row.id))
            await db.delete(row)
        await db.commit()

        analysis_id = str(uuid.uuid4())
        combined_raw = json.dumps({
            "yolo": yolo_damages,
            "vlm_raw": vlm_raw,
            "merged": final,
        }, ensure_ascii=False)[:50000]

        db.add(AnalysisResult(
            id=analysis_id,
            session_id=sid,
            status="completed",
            raw_response=combined_raw,
        ))
        for d in final:
            db.add(Damage(
                id=str(uuid.uuid4()),
                analysis_id=analysis_id,
                damage_type=d["damage_type"],
                severity=d["severity"],
                zone=d["zone"],
                description=d.get("description", ""),
                bounding_box=d.get("bounding_box"),
                confidence=d.get("confidence"),
            ))

        sess = await db.get(Session, sid)
        if sess:
            sess.status = "completed"
        await db.commit()

    logger.info("Session %s done: %s", sid, stats)
    return stats


async def main():
    conn = sqlite3.connect(API_ROOT / "data" / "db.sqlite3")
    scudo_sids = [r[0] for r in conn.execute(
        "SELECT id FROM sessions WHERE vehicle_id=? ORDER BY started_at",
        (SCUDO_VEHICLE_ID,),
    )]
    conn.close()
    logger.info("Processing %d scudo sessions with hybrid pipeline", len(scudo_sids))

    all_stats = []
    for sid in scudo_sids:
        try:
            s = await process_session(sid)
        except Exception as e:
            logger.exception("Session %s failed: %s", sid, e)
            s = {"session_id": sid, "error": str(e)}
        all_stats.append(s)

    print("\n======== SUMMARY ========")
    print(f"{'session':38} {'invalid':>7} {'yolo':>5} {'vlm':>4} {'final':>5}")
    for s in all_stats:
        if "error" in s:
            print(f"{s['session_id']:38} ERROR: {s['error'][:60]}")
        else:
            print(f"{s['session_id']:38} {s['invalid_photos']:>7} {s['yolo_damages']:>5} {s['vlm_damages']:>4} {s['final']:>5}")


if __name__ == "__main__":
    asyncio.run(main())
