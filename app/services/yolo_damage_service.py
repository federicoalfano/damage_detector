"""YOLO-based damage detection service (ensemble of two pretrained models).

Models:
  - vineetsarpal/yolov11n-car-damage (HuggingFace, 14 classes component+damage)
  - shawnmichael/yolo-car-damage-detection (HuggingFace, 6 generic damage classes)

Outputs damages in the project schema: damage_type, severity, zone, description, bounding_box.
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_N: Optional[object] = None  # YOLOv11n (14 classes)
_MODEL_M: Optional[object] = None  # YOLO generic (6 classes)

# Map angle_label -> canonical zone
ANGLE_TO_ZONE = {
    "fronte": "frontale",
    "lato_destro": "laterale_destro",
    "lato_sinistro": "laterale_sinistro",
    "retro": "posteriore",
}

# YOLOv11n class -> (damage_type, inferred_zone_hint, severity, desc_prefix)
YOLO_N_MAPPING = {
    "Front-windscreen-damage": ("crepa", "frontale", "moderato", "parabrezza danneggiato"),
    "Headlight-damage": ("rottura", "frontale", "grave", "faro anteriore danneggiato"),
    "Rear-windscreen-Damage": ("crepa", "posteriore", "moderato", "vetro posteriore danneggiato"),
    "Runningboard-Damage": ("graffio", None, "moderato", "pedana/brancardo danneggiato"),
    "Sidemirror-Damage": ("rottura", None, "moderato", "specchietto laterale danneggiato"),
    "Taillight-Damage": ("rottura", "posteriore", "grave", "fanale posteriore danneggiato"),
    "bonnet-dent": ("ammaccatura", "frontale", "moderato", "ammaccatura sul cofano"),
    "boot-dent": ("ammaccatura", "posteriore", "moderato", "ammaccatura sul portellone posteriore"),
    "doorouter-dent": ("ammaccatura", None, "moderato", "ammaccatura su una portiera"),
    "fender-dent": ("ammaccatura", None, "moderato", "ammaccatura sul parafango"),
    "front-bumper-dent": ("ammaccatura", "frontale", "moderato", "ammaccatura sul paraurti anteriore"),
    "quaterpanel-dent": ("ammaccatura", None, "moderato", "ammaccatura sul pannello posteriore"),
    "rear-bumper-dent": ("ammaccatura", "posteriore", "moderato", "ammaccatura sul paraurti posteriore"),
    "roof-dent": ("ammaccatura", "superiore", "moderato", "ammaccatura sul tetto"),
}

# YOLO generic class -> (damage_type, severity, desc_prefix)
# Covers both naming conventions (underscore and space variants).
YOLO_M_MAPPING = {
    "dent": ("ammaccatura", "moderato", "ammaccatura"),
    "scratch": ("graffio", "lieve", "graffio"),
    "crack": ("crepa", "moderato", "crepa"),
    "shattered_glass": ("crepa", "grave", "vetro infranto"),
    "glass shatter": ("crepa", "grave", "vetro infranto"),
    "broken_lamp": ("rottura", "grave", "faro/fanale rotto"),
    "lamp broken": ("rottura", "grave", "faro/fanale rotto"),
    "flat_tire": ("usura", "grave", "pneumatico sgonfio o molto usurato"),
    "tire flat": ("usura", "grave", "pneumatico sgonfio o molto usurato"),
}


def _load_models():
    global _MODEL_N, _MODEL_M
    if _MODEL_N is not None and _MODEL_M is not None:
        return
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
    except Exception as e:
        logger.warning("ultralytics/huggingface_hub not available: %s — YOLO disabled", e)
        return

    if _MODEL_N is None:
        try:
            path = hf_hub_download(repo_id="vineetsarpal/yolov11n-car-damage", filename="best.pt")
            _MODEL_N = YOLO(path)
            logger.info("Loaded YOLOv11n damage model from %s", path)
        except Exception as e:
            logger.warning("Failed to load YOLOv11n: %s", e)

    if _MODEL_M is None:
        try:
            path = hf_hub_download(repo_id="shawnmichael/yolo-car-damage-detection", filename="best_dts.pt")
            _MODEL_M = YOLO(path)
            logger.info("Loaded generic YOLO damage model from %s", path)
        except Exception as e:
            logger.warning("Failed to load shawnmichael YOLO: %s", e)


def yolo_available() -> bool:
    _load_models()
    return _MODEL_N is not None or _MODEL_M is not None


def _det_signature(damage_type: str, zone: str) -> tuple:
    """Signature for near-duplicate dedup across YOLO models."""
    return (damage_type, zone)


def _upright_path(file_path: str) -> str:
    """Return path to an upright copy of the image (EXIF-transposed).
    Browsers honor EXIF for img tags, but YOLO reads raw pixels. Caller must
    use this so bbox coords match the rendered image.
    """
    import tempfile
    from PIL import Image, ImageOps
    try:
        with Image.open(file_path) as im:
            ori = im.getexif().get(274)
            if not ori or ori == 1:
                return file_path
            upright = ImageOps.exif_transpose(im).convert("RGB")
            fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="yolo_upright_")
            os.close(fd)
            upright.save(tmp, format="JPEG", quality=92)
            return tmp
    except Exception as e:
        logger.warning("EXIF transpose failed for %s: %s — using raw", file_path, e)
        return file_path


def detect_on_photo(file_path: str, angle_label: str) -> list[dict]:
    """Run both YOLO models on a photo and return merged damages in schema format."""
    _load_models()
    zone = ANGLE_TO_ZONE.get(angle_label, "frontale")
    found: list[dict] = []
    seen: set = set()
    file_path = _upright_path(file_path)

    # YOLOv11n pass (component-aware)
    if _MODEL_N is not None:
        try:
            res = _MODEL_N.predict(file_path, conf=0.30, verbose=False)[0]
            for b in res.boxes:
                cls_name = _MODEL_N.names[int(b.cls[0])]
                mapping = YOLO_N_MAPPING.get(cls_name)
                if not mapping:
                    continue
                dtype, zhint, sev, desc = mapping
                final_zone = zhint or zone
                sig = _det_signature(dtype, final_zone)
                if sig in seen:
                    continue
                seen.add(sig)
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                found.append({
                    "damage_type": dtype,
                    "severity": sev,
                    "zone": final_zone,
                    "description": f"{desc} (YOLO-n {conf:.0%})",
                    "bounding_box": f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}",
                })
        except Exception as e:
            logger.warning("YOLOv11n inference failed on %s: %s", file_path, e)

    # YOLO11m pass (generic damages) — per-class thresholds because shattered_glass
    # and broken_lamp trigger on asphalt cracks, shoes, parking lot textures.
    YOLO_M_CLASS_THRESHOLDS = {
        "dent": 0.45,
        "scratch": 0.50,
        "crack": 0.55,
        "shattered_glass": 0.85,
        "glass shatter": 0.85,
        "broken_lamp": 0.55,
        "lamp broken": 0.55,
        "flat_tire": 0.60,
        "tire flat": 0.60,
        "none": 1.1,  # never trigger (placeholder class)
    }
    if _MODEL_M is not None:
        try:
            res = _MODEL_M.predict(file_path, conf=0.30, verbose=False)[0]
            for b in res.boxes:
                cls_name = _MODEL_M.names[int(b.cls[0])]
                mapping = YOLO_M_MAPPING.get(cls_name)
                if not mapping:
                    continue
                conf = float(b.conf[0])
                if conf < YOLO_M_CLASS_THRESHOLDS.get(cls_name, 0.45):
                    continue
                dtype, sev, desc = mapping
                sig = _det_signature(dtype, zone)
                if sig in seen:
                    continue
                seen.add(sig)
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                found.append({
                    "damage_type": dtype,
                    "severity": sev,
                    "zone": zone,
                    "description": f"{desc} (YOLO-m {conf:.0%})",
                    "bounding_box": f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}",
                })
        except Exception as e:
            logger.warning("YOLO11m inference failed on %s: %s", file_path, e)

    return found
