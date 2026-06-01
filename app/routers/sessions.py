import os
import uuid as uuid_mod
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
import shutil

from sqlalchemy import select, delete

from app.config import settings
from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.models.session import Session
from app.models.photo import Photo
from app.models.vehicle import Vehicle
from app.models.user import User
from app.schemas.session import SessionCreate, SessionResponse
from app.services.ai_service import spawn_analysis
from app.utils.response import success_response

router = APIRouter(prefix="/sessions", tags=["sessions"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sessions")

CANONICAL_ANGLES = ("retro", "lato_destro", "fronte", "lato_sinistro")


@router.post("", status_code=201)
async def create_session(payload: SessionCreate):
    async with async_session() as session:
        vehicle = await session.get(Vehicle, payload.vehicle_id)
        if not vehicle:
            raise HTTPException(status_code=404, detail="Veicolo non trovato")

        user = await session.get(User, payload.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")

        session_id = payload.id or str(uuid_mod.uuid4())

        # Check if session already exists (idempotent create)
        existing = await session.get(Session, session_id)
        if existing:
            data = SessionResponse.model_validate(existing).model_dump()
            return success_response(data=data)

        new_session = Session(
            id=session_id,
            vehicle_id=payload.vehicle_id,
            user_id=payload.user_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="in_progress",
            total_photos=4,
            valid_photos=0,
            name=payload.name,
        )
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)

        data = SessionResponse.model_validate(new_session).model_dump()
    return success_response(data=data)


@router.post("/{session_id}/photos", status_code=201)
async def upload_photo(
    session_id: str,
    file: UploadFile = File(...),
    angle_index: int = Form(...),
    angle_label: str = Form(...),
):
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        vehicle = await db_session.get(Vehicle, sess.vehicle_id)

        # Save file to disk
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        photo_id = str(uuid_mod.uuid4())
        filename = f"{photo_id}.jpg"
        file_path = os.path.join(session_dir, filename)

        # Reject non-image uploads (lenient when content_type is absent).
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Tipo file non supportato")

        content = await file.read()

        # Reject oversized uploads before touching disk/DB.
        if len(content) > settings.max_photo_size_bytes:
            raise HTTPException(status_code=413, detail="Foto troppo grande")

        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except OSError:
            # Disk write may fail on read-only filesystems; DB blob is enough.
            pass

        # Photo validation disabled — saves one API call per photo
        # vehicle_type = vehicle.type if vehicle else ""
        # validation = await validate_photo(file_path, vehicle_type)
        # if not validation["valid"]:
        #     os.remove(file_path)
        #     raise HTTPException(status_code=422, detail=f"Foto non valida: {validation['reason']}")

        # Create photo record. Image bytes also stored in DB so the photo
        # survives Render's ephemeral disk wipes.
        photo = Photo(
            id=photo_id,
            session_id=session_id,
            angle_index=angle_index,
            angle_label=angle_label,
            file_path=file_path,
            image_data=content,
            captured_at=datetime.now(timezone.utc).isoformat(),
            is_valid=1,
            upload_status="uploaded",
        )
        db_session.add(photo)
        await db_session.commit()

    return success_response(data={"photo_id": photo_id, "size_bytes": len(content)})


@router.post("/{session_id}/complete")
async def complete_session(session_id: str):
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        sess.status = "uploaded"
        sess.completed_at = datetime.now(timezone.utc).isoformat()

        # Count uploaded photos
        result = await db_session.execute(
            select(Photo).where(Photo.session_id == session_id)
        )
        photos = result.scalars().all()
        sess.valid_photos = len([p for p in photos if p.is_valid])

        await db_session.commit()
        await db_session.refresh(sess)

        data = SessionResponse.model_validate(sess).model_dump()

    # Trigger AI analysis asynchronously (GC-safe, ref held in ai_service)
    spawn_analysis(session_id)

    return success_response(data=data)


@router.post("/{session_id}/incomplete")
async def mark_incomplete(session_id: str):
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        sess.status = "incomplete"
        sess.completed_at = datetime.now(timezone.utc).isoformat()

        # Count uploaded photos
        result = await db_session.execute(
            select(Photo).where(Photo.session_id == session_id)
        )
        photos = result.scalars().all()
        sess.valid_photos = len([p for p in photos if p.is_valid])

        await db_session.commit()
        await db_session.refresh(sess)

        data = SessionResponse.model_validate(sess).model_dump()
    return success_response(data=data)


@router.get("")
async def list_sessions():
    async with async_session() as db_session:
        sessions = (await db_session.execute(select(Session))).scalars().all()

        # Bulk-load analyses + damages to avoid an N+1 (was 1 + 2N queries).
        analyses = (await db_session.execute(select(AnalysisResult))).scalars().all()
        analysis_by_session: dict[str, AnalysisResult] = {}
        for a in analyses:
            analysis_by_session.setdefault(a.session_id, a)

        completed_ids = [a.id for a in analyses if a.status == "completed"]
        dmg_by_analysis: dict[str, list] = defaultdict(list)
        if completed_ids:
            damages = (await db_session.execute(
                select(Damage).where(Damage.analysis_id.in_(completed_ids))
            )).scalars().all()
            for d in damages:
                dmg_by_analysis[d.analysis_id].append(d)

        data = []
        for s in sessions:
            session_data = SessionResponse.model_validate(s).model_dump()
            analysis = analysis_by_session.get(s.id)

            damage_types: list[str] = []
            damage_count = 0
            if analysis and analysis.status == "completed":
                dmgs = dmg_by_analysis.get(analysis.id, [])
                damage_count = len(dmgs)
                damage_types = list({d.damage_type for d in dmgs})

            session_data["analysis_status"] = analysis.status if analysis else "pending"
            session_data["damage_types"] = damage_types
            session_data["damage_count"] = damage_count
            data.append(session_data)

    return success_response(data=data)


@router.get("/{session_id}/details")
async def get_session_details(session_id: str):
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        vehicle = await db_session.get(Vehicle, sess.vehicle_id)

        # Get photos
        result = await db_session.execute(
            select(Photo).where(Photo.session_id == session_id)
        )
        photos = result.scalars().all()

        # Get analysis results
        result = await db_session.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        analysis = result.scalars().first()

        damages_list = []
        if analysis and analysis.status == "completed":
            result = await db_session.execute(
                select(Damage).where(Damage.analysis_id == analysis.id)
            )
            damages = result.scalars().all()
            damages_list = [
                {
                    "damage_type": d.damage_type,
                    "severity": d.severity,
                    "zone": d.zone,
                    "description": d.description,
                    "bounding_box": d.bounding_box,
                    "confidence": d.confidence,
                    "needs_review": bool(getattr(d, "needs_review", 0)),
                }
                for d in damages
            ]

        photos_list = [
            {
                "id": p.id,
                "angle_index": p.angle_index,
                "angle_label": p.angle_label,
                "upload_status": p.upload_status,
                "is_valid": bool(p.is_valid),
                "validation_message": p.validation_message,
            }
            for p in photos
        ]

        vehicle_data = None
        if vehicle:
            vehicle_data = {
                "id": vehicle.id,
                "type": vehicle.type,
                "model": vehicle.model,
                "plate": vehicle.plate,
            }

        return success_response(data={
            "session": SessionResponse.model_validate(sess).model_dump(),
            "vehicle": vehicle_data,
            "photos": photos_list,
            "analysis_status": analysis.status if analysis else "pending",
            "damages": damages_list,
        })


@router.get("/{session_id}/results")
async def get_session_results(session_id: str):
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        result = await db_session.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        analysis = result.scalars().first()

        if not analysis:
            return success_response(data={
                "analysis_status": "pending",
                "damages": [],
            })

        damages_list = []
        if analysis.status == "completed":
            result = await db_session.execute(
                select(Damage).where(Damage.analysis_id == analysis.id)
            )
            damages = result.scalars().all()
            damages_list = [
                {
                    "damage_type": d.damage_type,
                    "severity": d.severity,
                    "zone": d.zone,
                    "description": d.description,
                    "bounding_box": d.bounding_box,
                    "confidence": d.confidence,
                    "needs_review": bool(getattr(d, "needs_review", 0)),
                }
                for d in damages
            ]

        response_data: dict = {
            "analysis_status": analysis.status,
            "damages": damages_list,
        }
        if analysis.raw_response:
            response_data["raw_response"] = analysis.raw_response

        return success_response(data=response_data)


@router.get("/{session_id}/photos/{photo_id}")
async def get_photo_file(session_id: str, photo_id: str):
    """Stream the JPEG file for a photo. Tries disk first (faster, supports
    HTTP range), falls back to DB blob when the disk file was wiped (Render
    free tier ephemeral storage). Auth via API key dependency."""
    async with async_session() as db_session:
        photo = await db_session.get(Photo, photo_id)
        if not photo or photo.session_id != session_id:
            raise HTTPException(status_code=404, detail="Foto non trovata")
        file_path = photo.file_path
        blob = photo.image_data

    if file_path and os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")

    if blob:
        # Rehydrate disk cache opportunistically so subsequent reads are fast.
        if file_path:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(blob)
            except OSError:
                pass
        return Response(content=bytes(blob), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="File foto non disponibile")


@router.get("/{session_id}/debug-photos")
async def debug_photos(session_id: str):
    if not settings.debug_endpoints:
        raise HTTPException(status_code=404, detail="Not found")
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        result = await db_session.execute(
            select(Photo).where(Photo.session_id == session_id)
        )
        photos = result.scalars().all()

        info = []
        for p in photos:
            exists = os.path.exists(p.file_path) if p.file_path else False
            size = os.path.getsize(p.file_path) if exists else 0
            blob_size = len(p.image_data) if p.image_data else 0
            info.append({
                "id": p.id,
                "angle": p.angle_label,
                "path": p.file_path,
                "disk_exists": exists,
                "disk_size_bytes": size,
                "blob_size_bytes": blob_size,
            })

    return success_response(data=info)


@router.post("/{session_id}/reanalyze")
async def reanalyze_session(session_id: str, files: list[UploadFile] = File(default=[])):
    # Cap the number of files (normal session has 4; 8 is a safe upper bound).
    if len(files) > 8:
        raise HTTPException(status_code=413, detail="Troppe foto")

    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        # Delete old analysis and damages
        analyses = await db_session.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        for analysis in analyses.scalars().all():
            await db_session.execute(
                delete(Damage).where(Damage.analysis_id == analysis.id)
            )
            await db_session.delete(analysis)

        # If photos provided, save them to disk and update records
        if files:
            # Delete existing photo records
            await db_session.execute(
                delete(Photo).where(Photo.session_id == session_id)
            )

            session_dir = os.path.join(UPLOAD_DIR, session_id)
            os.makedirs(session_dir, exist_ok=True)

            for i, file in enumerate(files):
                photo_id = str(uuid_mod.uuid4())
                filename = f"{photo_id}.jpg"
                file_path = os.path.join(session_dir, filename)

                content = await file.read()

                # Enforce the same size limit as the single-photo upload.
                if len(content) > settings.max_photo_size_bytes:
                    raise HTTPException(status_code=413, detail="Foto troppo grande")

                try:
                    with open(file_path, "wb") as f:
                        f.write(content)
                except OSError:
                    pass

                # Extract angle info from filename (phone sends angle_label as filename).
                # A non-canonical filename would silently disable the per-angle
                # prompt / reference / tiled pass downstream, so fall back to the
                # canonical angle for this index.
                angle_label = file.filename.rsplit('.', 1)[0] if file.filename else f"angle_{i}"
                if angle_label not in CANONICAL_ANGLES:
                    angle_label = CANONICAL_ANGLES[i] if i < 4 else f"angle_{i}"

                photo = Photo(
                    id=photo_id,
                    session_id=session_id,
                    angle_index=i,
                    angle_label=angle_label,
                    file_path=file_path,
                    image_data=content,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    is_valid=1,
                    upload_status="uploaded",
                )
                db_session.add(photo)

        # Reset session status
        sess.status = "uploaded"
        await db_session.commit()

    # Trigger new analysis (GC-safe, ref held in ai_service)
    spawn_analysis(session_id)

    return success_response(data={"message": "Rianalisi avviata"})


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    async with async_session() as db_session:
        sess = await db_session.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Sessione non trovata")

        # Delete damages -> analysis -> photos -> session
        analyses = await db_session.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        for analysis in analyses.scalars().all():
            await db_session.execute(
                delete(Damage).where(Damage.analysis_id == analysis.id)
            )
            await db_session.delete(analysis)

        await db_session.execute(
            delete(Photo).where(Photo.session_id == session_id)
        )
        await db_session.delete(sess)
        await db_session.commit()

    # Remove photos from disk
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)

    return success_response(data={"deleted": session_id})
