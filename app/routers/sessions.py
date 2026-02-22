import asyncio
import os
import uuid as uuid_mod
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from sqlalchemy import select

from app.database import async_session
from app.models.analysis import AnalysisResult, Damage
from app.models.session import Session
from app.models.photo import Photo
from app.models.vehicle import Vehicle
from app.models.user import User
from app.schemas.session import SessionCreate, SessionResponse
from app.services.ai_service import analyze_session
from app.services.photo_validator import validate_photo
from app.utils.response import success_response

router = APIRouter(prefix="/sessions", tags=["sessions"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sessions")


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

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Validate photo against expected vehicle type
        vehicle_type = vehicle.type if vehicle else ""
        validation = await validate_photo(file_path, vehicle_type)

        if not validation["valid"]:
            # Remove the invalid file
            os.remove(file_path)
            raise HTTPException(
                status_code=422,
                detail=f"Foto non valida: {validation['reason']}",
            )

        # Create photo record
        photo = Photo(
            id=photo_id,
            session_id=session_id,
            angle_index=angle_index,
            angle_label=angle_label,
            file_path=file_path,
            captured_at=datetime.now(timezone.utc).isoformat(),
            is_valid=1,
            upload_status="uploaded",
        )
        db_session.add(photo)
        await db_session.commit()

    return success_response(data={"photo_id": photo_id})


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

    # Trigger AI analysis asynchronously
    asyncio.create_task(analyze_session(session_id))

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
        result = await db_session.execute(select(Session))
        sessions = result.scalars().all()

        data = []
        for s in sessions:
            session_data = SessionResponse.model_validate(s).model_dump()

            # Fetch analysis info
            ar_result = await db_session.execute(
                select(AnalysisResult).where(AnalysisResult.session_id == s.id)
            )
            analysis = ar_result.scalars().first()

            damage_types: list[str] = []
            damage_count = 0
            if analysis and analysis.status == "completed":
                dmg_result = await db_session.execute(
                    select(Damage).where(Damage.analysis_id == analysis.id)
                )
                damages = dmg_result.scalars().all()
                damage_count = len(damages)
                damage_types = list({d.damage_type for d in damages})

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
                }
                for d in damages
            ]

        return success_response(data={
            "analysis_status": analysis.status,
            "damages": damages_list,
        })
