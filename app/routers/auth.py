from datetime import date

import bcrypt
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse
from app.utils.exceptions import AppException
from app.utils.response import success_response

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == request.username))
    user = result.scalars().first()

    if user is None:
        raise AppException("Credenziali non valide", status_code=400)

    if not bcrypt.checkpw(request.password.encode(), user.password_hash.encode()):
        raise AppException("Credenziali non valide", status_code=400)

    # Check expiration date
    if user.enabled_until:
        try:
            expiry = date.fromisoformat(user.enabled_until)
            if date.today() > expiry:
                raise AppException("Utente scaduto. Contattare l'amministratore.", status_code=403)
        except ValueError:
            pass

    # Check remaining calls
    if user.remaining_calls is not None and user.remaining_calls <= 0:
        raise AppException("Chiamate esaurite. Contattare l'amministratore.", status_code=403)

    return success_response(
        data=LoginResponse(user_id=user.id, username=user.username).model_dump()
    )
