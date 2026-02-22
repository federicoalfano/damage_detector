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

    return success_response(
        data=LoginResponse(user_id=user.id, username=user.username).model_dump()
    )
