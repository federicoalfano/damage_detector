from fastapi import Header, HTTPException

from app.config import settings


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
