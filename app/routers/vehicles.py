from fastapi import APIRouter
from sqlalchemy import select

from app.database import async_session
from app.models.vehicle import Vehicle
from app.schemas.vehicle import VehicleResponse
from app.utils.response import success_response

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


@router.get("")
async def get_vehicles():
    async with async_session() as session:
        result = await session.execute(select(Vehicle))
        vehicles = result.scalars().all()
        data = [VehicleResponse.model_validate(v).model_dump() for v in vehicles]
    return success_response(data=data)
