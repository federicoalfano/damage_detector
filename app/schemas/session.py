from pydantic import BaseModel


class SessionCreate(BaseModel):
    vehicle_id: str
    user_id: str
    id: str | None = None


class SessionResponse(BaseModel):
    id: str
    vehicle_id: str
    user_id: str
    started_at: str
    completed_at: str | None = None
    status: str
    total_photos: int
    valid_photos: int

    model_config = {"from_attributes": True}
