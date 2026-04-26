from pydantic import BaseModel


class SessionCreate(BaseModel):
    vehicle_id: str
    user_id: str
    id: str | None = None
    name: str | None = None


class SessionResponse(BaseModel):
    id: str
    vehicle_id: str
    user_id: str
    started_at: str
    completed_at: str | None = None
    status: str
    total_photos: int
    valid_photos: int
    name: str | None = None

    model_config = {"from_attributes": True}
