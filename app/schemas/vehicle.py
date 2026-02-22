from pydantic import BaseModel


class VehicleResponse(BaseModel):
    id: str
    model: str
    plate: str
    type: str

    model_config = {"from_attributes": True}
