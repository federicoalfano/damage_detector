from sqlalchemy import Column, String, Integer, ForeignKey

from app.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    vehicle_id = Column(String, ForeignKey("vehicles.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    started_at = Column(String, nullable=False)
    completed_at = Column(String, nullable=True)
    status = Column(String, nullable=False, default="in_progress")
    total_photos = Column(Integer, nullable=False)
    valid_photos = Column(Integer, nullable=False, default=0)
