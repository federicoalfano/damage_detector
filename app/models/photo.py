from sqlalchemy import Column, String, Integer
from sqlalchemy import ForeignKey

from app.database import Base


class Photo(Base):
    __tablename__ = "photos"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    angle_index = Column(Integer, nullable=False)
    angle_label = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    captured_at = Column(String, nullable=False)
    is_valid = Column(Integer, nullable=False, default=0)
    validation_message = Column(String, nullable=True)
    upload_status = Column(String, nullable=False, default="pending")
