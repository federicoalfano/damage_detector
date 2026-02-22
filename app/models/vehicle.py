from sqlalchemy import Column, String

from app.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(String, primary_key=True)
    model = Column(String, nullable=False)
    plate = Column(String, nullable=False, unique=True)
    type = Column(String, nullable=False)
