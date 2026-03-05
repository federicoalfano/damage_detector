from sqlalchemy import Column, String, Integer

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=False)
    enabled_until = Column(String, nullable=True)  # ISO 8601 date, null = no expiry
    remaining_calls = Column(Integer, nullable=False, default=50)
