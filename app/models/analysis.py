from sqlalchemy import Column, String, ForeignKey

from app.database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    status = Column(String, nullable=False, default="pending")
    completed_at = Column(String, nullable=True)
    raw_response = Column(String, nullable=True)


class Damage(Base):
    __tablename__ = "damages"

    id = Column(String, primary_key=True)
    analysis_id = Column(String, ForeignKey("analysis_results.id"), nullable=False)
    damage_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    zone = Column(String, nullable=False)
    description = Column(String, nullable=True)
    bounding_box = Column(String, nullable=True)
