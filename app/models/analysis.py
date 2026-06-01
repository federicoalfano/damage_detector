from sqlalchemy import Column, String, ForeignKey, Float, Integer

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
    confidence = Column(Float, nullable=True)  # 0.0-1.0 (votes/passes or YOLO prob)
    # 1 = surface as an amber "da verificare" triage state (resolution-sensitive
    # findings like a possibly-cracked light lens, or a checklist-derived missing
    # part). Previously this signal lived only in the description text + a low
    # confidence and was lost at persistence.
    needs_review = Column(Integer, nullable=False, default=0)
