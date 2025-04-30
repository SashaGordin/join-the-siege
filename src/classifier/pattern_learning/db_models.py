"""SQLAlchemy models for pattern learning system."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
from ..config.database import Base
from .models import PatternType, MatchType

class Pattern(Base):
    """SQLAlchemy model for patterns."""
    __tablename__ = 'patterns'

    id = Column(String, primary_key=True)
    type = Column(SQLEnum(PatternType))
    expression = Column(String, nullable=False)
    feature_type = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    confidence_value = Column(Float, default=1.0)
    confidence_factors = Column(JSON, default=dict)
    pattern_metadata = Column(JSON, default=dict)

    # Relationships
    matches = relationship("PatternMatch", back_populates="pattern", cascade="all, delete-orphan")
    examples = relationship("PatternExample", back_populates="pattern", cascade="all, delete-orphan")
    validation_results = relationship("PatternValidation", back_populates="pattern", cascade="all, delete-orphan")

class PatternMatch(Base):
    """SQLAlchemy model for pattern matches."""
    __tablename__ = 'pattern_matches'

    id = Column(Integer, primary_key=True)
    pattern_id = Column(String, ForeignKey('patterns.id'), nullable=False)
    text = Column(String, nullable=False)
    start = Column(Integer, nullable=False)
    end = Column(Integer, nullable=False)
    match_type = Column(SQLEnum(MatchType))
    confidence_value = Column(Float, default=1.0)
    confidence_factors = Column(JSON, default=dict)
    context = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pattern = relationship("Pattern", back_populates="matches")

class PatternExample(Base):
    """SQLAlchemy model for pattern examples."""
    __tablename__ = 'pattern_examples'

    id = Column(Integer, primary_key=True)
    pattern_id = Column(String, ForeignKey('patterns.id'), nullable=False)
    text = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pattern = relationship("Pattern", back_populates="examples")

class PatternValidation(Base):
    """SQLAlchemy model for pattern validation results."""
    __tablename__ = 'pattern_validations'

    id = Column(Integer, primary_key=True)
    pattern_id = Column(String, ForeignKey('patterns.id'), nullable=False)
    true_positives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pattern = relationship("Pattern", back_populates="validation_results")