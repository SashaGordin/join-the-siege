"""Database service for pattern learning system."""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime

from .db_models import Pattern as DBPattern
from .db_models import PatternMatch as DBPatternMatch
from .db_models import PatternExample, PatternValidation
from .models import Pattern, PatternMatch, PatternType, MatchType, ConfidenceScore
from ..config.database import get_db

class PatternDBService:
    """Service class for database operations."""

    def __init__(self):
        """Initialize database service."""
        self.db = next(get_db())

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

    def _convert_to_db_pattern(self, pattern: Pattern) -> DBPattern:
        """Convert Pattern model to database Pattern model."""
        return DBPattern(
            id=pattern.id,
            type=pattern.type,
            expression=pattern.expression,
            feature_type=pattern.feature_type,
            industry=pattern.industry,
            version=pattern.version,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at,
            confidence_value=pattern.confidence.value,
            confidence_factors=pattern.confidence.factors,
            pattern_metadata=pattern.metadata
        )

    def _convert_to_pattern(self, db_pattern: DBPattern) -> Pattern:
        """Convert database Pattern model to Pattern model."""
        return Pattern(
            id=db_pattern.id,
            type=db_pattern.type,
            expression=db_pattern.expression,
            feature_type=db_pattern.feature_type,
            industry=db_pattern.industry,
            version=db_pattern.version,
            created_at=db_pattern.created_at,
            updated_at=db_pattern.updated_at,
            confidence=ConfidenceScore(
                value=db_pattern.confidence_value,
                factors=db_pattern.confidence_factors
            ),
            metadata=db_pattern.pattern_metadata,
            examples=[example.text for example in db_pattern.examples]
        )

    def add_pattern(self, pattern: Pattern) -> Pattern:
        """Add a new pattern to the database."""
        db_pattern = self._convert_to_db_pattern(pattern)

        # Add examples if any
        for example in pattern.examples:
            db_pattern.examples.append(
                PatternExample(text=example)
            )

        self.db.add(db_pattern)
        self.db.commit()
        self.db.refresh(db_pattern)

        return self._convert_to_pattern(db_pattern)

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        db_pattern = self.db.query(DBPattern).filter(DBPattern.id == pattern_id).first()
        return self._convert_to_pattern(db_pattern) if db_pattern else None

    def update_pattern(self, pattern: Pattern) -> Pattern:
        """Update an existing pattern."""
        db_pattern = self.db.query(DBPattern).filter(DBPattern.id == pattern.id).first()
        if not db_pattern:
            raise ValueError(f"Pattern {pattern.id} not found")

        # Update fields
        db_pattern.type = pattern.type
        db_pattern.expression = pattern.expression
        db_pattern.feature_type = pattern.feature_type
        db_pattern.industry = pattern.industry
        db_pattern.version = pattern.version
        db_pattern.updated_at = datetime.utcnow()
        db_pattern.confidence_value = pattern.confidence.value
        db_pattern.confidence_factors = pattern.confidence.factors
        db_pattern.pattern_metadata = pattern.metadata

        # Update examples
        db_pattern.examples = []
        for example in pattern.examples:
            db_pattern.examples.append(PatternExample(text=example))

        self.db.commit()
        self.db.refresh(db_pattern)

        return self._convert_to_pattern(db_pattern)

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern by ID."""
        db_pattern = self.db.query(DBPattern).filter(DBPattern.id == pattern_id).first()
        if db_pattern:
            self.db.delete(db_pattern)
            self.db.commit()
            return True
        return False

    def get_patterns_by_industry(self, industry: str) -> List[Pattern]:
        """Get all patterns for a specific industry."""
        db_patterns = self.db.query(DBPattern).filter(DBPattern.industry == industry).all()
        return [self._convert_to_pattern(p) for p in db_patterns]

    def get_patterns_by_feature(self, feature_type: str) -> List[Pattern]:
        """Get all patterns for a specific feature type."""
        db_patterns = self.db.query(DBPattern).filter(DBPattern.feature_type == feature_type).all()
        return [self._convert_to_pattern(p) for p in db_patterns]

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Pattern]:
        """Get all patterns of a specific type."""
        db_patterns = self.db.query(DBPattern).filter(DBPattern.type == pattern_type).all()
        return [self._convert_to_pattern(p) for p in db_patterns]

    def get_all_patterns(self) -> List[Pattern]:
        """Get all patterns from the database."""
        db_patterns = self.db.query(DBPattern).all()
        return [self._convert_to_pattern(p) for p in db_patterns]

    def add_validation_result(self, pattern_id: str,
                            true_pos: int, false_pos: int,
                            true_neg: int, false_neg: int) -> None:
        """Add validation results for a pattern."""
        validation = PatternValidation(
            pattern_id=pattern_id,
            true_positives=true_pos,
            false_positives=false_pos,
            true_negatives=true_neg,
            false_negatives=false_neg
        )
        self.db.add(validation)
        self.db.commit()

    def get_validation_results(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Get validation history for a pattern."""
        results = self.db.query(PatternValidation)\
            .filter(PatternValidation.pattern_id == pattern_id)\
            .order_by(PatternValidation.created_at.desc())\
            .all()

        return [{
            'true_positives': r.true_positives,
            'false_positives': r.false_positives,
            'true_negatives': r.true_negatives,
            'false_negatives': r.false_negatives,
            'created_at': r.created_at
        } for r in results]