"""Pattern storage and management system."""

from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import logging
from .models import Pattern, PatternType, ConfidenceScore
from .db_service import PatternDBService

logger = logging.getLogger(__name__)

class PatternStore:
    """
    Manages storage and retrieval of patterns.

    Features:
    - Pattern storage and retrieval
    - Pattern versioning
    - Industry-specific pattern collections
    - Pattern metadata and statistics
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize pattern store.

        Args:
            storage_path: Optional path to save/load patterns (kept for backward compatibility)
        """
        self._db = PatternDBService()
        self._patterns: Dict[str, Pattern] = {}  # Cache for frequently accessed patterns

        # Load patterns from file if storage path is provided (for migration purposes)
        if storage_path and storage_path.exists():
            self._migrate_from_json(storage_path)

    def _migrate_from_json(self, json_path: Path) -> None:
        """Migrate patterns from JSON file to database."""
        try:
            # Load patterns from JSON
            from json import load
            with open(json_path) as f:
                patterns_dict = load(f)

            # Migrate each pattern to database
            for pattern_data in patterns_dict.values():
                pattern = Pattern(
                    id=pattern_data["id"],
                    type=PatternType(pattern_data["type"]),
                    expression=pattern_data["expression"],
                    feature_type=pattern_data["feature_type"],
                    industry=pattern_data["industry"],
                    version=pattern_data["version"],
                    created_at=datetime.fromisoformat(pattern_data["created_at"]),
                    updated_at=datetime.fromisoformat(pattern_data["updated_at"]),
                    confidence=ConfidenceScore(
                        value=pattern_data["confidence"]["value"],
                        factors=pattern_data["confidence"]["factors"]
                    ),
                    metadata=pattern_data["metadata"],
                    examples=pattern_data["examples"]
                )
                self.add_pattern(pattern)

            logger.info(f"Migrated {len(patterns_dict)} patterns from JSON to database")

        except Exception as e:
            logger.error(f"Error migrating patterns from JSON: {e}")
            raise

    def add_pattern(self, pattern: Pattern) -> None:
        """
        Add a new pattern to the store.

        Args:
            pattern: Pattern to add
        """
        if pattern.id in self._patterns:
            logger.warning(f"Pattern {pattern.id} already exists")
            self.update_pattern(pattern)
        else:
            self._db.add_pattern(pattern)
            self._patterns[pattern.id] = pattern
            logger.info(f"Added pattern {pattern.id} for {pattern.feature_type}")

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Retrieve a pattern by ID.

        Args:
            pattern_id: ID of pattern to retrieve

        Returns:
            Pattern if found, None otherwise
        """
        # Check cache first
        if pattern_id in self._patterns:
            return self._patterns[pattern_id]

        # If not in cache, get from database
        pattern = self._db.get_pattern(pattern_id)
        if pattern:
            self._patterns[pattern_id] = pattern
        return pattern

    def update_pattern(self, pattern: Pattern) -> None:
        """
        Update an existing pattern.

        Args:
            pattern: Pattern to update
        """
        self._db.update_pattern(pattern)
        self._patterns[pattern.id] = pattern
        logger.info(f"Updated pattern {pattern.id}")

    def get_patterns_by_industry(self, industry: str) -> List[Pattern]:
        """
        Get all patterns for a specific industry.

        Args:
            industry: Industry to get patterns for

        Returns:
            List of patterns for the industry
        """
        patterns = self._db.get_patterns_by_industry(industry)
        # Update cache
        for pattern in patterns:
            self._patterns[pattern.id] = pattern
        return patterns

    def get_patterns_by_feature(self, feature_type: str) -> List[Pattern]:
        """
        Get all patterns for a specific feature type.

        Args:
            feature_type: Feature type to get patterns for

        Returns:
            List of patterns for the feature type
        """
        patterns = self._db.get_patterns_by_feature(feature_type)
        # Update cache
        for pattern in patterns:
            self._patterns[pattern.id] = pattern
        return patterns

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Pattern]:
        """
        Get all patterns of a specific type.

        Args:
            pattern_type: Type of patterns to get

        Returns:
            List of patterns of the specified type
        """
        patterns = self._db.get_patterns_by_type(pattern_type)
        # Update cache
        for pattern in patterns:
            self._patterns[pattern.id] = pattern
        return patterns

    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a pattern from the store.

        Args:
            pattern_id: ID of pattern to delete

        Returns:
            True if pattern was deleted, False if not found
        """
        if self._db.delete_pattern(pattern_id):
            self._patterns.pop(pattern_id, None)
            logger.info(f"Deleted pattern {pattern_id}")
            return True
        return False

    def get_industries(self) -> Set[str]:
        """
        Get all industries that have patterns.

        Returns:
            Set of industry names
        """
        patterns = self._db.get_all_patterns()
        return {p.industry for p in patterns}

    def get_feature_types(self, industry: Optional[str] = None) -> Set[str]:
        """
        Get all feature types, optionally filtered by industry.

        Args:
            industry: Optional industry to filter by

        Returns:
            Set of feature type names
        """
        if industry:
            patterns = self._db.get_patterns_by_industry(industry)
        else:
            patterns = self._db.get_all_patterns()
        return {p.feature_type for p in patterns}

    def get_all_patterns(self) -> List[Pattern]:
        """
        Get all patterns from the store.

        Returns:
            List of all patterns
        """
        patterns = self._db.get_all_patterns()
        # Update cache
        for pattern in patterns:
            self._patterns[pattern.id] = pattern
        return patterns

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, '_db'):
            self._db.close()