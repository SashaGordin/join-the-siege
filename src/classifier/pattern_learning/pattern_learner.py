"""Pattern learning module for extracting and validating new patterns from classified documents."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, UTC
import re
import uuid

from .models import Pattern, PatternType, PatternMatch, ConfidenceScore
from ..content_classifier import ClassificationResult
from .pattern_store import PatternStore
from .pattern_validator import PatternValidator

logger = logging.getLogger(__name__)

@dataclass
class LearningCandidate:
    """Represents a potential pattern identified during learning."""
    text: str
    feature_type: str
    pattern_type: PatternType
    confidence: float
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any]

class PatternLearner:
    """Learns new patterns from successfully classified documents."""

    def __init__(
        self,
        pattern_store: PatternStore,
        pattern_validator: Optional[PatternValidator] = None,
        min_confidence: float = 0.8,
        min_occurrences: int = 3
    ):
        """
        Initialize the pattern learner.

        Args:
            pattern_store: Storage for learned patterns
            pattern_validator: Optional validator for new patterns
            min_confidence: Minimum confidence threshold for learning
            min_occurrences: Minimum occurrences before considering a pattern
        """
        self.pattern_store = pattern_store
        self.pattern_validator = pattern_validator or PatternValidator()
        self.min_confidence = min_confidence
        self.min_occurrences = min_occurrences
        self._candidates: Dict[str, LearningCandidate] = {}

    def learn_from_classification(
        self,
        text: str,
        llm_result: ClassificationResult,
        pattern_matches: List[PatternMatch],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Pattern]:
        """
        Learn new patterns from a successful classification.

        Args:
            text: The document text that was classified
            llm_result: The LLM classification result
            pattern_matches: Existing pattern matches found
            metadata: Additional context metadata

        Returns:
            List of newly learned patterns
        """
        if llm_result.confidence < self.min_confidence:
            logger.debug(
                f"Skipping pattern learning due to low confidence: {llm_result.confidence}"
            )
            return []

        # Extract potential patterns from features
        candidates = self._extract_candidates(text, llm_result, metadata)

        # Update candidate statistics
        self._update_candidates(candidates)

        # Validate mature candidates
        new_patterns = self._validate_candidates()

        # Store new patterns
        if new_patterns:
            for pattern in new_patterns:
                self.pattern_store.add_pattern(pattern)

        return new_patterns

    def _extract_candidates(
        self,
        text: str,
        llm_result: ClassificationResult,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[LearningCandidate]:
        """Extract pattern candidates from classification results."""
        candidates = []
        now = datetime.now(UTC)

        for feature in llm_result.features:
            if not feature.get("present") or not feature.get("values"):
                continue

            feature_type = feature["type"]
            for value in feature["values"]:
                # Skip if value is too short or not meaningful
                if len(str(value)) < 3:
                    continue

                # For predefined patterns, validate the value
                if feature_type in self.pattern_validator.feature_patterns:
                    pattern_str = self.pattern_validator.feature_patterns[feature_type]
                    if not re.match(pattern_str, str(value)):
                        continue

                # Create candidate
                candidate = LearningCandidate(
                    text=str(value),
                    feature_type=feature_type,
                    pattern_type=PatternType.REGEX,
                    confidence=llm_result.confidence,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                    metadata={
                        "doc_type": llm_result.doc_type,
                        "industry": metadata.get("industry") if metadata else None
                    }
                )
                candidates.append(candidate)

        return candidates

    def _update_candidates(self, new_candidates: List[LearningCandidate]):
        """Update statistics for pattern candidates."""
        now = datetime.now(UTC)

        for candidate in new_candidates:
            key = f"{candidate.feature_type}:{candidate.text}"

            if key in self._candidates:
                existing = self._candidates[key]
                existing.occurrences += 1
                existing.last_seen = now
                existing.confidence = (
                    existing.confidence * 0.7 + candidate.confidence * 0.3
                )
            else:
                self._candidates[key] = candidate

    def _validate_candidates(self) -> List[Pattern]:
        """Validate mature candidates and convert them to patterns."""
        new_patterns = []

        # Find candidates ready for validation
        mature_candidates = {
            key: candidate
            for key, candidate in self._candidates.items()
            if (
                candidate.occurrences >= self.min_occurrences
                and candidate.confidence >= self.min_confidence
            )
        }

        for key, candidate in mature_candidates.items():
            try:
                pattern = None
                # For predefined patterns, use them directly
                if candidate.feature_type in self.pattern_validator.feature_patterns:
                    pattern_str = self.pattern_validator.feature_patterns[candidate.feature_type]
                    # Verify the text matches the predefined pattern
                    if re.match(pattern_str, candidate.text):
                        pattern = Pattern(
                            id=str(uuid.uuid4()),
                            type=PatternType.REGEX,
                            expression=pattern_str,
                            feature_type=candidate.feature_type,
                            industry=candidate.metadata.get("industry")
                        )
                else:
                    # For custom patterns, generate and validate
                    pattern = self.pattern_validator.generate_pattern(
                        candidate.text,
                        candidate.feature_type,
                        candidate.metadata
                    )

                if pattern:
                    new_patterns.append(pattern)
                    del self._candidates[key]
                    logger.debug(
                        f"Generated pattern for {candidate.feature_type}: {pattern.expression}"
                    )

            except Exception as e:
                logger.error(f"Error validating pattern candidate: {str(e)}")
                continue

        if new_patterns:
            logger.info(f"Learned {len(new_patterns)} new patterns")

        return new_patterns

    def clear_stale_candidates(self, max_age_days: int = 30):
        """Remove candidates that haven't been seen in a while."""
        now = datetime.now(UTC)
        stale_keys = [
            key
            for key, candidate in self._candidates.items()
            if (now - candidate.last_seen).days > max_age_days
        ]

        for key in stale_keys:
            del self._candidates[key]