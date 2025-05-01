"""Tests for pattern learning functionality."""

import re
import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch

from src.classifier.pattern_learning.pattern_learner import (
    PatternLearner,
    LearningCandidate
)
from src.classifier.pattern_learning.pattern_validator import PatternValidator
from src.classifier.pattern_learning.models import (
    Pattern,
    PatternType,
    PatternMatch,
    ConfidenceScore
)
from src.classifier.content_classifier import ClassificationResult

@pytest.fixture
def pattern_store():
    """Create a mock pattern store."""
    return Mock()

@pytest.fixture
def pattern_validator():
    """Create a pattern validator instance."""
    return PatternValidator()

@pytest.fixture
def pattern_learner(pattern_store, pattern_validator):
    """Create a pattern learner instance."""
    return PatternLearner(
        pattern_store=pattern_store,
        pattern_validator=pattern_validator
    )

@pytest.fixture
def sample_llm_result():
    """Create a sample LLM classification result."""
    return ClassificationResult(
        doc_type="invoice",
        confidence=0.9,
        features=[
            {
                "type": "amount",
                "values": ["$100.00"],
                "present": True
            },
            {
                "type": "date",
                "values": ["2024-03-15"],
                "present": True
            }
        ]
    )

class TestPatternValidator:
    """Tests for the PatternValidator class."""

    def test_generate_pattern_predefined(self, pattern_validator):
        """Test generating pattern for predefined feature type."""
        pattern = pattern_validator.generate_pattern(
            "$100.00",
            "amount",
            {"industry": "financial"}
        )
        assert pattern is not None
        assert pattern.type == PatternType.REGEX
        assert pattern.feature_type == "amount"
        assert pattern.industry == "financial"

        # Test matching
        assert re.match(pattern.expression, "$100.00")
        assert re.match(pattern.expression, "$50.00")
        assert not re.match(pattern.expression, "invalid")

    def test_generate_pattern_custom(self, pattern_validator):
        """Test generating custom pattern."""
        pattern = pattern_validator.generate_pattern(
            "INV-2024-001",
            "invoice_number",
            {"industry": "retail"}
        )
        assert pattern is not None
        assert pattern.type == PatternType.REGEX
        assert pattern.feature_type == "invoice_number"
        assert re.match(pattern.expression, "INV-2024-001")

    def test_validate_pattern(self, pattern_validator):
        """Test pattern validation."""
        assert pattern_validator._validate_pattern(r"\d+", "123")
        assert not pattern_validator._validate_pattern(".*", "test")
        assert not pattern_validator._validate_pattern(r"\d+", "abc")

    @pytest.mark.parametrize("feature_type,text,should_match", [
        ("amount", "$100.00", True),
        ("amount", "invalid", False),
        ("date", "2024-03-15", True),
        ("date", "15/03/2024", True),
        ("date", "invalid", False),
        ("email", "test@example.com", True),
        ("email", "invalid", False),
        ("phone", "+1 (555) 123-4567", True),
        ("phone", "invalid", False),
        ("url", "https://example.com", True),
        ("url", "invalid", False)
    ])
    def test_predefined_patterns(self, pattern_validator, feature_type, text, should_match):
        """Test predefined patterns for different feature types."""
        pattern = pattern_validator.generate_pattern(text, feature_type)
        if should_match:
            assert pattern is not None
            assert pattern.feature_type == feature_type
        else:
            assert pattern is None

class TestPatternLearner:
    """Tests for the PatternLearner class."""

    def test_learn_from_classification(
        self,
        pattern_learner,
        sample_llm_result,
        pattern_store
    ):
        """Test learning patterns from classification."""
        metadata = {"industry": "financial"}
        pattern_matches = []  # No existing matches

        # First learning attempt
        patterns = pattern_learner.learn_from_classification(
            "Invoice for $100.00 dated 2024-03-15",
            sample_llm_result,
            pattern_matches,
            metadata
        )

        # Should not learn yet (min_occurrences not met)
        assert not patterns
        assert len(pattern_learner._candidates) == 2  # Amount and date

        # Get the initial candidates
        amount_key = "amount:$100.00"
        date_key = "date:2024-03-15"

        assert amount_key in pattern_learner._candidates
        assert date_key in pattern_learner._candidates

        # Simulate multiple occurrences by directly updating occurrences
        pattern_learner._candidates[amount_key].occurrences = pattern_learner.min_occurrences
        pattern_learner._candidates[date_key].occurrences = pattern_learner.min_occurrences

        # One more classification should trigger pattern learning
        patterns = pattern_learner.learn_from_classification(
            "Invoice for $100.00 dated 2024-03-15",
            sample_llm_result,
            pattern_matches,
            metadata
        )

        # Should learn patterns now
        assert patterns
        assert len(patterns) == 2  # Amount and date patterns

        # Verify the patterns
        patterns_by_type = {p.feature_type: p for p in patterns}
        assert "amount" in patterns_by_type
        assert "date" in patterns_by_type

        # Verify pattern store was called
        pattern_store.add_pattern.assert_called()
        assert pattern_store.add_pattern.call_count == 2

    def test_skip_low_confidence(self, pattern_learner):
        """Test skipping pattern learning for low confidence results."""
        low_confidence_result = ClassificationResult(
            doc_type="invoice",
            confidence=0.5,  # Below minimum threshold
            features=[
                {
                    "type": "amount",
                    "values": ["$100.00"],
                    "present": True
                }
            ]
        )

        patterns = pattern_learner.learn_from_classification(
            "text",
            low_confidence_result,
            [],
            {}
        )
        assert not patterns
        assert not pattern_learner._candidates

    def test_clear_stale_candidates(self, pattern_learner):
        """Test clearing stale pattern candidates."""
        now = datetime.now(UTC)
        old_date = now - timedelta(days=31)

        # Add some candidates
        pattern_learner._candidates = {
            "fresh": LearningCandidate(
                text="fresh",
                feature_type="test",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                occurrences=1,
                first_seen=now,
                last_seen=now,
                metadata={}
            ),
            "stale": LearningCandidate(
                text="stale",
                feature_type="test",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                occurrences=1,
                first_seen=old_date,
                last_seen=old_date,
                metadata={}
            )
        }

        pattern_learner.clear_stale_candidates(max_age_days=30)
        assert len(pattern_learner._candidates) == 1
        assert "fresh" in pattern_learner._candidates
        assert "stale" not in pattern_learner._candidates

    def test_update_candidates(self, pattern_learner):
        """Test updating candidate statistics."""
        now = datetime.now(UTC)
        candidate = LearningCandidate(
            text="test",
            feature_type="type",
            pattern_type=PatternType.REGEX,
            confidence=0.8,
            occurrences=1,
            first_seen=now,
            last_seen=now,
            metadata={}
        )

        # First update
        pattern_learner._update_candidates([candidate])
        assert len(pattern_learner._candidates) == 1
        first_confidence = pattern_learner._candidates["type:test"].confidence

        # Second update with higher confidence
        candidate.confidence = 1.0
        pattern_learner._update_candidates([candidate])
        assert pattern_learner._candidates["type:test"].occurrences == 2
        assert pattern_learner._candidates["type:test"].confidence > first_confidence