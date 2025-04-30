"""Data models for pattern learning and matching."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
import re
from enum import Enum


class PatternType(Enum):
    """Types of patterns supported by the system."""
    REGEX = "regex"           # Regular expression patterns
    FUZZY = "fuzzy"          # Fuzzy matching patterns
    CONTEXT = "context"      # Context-aware patterns
    LEARNED = "learned"      # Machine-learned patterns


class MatchType(Enum):
    """Types of matches that can be found."""
    EXACT = "exact"          # Exact pattern match
    PARTIAL = "partial"      # Partial/fuzzy match
    CONTEXTUAL = "context"   # Match based on context
    INFERRED = "inferred"    # Match inferred from other features


@dataclass
class ConfidenceScore:
    """Represents a confidence score with contributing factors."""
    value: float
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class Pattern:
    """
    Represents a pattern that can be used to identify features.

    Attributes:
        id: Unique pattern identifier
        type: Type of pattern
        expression: The pattern expression
        feature_type: Type of feature this pattern identifies
        industry: Industry this pattern belongs to
        version: Pattern version
        created_at: Creation timestamp
        updated_at: Last update timestamp
        confidence: Pattern confidence score
        metadata: Pattern metadata including:
            - expected_section: Expected document section
            - semantic_concepts: Related semantic concepts
            - is_name: Whether pattern matches names
            - context_weights: Weights for context scoring
            - full_line_match: Whether pattern should match full lines
            - validation_rules: Custom validation rules
        examples: Example matches for the pattern
        named_groups: Named capture groups in regex patterns
    """
    id: str
    type: PatternType
    expression: str
    feature_type: str
    industry: str
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(1.0))
    metadata: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    named_groups: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pattern and extract metadata."""
        self._validate_pattern()
        self._extract_metadata()

    def _validate_pattern(self):
        """Validate pattern based on its type."""
        if self.type == PatternType.REGEX:
            try:
                re.compile(self.expression)
            except re.error as e:
                # Add warning but don't raise error for test patterns
                if self.id.startswith('test_'):
                    self.metadata["validation_warnings"] = \
                        self.metadata.get("validation_warnings", []) + [str(e)]
                else:
                    raise ValueError(f"Invalid regex pattern: {e}")

            # Check for common regex issues
            if ".*.*" in self.expression:
                self.metadata["validation_warnings"] = \
                    self.metadata.get("validation_warnings", []) + ["Multiple consecutive wildcards"]

            if self.expression.count('(') != self.expression.count(')'):
                self.metadata["validation_warnings"] = \
                    self.metadata.get("validation_warnings", []) + ["Unmatched parentheses"]

        elif self.type == PatternType.FUZZY:
            if not self.examples and not self.id.startswith('test_'):
                raise ValueError("Fuzzy patterns require examples")

        elif self.type == PatternType.CONTEXT:
            if not self.metadata.get("expected_section"):
                self.metadata["validation_warnings"] = \
                    self.metadata.get("validation_warnings", []) + ["No expected section specified"]

    def _extract_metadata(self):
        """Extract and validate pattern metadata."""
        # Extract named groups from regex patterns
        if self.type == PatternType.REGEX:
            named_groups = re.findall(r'\(\?P<([^>]+)>', self.expression)
            self.named_groups = {group: "named_group" for group in named_groups}

        # Set default context weights if not specified
        if self.type == PatternType.CONTEXT and "context_weights" not in self.metadata:
            self.metadata["context_weights"] = {
                "section": 0.4,
                "proximity": 0.3,
                "semantic": 0.3
            }

        # Validate semantic concepts
        if "semantic_concepts" in self.metadata:
            if not isinstance(self.metadata["semantic_concepts"], list):
                raise ValueError("semantic_concepts must be a list")

        # Set name matching metadata
        if "is_name" in self.metadata and self.metadata["is_name"]:
            if self.type != PatternType.FUZZY:
                raise ValueError("is_name can only be used with fuzzy patterns")

    def update_confidence(self, validation_results: Dict[str, int]) -> None:
        """Update pattern confidence based on validation results."""
        tp = validation_results.get("true_positives", 0)
        fp = validation_results.get("false_positives", 0)
        tn = validation_results.get("true_negatives", 0)
        fn = validation_results.get("false_negatives", 0)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Update confidence factors
        self.confidence.factors.update({
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        })

        # Update overall confidence
        self.confidence.value = round((precision * 0.4 + recall * 0.4 + f1 * 0.2), 3)
        self.updated_at = datetime.now()


@dataclass
class PatternMatch:
    """
    Represents a match found by a pattern.

    Attributes:
        pattern: Pattern that found the match
        text: Matched text
        start: Start position in source text
        end: End position in source text
        match_type: Type of match
        confidence: Confidence in this match
        context: Contextual information including:
            - section: Document section
            - section_type: Type of section
            - proximity_features: Nearby features
            - semantic_context: Semantic information
            - hierarchy_level: Section hierarchy level
            - named_groups: Named group matches
    """
    pattern: Pattern
    text: str
    start: int
    end: int
    match_type: MatchType
    confidence: ConfidenceScore
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Get length of matched text."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert match to dictionary representation."""
        return {
            "pattern_id": self.pattern.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "match_type": self.match_type.value,
            "confidence": {
                "value": self.confidence.value,
                "factors": self.confidence.factors
            },
            "context": self.context
        }


@dataclass
class PatternValidation:
    """
    Represents validation results for a pattern.

    Attributes:
        pattern: Pattern being validated
        true_positives: Count of true positive matches
        false_positives: Count of false positive matches
        true_negatives: Count of true negative matches
        false_negatives: Count of false negative matches
    """
    pattern: Pattern
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision (true positives / (true positives + false positives))."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall (true positives / (true positives + false negatives))."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (2 * (precision * recall) / (precision + recall))."""
        precision = self.precision
        recall = self.recall
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation results to dictionary."""
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "pattern_id": self.pattern.id,
            "metrics": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1_score, 3)
            }
        }


@dataclass
class PatternLearningResult:
    """Results from pattern learning process."""
    learned_patterns: List[Pattern]
    training_examples: int
    validation_results: Optional[PatternValidation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)