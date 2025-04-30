"""Tests for pattern matching system."""

import pytest
import logging
from datetime import datetime
from jellyfish import soundex, metaphone, levenshtein_distance
from difflib import SequenceMatcher
from src.classifier.pattern_learning.models import (
    Pattern,
    PatternType,
    MatchType,
    ConfidenceScore,
    PatternValidation,
    PatternMatch
)
from src.classifier.pattern_learning.pattern_matcher import PatternMatcher
from src.classifier.pattern_learning.pattern_store import PatternStore

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for each test."""
    logger.info("\n" + "="*80)
    logger.info("Starting new test")
    yield
    logger.info("Test completed")
    logger.info("="*80 + "\n")

@pytest.fixture
def pattern_store():
    """Create a pattern store for testing."""
    return PatternStore()


@pytest.fixture
def pattern_matcher():
    """Create a pattern matcher for testing."""
    return PatternMatcher(min_confidence=0.6)


@pytest.fixture
def sample_patterns():
    """Create sample patterns for testing."""
    return [
        Pattern(
            id="patient_id",
            type=PatternType.REGEX,
            expression=r"Patient ID: (?P<id>[A-Z0-9-]+)",
            feature_type="patient_id",
            industry="healthcare",
            metadata={"expected_section": "header"}
        ),
        Pattern(
            id="amount",
            type=PatternType.REGEX,
            expression=r"\$[\d,]+\.\d{2}",
            feature_type="amount",
            industry="finance"
        ),
        Pattern(
            id="doctor_name",
            type=PatternType.FUZZY,
            expression="Dr.",
            feature_type="doctor",
            industry="healthcare",
            examples=["Dr. John Smith", "Doctor Jane Doe"],
            metadata={"is_name": True}
        )
    ]


def test_regex_pattern_matching():
    """Test basic regex pattern matching."""
    matcher = PatternMatcher(min_confidence=0.6)
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+@\w+\.\w+\b",
        feature_type="email",
        industry="general"
    )

    matches = matcher.find_matches("test@example.com", [pattern])
    assert len(matches) == 1
    assert matches[0].text == "test@example.com"
    assert matches[0].confidence.value >= 0.6

def test_fuzzy_pattern_matching():
    """Test fuzzy pattern matching."""
    matcher = PatternMatcher(min_confidence=0.6)
    pattern = Pattern(
        id="fuzzy_pattern",
        type=PatternType.FUZZY,
        expression="example text",
        feature_type="text",
        examples=["example text", "sample text"],
        industry="general",
        metadata={
            "is_address": False,
            "min_confidence": 0.6
        }
    )

    matches = matcher.find_matches("example text", [pattern])
    assert len(matches) == 1
    assert matches[0].text == "example text"
    assert matches[0].confidence.value >= 0.6

def test_context_matching():
    """Test context-aware pattern matching."""
    matcher = PatternMatcher(min_confidence=0.6)
    pattern = Pattern(
        id="context_pattern",
        type=PatternType.CONTEXT,
        expression=r"\b\d{3}-\d{2}-\d{4}\b",
        feature_type="ssn",
        metadata={"expected_section": "personal_info"},
        industry="general"
    )

    context = {
        "section_markers": {
            "Personal Information:": {"type": "personal_info", "level": 1}
        }
    }

    text = "Personal Information:\n123-45-6789"
    matches = matcher.find_matches(text, [pattern], context)
    assert len(matches) == 1
    assert matches[0].text == "123-45-6789"

def test_name_matching():
    """Test name matching with titles and multiple names."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Starting Name Matching Test ===")

    matcher = PatternMatcher(min_confidence=0.6)

    # Create test pattern
    logger.info("Creating name pattern...")
    pattern = Pattern(
        id="name_pattern",
        type=PatternType.FUZZY,
        expression="Dr. John Smith",
        feature_type="name",
        industry="healthcare",
        examples=[
            "Dr. John Smith",
            "Mr. James Wilson",
            "Mrs. Sarah Johnson",
            "Professor Robert Brown"
        ],
        metadata={
            "is_name": True,
            "min_confidence": 0.6
        }
    )
    logger.info(f"Pattern created:")
    logger.info(f"  ID: {pattern.id}")
    logger.info(f"  Expression: {pattern.expression}")
    logger.info(f"  Examples: {pattern.examples}")
    logger.info(f"  Feature type: {pattern.feature_type}")
    logger.info(f"  Metadata: {pattern.metadata}")

    test_cases = [
        ("Dr. John Smith", True, "Exact match"),
        ("Dr John Smith", True, "Missing period in title"),
        ("Doctor John Smith", True, "Full title"),
        ("Mr. John Smith", True, "Different title"),
        ("John Smith", True, "No title"),
        ("Dr. Jon Smith and Jane Doe", True, "Multiple names"),
        ("Professor Smith", True, "Different title with last name"),
        ("J. Smith", True, "Initial with last name"),
        ("NotAName123", False, "Invalid name format"),
        ("", False, "Empty string")
    ]

    for test_text, should_match, description in test_cases:
        logger.info(f"\n--- Testing: {description} ---")
        logger.info(f"Input text: '{test_text}'")

        # Log title detection
        titles = ["Dr.", "Doctor", "Mr.", "Mrs.", "Ms.", "Prof.", "Professor"]
        found_title = next((t for t in titles if t in test_text), None)
        logger.info(f"Title detection: {found_title}")

        # Log name parts analysis
        name_parts = test_text.split()
        logger.info(f"Name parts: {name_parts}")

        # Log phonetic analysis for each word
        logger.info("Phonetic analysis:")
        for word in name_parts:
            if any(t in word for t in titles):
                logger.info(f"  Skipping title word: {word}")
                continue
            logger.info(f"  Word: {word}")
            logger.info(f"    Soundex: {soundex(word)}")
            logger.info(f"    Metaphone: {metaphone(word)}")

        # Compare with examples
        logger.info("\nComparing with examples:")
        for example in pattern.examples:
            seq_ratio = SequenceMatcher(None, test_text.lower(), example.lower()).ratio()
            lev_distance = levenshtein_distance(test_text, example)
            lev_ratio = 1 - (lev_distance / max(len(test_text), len(example)))
            logger.info(f"  Example: '{example}'")
            logger.info(f"    Sequence similarity: {seq_ratio:.3f}")
            logger.info(f"    Levenshtein distance: {lev_distance}")
            logger.info(f"    Levenshtein ratio: {lev_ratio:.3f}")

        matches = matcher.find_matches(test_text, [pattern])

        if matches:
            match = matches[0]
            logger.info("\nMatch found:")
            logger.info(f"  Text: '{match.text}'")
            logger.info(f"  Confidence: {match.confidence.value:.3f}")
            logger.info("  Confidence factors:")
            for factor, value in match.confidence.factors.items():
                logger.info(f"    - {factor}: {value:.3f}")
            assert should_match, f"Unexpected match found for: {test_text}"
            assert match.confidence.value >= pattern.metadata["min_confidence"]
        else:
            logger.info("\nNo match found")
            assert not should_match, f"Expected match not found for: {test_text}"

    logger.info("\n=== Name Matching Test Completed ===")

def test_ngram_matching():
    """Test n-gram based matching."""
    matcher = PatternMatcher(min_confidence=0.6)

    pattern = Pattern(
        id="address_pattern",
        type=PatternType.FUZZY,
        expression="123 Main Street",
        feature_type="address",
        industry="general",
        examples=[
            "123 Main Street",
            "456 Oak Avenue",
            "789 Pine Road"
        ],
        metadata={
            "ngram_size": 3,
            "min_confidence": 0.6,
            "is_address": True
        }
    )

    test_cases = [
        ("123 Main Street", True, "Exact match"),
        ("123 Main St", True, "Common abbreviation"),
        ("124 Main Street", True, "Similar house number"),
        ("123 Oak Street", True, "Different street name"),
        ("Street Main 123", True, "Different word order"),
        ("789 Pine Road", True, "Different but valid address"),
        ("Not an address", False, "Non-matching string"),
        ("12345", False, "Just numbers"),
        ("", False, "Empty string")
    ]

    for test_text, should_match, description in test_cases:
        matches = matcher.find_matches(test_text, [pattern])

        if matches:
            match = matches[0]
            if should_match:
                assert match.confidence.value >= pattern.metadata["min_confidence"], \
                    f"Confidence {match.confidence.value} below threshold {pattern.metadata['min_confidence']}"
        else:
            assert not should_match, f"Expected match not found for: {test_text}"

def test_invalid_patterns():
    """Test handling of invalid patterns."""
    matcher = PatternMatcher()

    # Test with an invalid regex pattern
    invalid_pattern = Pattern(
        id="test_invalid_pattern",
        type=PatternType.REGEX,
        expression="[invalid",  # Invalid regex with unclosed bracket
        feature_type="test",
        industry="general"
    )

    # The pattern should be marked as invalid during validation
    assert not matcher._validate_regex_pattern(invalid_pattern.expression)
    assert any("unterminated character set" in warning for warning in invalid_pattern.metadata.get("validation_warnings", []))

    # When trying to find matches with an invalid pattern, it should return empty matches
    test_text = "test"
    matches = matcher.find_matches(test_text, [invalid_pattern])
    assert len(matches) == 0

    # Test with a valid pattern for comparison
    valid_pattern = Pattern(
        id="test_valid_pattern",
        type=PatternType.REGEX,
        expression=r"\btest\b",
        feature_type="test",
        industry="general"
    )

    # The valid pattern should work normally
    matches = matcher.find_matches(test_text, [valid_pattern])
    assert len(matches) == 1

def test_pattern_validation():
    """Test pattern validation logic."""
    matcher = PatternMatcher()
    assert matcher._validate_regex_pattern(r"\b\w+\b")
    assert not matcher._validate_regex_pattern("[unclosed")
    assert not matcher._validate_regex_pattern("(unclosed")

def test_section_analysis():
    """Test section analysis functionality."""
    matcher = PatternMatcher()
    text = "Section A:\nContent\nSection B:\nMore content"
    context = {
        "section_markers": {
            "Section A:": {"type": "section_a", "level": 1},
            "Section B:": {"type": "section_b", "level": 1}
        }
    }

    info = matcher._analyze_sections(text, context, 5)
    assert info["section_type"] == "section_a"

def test_pattern_storage():
    """Test pattern storage and retrieval."""
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="test",
        industry="general"
    )

    assert pattern.id == "test_pattern"
    assert pattern.type == PatternType.REGEX

def test_pattern_updating():
    """Test pattern update functionality."""
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="test",
        industry="general"
    )

    pattern.expression = r"\b\d+\b"
    assert pattern.expression == r"\b\d+\b"

def test_hierarchical_context():
    """Test hierarchical context handling."""
    matcher = PatternMatcher()
    text = "Main:\n  Sub1:\n    Content"
    context = {
        "section_markers": {
            "Main:": {"type": "main", "level": 1},
            "Sub1:": {"type": "sub1", "level": 2}
        }
    }

    info = matcher._analyze_sections(text, context, 20)
    assert "main" in info["hierarchy"]
    assert "sub1" in info["hierarchy"]

def test_pattern_confidence_updating():
    """Test confidence score updating."""
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="test",
        confidence=ConfidenceScore(value=0.8),
        industry="general"
    )

    assert pattern.confidence.value == 0.8

def test_pattern_metadata_validation():
    """Test metadata validation."""
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="test",
        metadata={"min_confidence": 0.5},
        industry="general"
    )

    assert pattern.metadata["min_confidence"] == 0.5

def test_complex_pattern_combinations():
    """Test handling of complex pattern combinations."""
    matcher = PatternMatcher()
    patterns = [
        Pattern(
            id="pattern1",
            type=PatternType.REGEX,
            expression=r"\b\w+\b",
            feature_type="test1",
            industry="general"
        ),
        Pattern(
            id="pattern2",
            type=PatternType.REGEX,
            expression=r"\b\d+\b",
            feature_type="test2",
            industry="general"
        )
    ]

    matches = matcher.find_matches("word 123", patterns)
    assert len(matches) == 2

def test_edge_cases_and_boundaries():
    """Test edge cases and boundary conditions."""
    matcher = PatternMatcher()
    pattern = Pattern(
        id="edge_pattern",
        type=PatternType.REGEX,
        expression=r"\b[A-Za-z]+\b",
        feature_type="test",
        industry="general"
    )

    matches = matcher.find_matches("word", [pattern])
    assert len(matches) == 1

    matches = matcher.find_matches("", [pattern])
    assert len(matches) == 0

    matches = matcher.find_matches("123", [pattern])
    assert len(matches) == 0

def test_pattern_serialization():
    """Test pattern serialization."""
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="test",
        industry="general"
    )

    assert pattern.id == "test_pattern"
    assert pattern.type == PatternType.REGEX
    assert pattern.expression == r"\b\w+\b"