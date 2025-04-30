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
from src.classifier.pattern_learning.pattern_store import PatternStore
import re
import time
import json

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
def pattern_matcher(cached_pattern_matcher):
    """Get a pattern matcher instance."""
    return cached_pattern_matcher

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

def test_regex_pattern_matching(pattern_matcher):
    """Test basic regex pattern matching."""
    pattern = Pattern(
        id="test_pattern",
        type=PatternType.REGEX,
        expression=r"\b\w+@\w+\.\w+\b",
        feature_type="email",
        industry="general"
    )

    matches = pattern_matcher.find_matches("test@example.com", [pattern])
    assert len(matches) == 1
    assert matches[0].text == "test@example.com"
    assert matches[0].confidence.value >= 0.6

def test_fuzzy_pattern_matching(pattern_matcher):
    """Test fuzzy pattern matching."""
    pattern = Pattern(
        id="fuzzy_pattern",
        type=PatternType.FUZZY,
        expression="test",
        feature_type="text",
        examples=["test", "text"],
        metadata={"min_confidence": 0.6},
        industry="general"
    )

    test_cases = [
        ("test", True),
        ("text", True),
        ("testing", True),
        ("completely different", False),
        ("", False)
    ]

    for test_text, should_match in test_cases:
        matches = pattern_matcher.find_matches(test_text, [pattern])
        if should_match:
            assert len(matches) > 0
            assert matches[0].confidence.value >= 0.6
        else:
            assert len(matches) == 0

def test_context_matching(pattern_matcher):
    """Test context-aware pattern matching."""
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
    matches = pattern_matcher.find_matches(text, [pattern], context)
    assert len(matches) == 1
    assert matches[0].text == "123-45-6789"

def test_name_matching(pattern_matcher, caplog):
    """Test name matching with fuzzy matching."""
    caplog.set_level(logging.INFO)
    logger.info("\n=== Starting Name Matching Test ===")

    # Create a pattern for name matching
    logger.info("Creating name matching pattern...")
    pattern = Pattern(
        id="test_name_pattern",  # Changed from name to id
        type=PatternType.FUZZY,  # Changed from pattern_type to type
        expression="John Smith",
        examples=["John Smith", "Dr. John Smith", "Mr. Smith"],
        feature_type="name",  # Added feature_type
        industry="general",   # Added industry
        metadata={
            "is_name": True,  # Set name pattern flag
            "min_confidence": 0.6
        }
    )
    logger.info(f"Created pattern with ID: {pattern.id}")
    logger.info(f"Pattern details: {pattern.__dict__}")

    # Test exact match
    logger.info("\nTesting exact match: 'John Smith'")
    matches = pattern_matcher.find_matches("John Smith", [pattern])
    logger.info(f"Found {len(matches)} matches")
    for idx, match in enumerate(matches):
        logger.info(f"Match {idx + 1}:")
        logger.info(f"  Text: '{match.text}'")
        logger.info(f"  Confidence: {match.confidence.value}")
        logger.info(f"  Confidence factors: {match.confidence.factors}")
        logger.info(f"  Position: {match.start} to {match.end}")
    assert len(matches) == 1, f"Expected 1 match for 'John Smith', got {len(matches)}"
    assert matches[0].text == "John Smith"
    assert matches[0].confidence.value >= 0.9  # High confidence for exact match
    logger.info("Exact match test passed")

    # Test partial match
    logger.info("\nTesting partial match: 'Mr. John Smith Jr.'")
    matches = pattern_matcher.find_matches("Mr. John Smith Jr.", [pattern])
    logger.info(f"Found {len(matches)} matches")
    for idx, match in enumerate(matches):
        logger.info(f"Match {idx + 1}:")
        logger.info(f"  Text: '{match.text}'")
        logger.info(f"  Confidence: {match.confidence.value}")
        logger.info(f"  Confidence factors: {match.confidence.factors}")
        logger.info(f"  Position: {match.start} to {match.end}")
    assert len(matches) > 0, "Expected at least one match for 'Mr. John Smith Jr.'"
    assert any("John Smith" in m.text for m in matches), "Expected to find 'John Smith' within the matched text"
    assert matches[0].confidence.value >= 0.8  # High confidence for partial match
    logger.info("Partial match test passed")

    # Test no match
    logger.info("\nTesting no match case: 'Jane Doe'")
    matches = pattern_matcher.find_matches("Jane Doe", [pattern])
    logger.info(f"Found {len(matches)} matches (expected 0)")
    assert len(matches) == 0, "Expected no matches for 'Jane Doe'"
    logger.info("No match test passed")

    logger.info("\n=== Name Matching Test Completed ===")

def test_ngram_matching(pattern_matcher):
    """Test n-gram based matching with detailed logging."""
    logger.info("\n=== Starting N-gram Matching Test ===")

    pattern = Pattern(
        id="address_pattern",
        type=PatternType.FUZZY,
        expression="123 Main Street",
        feature_type="address",
        examples=["123 Main St", "123 Main Street"],  # Only use examples of the same address
        metadata={"is_address": True},
        industry="general"
    )
    logger.info(f"Created test pattern: {pattern.to_dict()}")

    test_cases = [
        ("123 Main Street", True, "Exact match"),
        ("123 Main St", True, "Abbreviated street type"),
        ("Main Street 123", True, "Different word order"),
        ("456 Oak Ave", False, "Different address"),
        ("123 Oak Street", False, "Same number, different street"),
        ("456 Main Street", False, "Same street, different number"),
        ("", False, "Empty string")
    ]

    for test_text, should_match, description in test_cases:
        logger.info(f"\nTesting case: {description}")
        logger.info(f"Input text: '{test_text}'")

        # Log address parsing
        logger.info("Address parsing:")
        parts = test_text.split()
        logger.info(f"  Parts: {parts}")
        if parts:
            number = next((p for p in parts if p.isdigit()), None)
            logger.info(f"  Number component: {number}")
            if number:
                remaining = parts[:parts.index(number)] + parts[parts.index(number)+1:]
                logger.info(f"  Remaining parts: {remaining}")
                street_type = next((p for p in remaining if p.lower() in {
                    'st', 'street', 'ave', 'avenue', 'rd', 'road'
                }), None)
                logger.info(f"  Street type: {street_type}")
                if street_type:
                    street_name = ' '.join(p for p in remaining if p != street_type)
                    logger.info(f"  Street name: {street_name}")

        matches = pattern_matcher.find_matches(test_text, [pattern])
        logger.info(f"Number of matches found: {len(matches)}")

        for idx, match in enumerate(matches):
            logger.info(f"Match {idx + 1}:")
            logger.info(f"  Matched text: '{match.text}'")
            logger.info(f"  Confidence: {match.confidence.value}")
            logger.info(f"  Confidence factors: {match.confidence.factors}")
            logger.info(f"  Match type: {match.match_type}")

        if should_match:
            assert len(matches) > 0, f"Expected match for '{test_text}' but found none"
            assert matches[0].confidence.value >= 0.6, \
                f"Low confidence ({matches[0].confidence.value}) for '{test_text}'"
        else:
            assert len(matches) == 0, \
                f"Unexpected match found for '{test_text}': {[m.text for m in matches]}"

        logger.info(f"Test case {'PASSED' if (len(matches) > 0) == should_match else 'FAILED'}")

    logger.info("\n=== N-gram Matching Test Completed ===")

def test_invalid_patterns(pattern_matcher):
    """Test handling of invalid patterns."""
    invalid_patterns = [
        Pattern(
            id="invalid_regex",
            type=PatternType.REGEX,
            expression="[invalid",
            feature_type="test",
            industry="general"
        ),
        Pattern(
            id="empty_pattern",
            type=PatternType.REGEX,
            expression="",
            feature_type="test",
            industry="general"
        )
    ]

    for pattern in invalid_patterns:
        matches = pattern_matcher.find_matches("test text", [pattern])
        assert len(matches) == 0

def test_pattern_validation(pattern_matcher):
    """Test pattern validation logic."""
    pattern = Pattern(
        id="test_validation",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="word",
        industry="general"
    )

    matches = pattern_matcher.find_matches("test", [pattern])
    assert len(matches) == 1

def test_section_analysis(pattern_matcher):
    """Test section analysis functionality."""
    pattern = Pattern(
        id="section_test",
        type=PatternType.CONTEXT,
        expression=r"\b\w+\b",
        feature_type="word",
        metadata={"expected_section": "summary"},
        industry="general"
    )

    context = {
        "section_markers": {
            "Summary:": {"type": "summary", "level": 1}
        }
    }

    text = "Summary:\ntest word"
    matches = pattern_matcher.find_matches(text, [pattern], context)
    assert len(matches) > 0

def test_hierarchical_context(pattern_matcher):
    """Test hierarchical context handling."""
    pattern = Pattern(
        id="hierarchical_test",
        type=PatternType.CONTEXT,
        expression=r"\b\w+\b",
        feature_type="word",
        metadata={"expected_section": "subsection"},
        industry="general"
    )

    context = {
        "section_markers": {
            "Main:": {"type": "main", "level": 1},
            "Sub:": {"type": "subsection", "level": 2}
        }
    }

    text = "Main:\nSub:\ntest word"
    matches = pattern_matcher.find_matches(text, [pattern], context)
    assert len(matches) > 0

def test_complex_pattern_combinations(pattern_matcher):
    """Test handling of complex pattern combinations."""
    patterns = [
        Pattern(
            id="pattern1",
            type=PatternType.REGEX,
            expression=r"\b\d+\b",
            feature_type="number",
            industry="general"
        ),
        Pattern(
            id="pattern2",
            type=PatternType.FUZZY,
            expression="test",
            feature_type="text",
            examples=["test", "text"],
            industry="general"
        )
    ]

    text = "test 123 text"
    matches = pattern_matcher.find_matches(text, patterns)
    assert len(matches) >= 2

def test_edge_cases_and_boundaries(pattern_matcher):
    """Test edge cases and boundary conditions."""
    pattern = Pattern(
        id="edge_test",
        type=PatternType.REGEX,
        expression=r"\b\w+\b",
        feature_type="word",
        industry="general"
    )

    test_cases = [
        "",  # Empty string
        " ",  # Whitespace only
        "!@#",  # Special characters only
        "a",  # Single character
        "a "*1000  # Very long input
    ]

    for test_text in test_cases:
        matches = pattern_matcher.find_matches(test_text, [pattern])
        # Just ensure it doesn't crash
        assert isinstance(matches, list)

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

def test_caching(pattern_matcher, caplog, mock_cache_service):
    """Test that pattern matching results are properly cached."""
    caplog.set_level(logging.INFO)
    logger.info("\n=== Starting Cache Testing ===")

    # Inject mock cache service into pattern matcher
    pattern_matcher.cache_service = mock_cache_service
    logger.info("Injected mock cache service into pattern matcher")

    # Create a test pattern
    pattern = Pattern(
        id="cache_test_pattern",
        type=PatternType.FUZZY,
        expression="test pattern",
        examples=["test pattern", "test"],
        feature_type="test",
        industry="general",
        metadata={"min_confidence": 0.6}
    )
    logger.info("Created test pattern")

    # Mock the find_matches method to add delays
    original_find_matches = pattern_matcher._find_fuzzy_matches
    def delayed_find_matches(*args, **kwargs):
        time.sleep(0.1)  # Add 100ms delay
        return original_find_matches(*args, **kwargs)
    pattern_matcher._find_fuzzy_matches = delayed_find_matches

    # First call - should not be cached
    logger.info("\nFirst call - should miss cache")
    mock_cache_service.clear_cache()  # Changed from clear() to clear_cache()
    start_time = datetime.now()
    first_matches = pattern_matcher.find_matches("test pattern", [pattern])
    first_call_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"First call took {first_call_time:.4f} seconds")
    logger.info(f"First call matches: {[m.text for m in first_matches]}")
    assert len(first_matches) > 0, "Should find matches on first call"

    # Mock Redis get to return cached result for second call
    mock_cache_service.redis_client.get.return_value = json.dumps([m.to_dict() for m in first_matches])
    logger.info("Mocked Redis to return cached result")

    # Second call with same input - should be cached
    logger.info("\nSecond call - should hit cache")
    start_time = datetime.now()
    second_matches = pattern_matcher.find_matches("test pattern", [pattern])
    second_call_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Second call took {second_call_time:.4f} seconds")
    logger.info(f"Second call matches: {[m.text for m in second_matches]}")

    # Verify matches are the same
    assert len(second_matches) == len(first_matches), "Cached result should have same number of matches"
    assert [m.text for m in second_matches] == [m.text for m in first_matches], "Cached result should have same matches"

    # Reset mock for different input
    mock_cache_service.redis_client.get.return_value = None
    logger.info("Reset Redis mock for different input")

    # Different input - should not use cache
    logger.info("\nThird call with different input - should miss cache")
    start_time = datetime.now()
    different_matches = pattern_matcher.find_matches("different text", [pattern])
    different_call_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Different input call took {different_call_time:.4f} seconds")
    logger.info(f"Different input matches: {[m.text for m in different_matches]}")

    # Clear cache and verify cache miss
    logger.info("\nFourth call after cache clear - should miss cache")
    mock_cache_service.clear_cache()  # Changed from clear() to clear_cache()
    mock_cache_service.redis_client.get.return_value = None  # Ensure cache miss
    start_time = datetime.now()
    cleared_matches = pattern_matcher.find_matches("test pattern", [pattern])
    cleared_call_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Post-clear call took {cleared_call_time:.4f} seconds")
    logger.info(f"Post-clear matches: {[m.text for m in cleared_matches]}")

    # Verify cache behavior through timing
    assert second_call_time < first_call_time, "Cached call should be faster than initial call"
    assert different_call_time > second_call_time, "Cache miss should be slower than cache hit"
    assert cleared_call_time > second_call_time, "Cache miss after clear should be slower than cache hit"

    logger.info("\n=== Cache Testing Completed ===")