"""Tests for hybrid document classifier."""

import pytest
from unittest.mock import patch, MagicMock
from src.classifier.hybrid_classifier import HybridClassifier, HybridClassificationResult
from src.classifier.content_classifier import ClassificationResult
from src.classifier.pattern_learning.models import (
    Pattern,
    PatternMatch,
    PatternType,
    MatchType,
    ConfidenceScore
)

@pytest.fixture
def hybrid_classifier():
    """Create a hybrid classifier instance."""
    return HybridClassifier()

@pytest.fixture
def mock_llm_result():
    """Create a mock LLM classification result."""
    return ClassificationResult(
        doc_type="invoice",
        confidence=0.8,
        features=[
            {
                "type": "amount",
                "values": ["$100.00"],
                "present": True
            },
            {
                "type": "date",
                "values": ["2024-01-01"],
                "present": True
            }
        ]
    )

@pytest.fixture
def mock_pattern_matches():
    """Create mock pattern matches."""
    pattern = Pattern(
        id="test_amount",
        type=PatternType.REGEX,
        expression=r"\$\d+\.\d{2}",
        feature_type="amount",
        industry="financial"
    )

    return [
        PatternMatch(
            pattern=pattern,
            text="$100.00",
            start=0,
            end=7,
            match_type=MatchType.EXACT,
            confidence=ConfidenceScore(0.9)
        )
    ]

def test_hybrid_classifier_initialization():
    """Test HybridClassifier initialization."""
    classifier = HybridClassifier()
    assert classifier.content_classifier is not None
    assert classifier.pattern_matcher is not None
    assert classifier.pattern_store is not None

def test_classify_document_basic(hybrid_classifier, mock_llm_result, mock_pattern_matches):
    """Test basic document classification."""
    # Mock the LLM classification
    with patch.object(
        hybrid_classifier.content_classifier,
        '_classify_with_llm',
        return_value=mock_llm_result
    ), patch.object(
        hybrid_classifier.pattern_matcher,
        'find_matches',
        return_value=mock_pattern_matches
    ):
        result = hybrid_classifier.classify_document(
            "Invoice for $100.00 dated 2024-01-01",
            industry="financial"
        )

        assert isinstance(result, HybridClassificationResult)
        assert result.doc_type == "invoice"
        assert 0.8 <= result.confidence <= 0.9  # Should be weighted between LLM and pattern confidence
        assert len(result.features) >= 2  # Should have amount and date features
        assert len(result.pattern_matches) == 1
        assert result.llm_result == mock_llm_result
        assert result.metadata["industry"] == "financial"

def test_classify_document_no_patterns(hybrid_classifier, mock_llm_result):
    """Test classification with no pattern matches."""
    with patch.object(
        hybrid_classifier.content_classifier,
        '_classify_with_llm',
        return_value=mock_llm_result
    ), patch.object(
        hybrid_classifier.pattern_matcher,
        'find_matches',
        return_value=[]
    ):
        result = hybrid_classifier.classify_document("Simple text")

        assert result.confidence == mock_llm_result.confidence
        assert not result.pattern_matches
        assert result.metadata["pattern_confidence"] is None

def test_classify_document_error_handling(hybrid_classifier):
    """Test error handling in classification."""
    with patch.object(
        hybrid_classifier.content_classifier,
        '_classify_with_llm',
        side_effect=Exception("LLM Error")
    ):
        with pytest.raises(Exception) as exc_info:
            hybrid_classifier.classify_document("Test text")
        assert "LLM Error" in str(exc_info.value)

def test_merge_results_weighting(hybrid_classifier, mock_llm_result, mock_pattern_matches):
    """Test confidence weighting in result merging."""
    result = hybrid_classifier._merge_results(
        mock_llm_result,
        mock_pattern_matches,
        "financial"
    )

    # Expected confidence: (0.8 * 0.6) + (0.9 * 0.4) = 0.84
    assert 0.83 <= result.confidence <= 0.85
    assert result.metadata["llm_confidence"] == 0.8
    assert result.metadata["pattern_confidence"] == 0.9

def test_classify_document_with_industry(hybrid_classifier, mock_llm_result, mock_pattern_matches):
    """Test classification with industry context."""
    with patch.object(
        hybrid_classifier.content_classifier,
        '_classify_with_llm',
        return_value=mock_llm_result
    ), patch.object(
        hybrid_classifier.pattern_store,
        'get_patterns_by_industry',
        return_value=[mock_pattern_matches[0].pattern]
    ), patch.object(
        hybrid_classifier.pattern_matcher,
        'find_matches',
        return_value=mock_pattern_matches
    ):
        result = hybrid_classifier.classify_document(
            "Invoice text",
            industry="financial"
        )

        assert result.metadata["industry"] == "financial"
        assert len(result.pattern_matches) == 1
        assert result.pattern_matches[0].pattern.industry == "financial"