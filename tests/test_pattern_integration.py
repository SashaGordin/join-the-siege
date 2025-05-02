import pytest
from pathlib import Path
from src.classifier.content_classifier import ContentClassifier, ClassificationResult
from src.classifier.pattern_learning.pattern_matcher import PatternMatcher
from src.classifier.pattern_learning.pattern_store import PatternStore
from src.classifier.pattern_learning.models import Pattern, PatternType, ConfidenceScore
from src.classifier.hybrid_classifier import HybridClassifier
import tempfile
import os

@pytest.fixture
def dummy_patterns():
    return [
        Pattern(
            id="test_date",
            type=PatternType.REGEX,
            expression=r"\d{4}-\d{2}-\d{2}",
            feature_type="date",
            industry="financial",
            confidence=ConfidenceScore(1.0),
            examples=["2024-01-01"]
        ),
        Pattern(
            id="test_amount",
            type=PatternType.REGEX,
            expression=r"\$\d+\.\d{2}",
            feature_type="amount",
            industry="financial",
            confidence=ConfidenceScore(1.0),
            examples=["$100.00"]
        )
    ]

@pytest.fixture
def classifier_with_patterns(monkeypatch, dummy_patterns):
    # Patch PatternStore to return dummy patterns
    pattern_store = PatternStore()
    monkeypatch.setattr(pattern_store, 'get_patterns_by_industry', lambda industry: dummy_patterns)
    # Patch ContentClassifier to use our pattern store
    classifier = ContentClassifier()
    classifier.pattern_store = pattern_store
    classifier.pattern_matcher = PatternMatcher()
    return classifier

def test_pattern_feature_extraction(classifier_with_patterns):
    text = "The invoice date is 2024-01-01 and the amount is $100.00."
    industry = "financial"
    # Simulate extraction (integration will later use this in classify_file)
    patterns = classifier_with_patterns.pattern_store.get_patterns_by_industry(industry)
    matches = classifier_with_patterns.pattern_matcher.find_matches(text, patterns)
    assert any(m.pattern.feature_type == "date" for m in matches)
    assert any(m.pattern.feature_type == "amount" for m in matches)
    for m in matches:
        assert m.confidence.value >= 0.6

def test_pattern_confidence_scores(classifier_with_patterns):
    text = "2024-01-01 $100.00"
    industry = "financial"
    patterns = classifier_with_patterns.pattern_store.get_patterns_by_industry(industry)
    matches = classifier_with_patterns.pattern_matcher.find_matches(text, patterns)
    for m in matches:
        assert 0.0 <= m.confidence.value <= 1.0
        assert isinstance(m.confidence.factors, dict)

def test_classify_file_with_pattern_features(monkeypatch, dummy_patterns):
    # Create a temporary text file with known content
    text = "The invoice date is 2024-01-01 and the amount is $100.00."
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        # Patch pattern store to return dummy patterns
        classifier = HybridClassifier()
        monkeypatch.setattr(classifier.pattern_store, 'get_patterns_by_industry', lambda industry: dummy_patterns)
        # Patch LLM classification to return a dummy result of the correct type
        dummy_llm_result = ClassificationResult(
            doc_type="invoice",
            confidence=0.9,
            features=[{"type": "date", "values": ["2024-01-01"], "present": True}]
        )
        monkeypatch.setattr(classifier.content_classifier, '_classify_with_llm', lambda text, industry: dummy_llm_result)

        # Read the file as text
        with open(tmp_path, 'r') as f:
            file_text = f.read()
        result = classifier.classify_document(file_text, industry="financial")
        print("\n[TEST LOG] Classification result:", result)
        if hasattr(result, 'pattern_matches'):
            for m in result.pattern_matches:
                print(f"[TEST LOG] Pattern match: {m}")
        else:
            print("[TEST LOG] No pattern_matches in result")
        feature_types = {m.pattern.feature_type for m in result.pattern_matches}
        assert "date" in feature_types
        assert "amount" in feature_types
        for m in result.pattern_matches:
            assert m.confidence.value >= 0.6
    finally:
        os.unlink(tmp_path)