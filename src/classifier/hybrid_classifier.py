"""Hybrid document classifier combining LLM and pattern matching."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from .content_classifier import ContentClassifier, ClassificationResult
from .pattern_learning.pattern_matcher import PatternMatcher
from .pattern_learning.pattern_store import PatternStore
from .pattern_learning.models import PatternMatch, ConfidenceScore

logger = logging.getLogger(__name__)

@dataclass
class HybridClassificationResult:
    """Result from hybrid classification."""
    doc_type: str
    confidence: float
    features: List[Dict[str, Any]]
    pattern_matches: List[PatternMatch]
    llm_result: Optional[ClassificationResult]
    metadata: Dict[str, Any]

class HybridClassifier:
    """Combines LLM and pattern-based classification."""

    def __init__(self):
        """Initialize the hybrid classifier."""
        self.pattern_store = PatternStore()
        self.pattern_matcher = PatternMatcher()
        self.content_classifier = ContentClassifier(pattern_store=self.pattern_store)

    def classify_document(
        self,
        text: str,
        industry: Optional[str] = None
    ) -> HybridClassificationResult:
        """
        Classify a document using both LLM and pattern matching.

        Args:
            text: Document text to classify
            industry: Optional industry context

        Returns:
            HybridClassificationResult containing combined classification
        """
        logger.info("Starting hybrid classification")

        try:
            # Get LLM classification
            llm_result = self.content_classifier._classify_with_llm(text, industry)
            logger.debug(f"LLM classification: {llm_result.doc_type} ({llm_result.confidence})")

            # Get pattern matches
            patterns = self.pattern_store.get_patterns_by_industry(industry) if industry else []
            pattern_matches = self.pattern_matcher.find_matches(text, patterns)
            logger.debug(f"Found {len(pattern_matches)} pattern matches")

            # Cross-validate and merge results
            final_result = self._merge_results(llm_result, pattern_matches, industry)
            logger.info(f"Final classification: {final_result.doc_type} ({final_result.confidence})")

            return final_result

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise

    def _merge_results(
        self,
        llm_result: ClassificationResult,
        pattern_matches: List[PatternMatch],
        industry: Optional[str] = None
    ) -> HybridClassificationResult:
        """Merge LLM and pattern matching results."""
        # Start with LLM confidence
        confidence = llm_result.confidence

        # Calculate pattern-based confidence
        if pattern_matches:
            pattern_confidence = sum(m.confidence.value for m in pattern_matches) / len(pattern_matches)
            # Weight LLM and pattern confidence (favoring LLM slightly)
            confidence = (confidence * 0.6) + (pattern_confidence * 0.4)

        # Merge features
        features = llm_result.features.copy()

        # Add pattern-based features
        pattern_features = []
        for match in pattern_matches:
            pattern_features.append({
                "type": match.pattern.feature_type,
                "text": match.text,
                "confidence": match.confidence.value,
                "context": match.context
            })

        # Create metadata
        metadata = {
            "industry": industry,
            "llm_confidence": llm_result.confidence,
            "pattern_confidence": pattern_confidence if pattern_matches else None,
            "pattern_count": len(pattern_matches)
        }

        return HybridClassificationResult(
            doc_type=llm_result.doc_type,
            confidence=confidence,
            features=features,
            pattern_matches=pattern_matches,
            llm_result=llm_result,
            metadata=metadata
        )