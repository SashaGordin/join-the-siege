"""
Unified feature extraction module.
"""
from typing import Dict, Any, Optional, List
from .content_classifier import ContentClassifier
from .pattern_learning.pattern_matcher import PatternMatcher
from .pattern_learning.pattern_store import PatternStore
from .config.config_manager import IndustryConfigManager
import logging

logger = logging.getLogger(__name__)

# Initialize shared resources (could be improved for dependency injection/testing)
content_classifier = ContentClassifier()
pattern_store = PatternStore()
pattern_matcher = PatternMatcher()
config_manager = IndustryConfigManager()

def extract_features_from_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Extract features from text using both LLM-based and pattern-based methods.

    Args:
        text: The document text to extract features from.
        metadata: Optional metadata (e.g., industry).

    Returns:
        List of extracted features (dicts with type, value, etc.).
    """
    industry = metadata.get("industry") if metadata else None
    logger.info(f"Extracting features for industry: {industry}")

    # 1. LLM-based extraction (via ContentClassifier)
    llm_features = []
    try:
        llm_result = content_classifier._classify_with_llm(text, industry)
        llm_features = llm_result.features if hasattr(llm_result, 'features') else []
        logger.debug(f"LLM-extracted features: {llm_features}")
    except Exception as e:
        logger.error(f"LLM feature extraction failed: {e}")

    # 2. Pattern-based extraction
    pattern_features = []
    try:
        patterns = pattern_store.get_patterns_by_industry(industry) if industry else pattern_store.get_all_patterns()
        matches = pattern_matcher.find_matches(text, patterns)
        for match in matches:
            pattern_features.append({
                "type": match.pattern.feature_type,
                "value": match.text,
                "start": match.start,
                "end": match.end,
                "confidence": match.confidence.value,
                "context": match.context
            })
        logger.debug(f"Pattern-extracted features: {pattern_features}")
    except Exception as e:
        logger.error(f"Pattern feature extraction failed: {e}")

    # 3. Merge and deduplicate features (by type and value)
    merged = {(f["type"], f["value"]): f for f in llm_features + pattern_features}
    merged_features = list(merged.values())
    logger.info(f"Total features extracted: {len(merged_features)}")
    return merged_features