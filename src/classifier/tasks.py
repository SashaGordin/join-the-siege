"""Celery tasks for asynchronous processing."""

import logging
import hashlib
from typing import Dict, List, Optional, Any
from celery import Task
from .config.celery_config import celery_app
from .pattern_learning.pattern_matcher import PatternMatcher
from .pattern_learning.pattern_store import PatternStore
from .pattern_learning.pattern_learner import PatternLearner
from .pattern_learning.pattern_validator import PatternValidator
from .hybrid_classifier import HybridClassifier
from .content_classifier import ContentClassifier
from .services.cache_service import CacheService
from .feature_extraction import extract_features_from_text

logger = logging.getLogger(__name__)

class BaseTask(Task):
    """Base task class with error handling and retry logic."""

    _cache_service = None
    _hybrid_classifier = None

    @property
    def cache_service(self) -> CacheService:
        """Get or create cache service instance."""
        if self._cache_service is None:
            self._cache_service = CacheService()
        return self._cache_service

    @property
    def hybrid_classifier(self) -> HybridClassifier:
        """Get or create hybrid classifier instance."""
        if self._hybrid_classifier is None:
            content_classifier = ContentClassifier()
            pattern_matcher = PatternMatcher()
            pattern_store = PatternStore()
            pattern_validator = PatternValidator()
            pattern_learner = PatternLearner(pattern_store, pattern_validator)

            self._hybrid_classifier = HybridClassifier(
                content_classifier=content_classifier,
                pattern_matcher=pattern_matcher,
                pattern_store=pattern_store,
                pattern_learner=pattern_learner
            )

            # Set up cache service
            pattern_matcher.cache_service = self.cache_service
            pattern_store.cache_service = self.cache_service

        return self._hybrid_classifier

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            f"Task {task_id} failed: {exc}\nArgs: {args}\nKwargs: {kwargs}\n"
            f"Exception Info: {einfo}"
        )
        # Implement failure notification if needed

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            f"Task {task_id} being retried: {exc}\nArgs: {args}\nKwargs: {kwargs}\n"
            f"Exception Info: {einfo}"
        )

@celery_app.task(base=BaseTask, bind=True, max_retries=3)
def process_document(
    self,
    document_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a document asynchronously.

    Args:
        document_id: Unique identifier for the document
        content: Document content to process
        metadata: Optional document metadata

    Returns:
        Dict containing processing results
    """
    logger.info(f"Processing document {document_id}")
    try:
        # Generate cache key using content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        cache_key = f"classification:{content_hash}"
        logger.info(f"Using cache key: {cache_key}")

        # Check cache first
        cached_result = self.cache_service.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for document {document_id}")
            return cached_result

        # Use hybrid classifier
        result = self.hybrid_classifier.classify_document(
            content,
            industry=metadata.get("industry") if metadata else None
        )

        # Convert result to serializable format
        serialized_result = {
            'document_id': document_id,
            'doc_type': result.doc_type,
            'confidence': result.confidence,
            'features': result.features,
            'pattern_matches': [
                {
                    'pattern_id': match.pattern.id,
                    'text': match.text,
                    'start': match.start,
                    'end': match.end,
                    'match_type': match.match_type.value,
                    'confidence': match.confidence.value,
                    'context': match.context
                }
                for match in result.pattern_matches
            ] if result.pattern_matches else [],
            'metadata': {
                **(metadata if metadata else {}),
                **result.metadata
            },
            'status': 'completed',
            'error': None
        }

        # Cache result
        logger.info(f"Caching result with key: {cache_key}")
        self.cache_service.set(cache_key, serialized_result, expire=3600)
        logger.info(f"Cache set complete for key: {cache_key}")

        return serialized_result

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        self.retry(exc=e, countdown=300)  # Retry after 5 minutes

@celery_app.task(base=BaseTask, bind=True)
def match_patterns(
    self,
    content: str,
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Match patterns in content asynchronously.

    Args:
        content: Text content to match patterns against
        context: Optional context information

    Returns:
        List of pattern matches
    """
    logger.info("Starting pattern matching")
    try:
        # Generate cache key from content and context
        cache_key = self.cache_service._generate_cache_key(
            "pattern_matching",
            content,
            context
        )

        # Check cache
        cached_result = self.cache_service.get(cache_key)
        if cached_result:
            logger.info("Cache hit for pattern matching")
            return cached_result

        # Get patterns (implement pattern loading logic)
        patterns: List[Pattern] = []  # Load patterns here

        # Perform matching
        matches = self.pattern_matcher.find_matches(content, patterns, context)

        # Convert matches to serializable format
        result = [
            {
                'pattern_id': match.pattern.id,
                'text': match.text,
                'start': match.start,
                'end': match.end,
                'match_type': match.match_type.value,
                'confidence': match.confidence.value,
                'context': match.context
            }
            for match in matches
        ]

        # Cache result
        self.cache_service.set(cache_key, result, expire=3600)

        return result

    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}")
        self.retry(exc=e, countdown=300)

@celery_app.task(base=BaseTask, bind=True)
def extract_features(
    self,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract features from content asynchronously.

    Args:
        content: Text content to extract features from
        metadata: Optional metadata to guide feature extraction

    Returns:
        Dict of extracted features
    """
    logger.info("Starting feature extraction")
    try:
        # Generate cache key
        cache_key = self.cache_service._generate_cache_key(
            "feature_extraction",
            content,
            metadata
        )

        # Check cache
        cached_result = self.cache_service.get(cache_key)
        if cached_result:
            logger.info("Cache hit for feature extraction")
            return cached_result

        # Extract features using the unified function
        features = extract_features_from_text(content, metadata)
        result = {"features": features}

        # Cache result
        self.cache_service.set(cache_key, result, expire=3600)

        return result

    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        self.retry(exc=e, countdown=300)