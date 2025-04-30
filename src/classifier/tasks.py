"""Celery tasks for asynchronous processing."""

import logging
from typing import Dict, List, Optional, Any
from celery import Task
from .config.celery_config import celery_app
from .pattern_learning.pattern_matcher import PatternMatcher
from .pattern_learning.models import Pattern, PatternMatch
from .services.cache_service import CacheService

logger = logging.getLogger(__name__)

class BaseTask(Task):
    """Base task class with error handling and retry logic."""

    _cache_service = None
    _pattern_matcher = None

    @property
    def cache_service(self) -> CacheService:
        """Get or create cache service instance."""
        if self._cache_service is None:
            self._cache_service = CacheService()
        return self._cache_service

    @property
    def pattern_matcher(self) -> PatternMatcher:
        """Get or create pattern matcher instance."""
        if self._pattern_matcher is None:
            self._pattern_matcher = PatternMatcher()
            self._pattern_matcher.cache_service = self.cache_service
        return self._pattern_matcher

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
        # Check cache first
        cache_key = f"document_processing:{document_id}"
        cached_result = self.cache_service.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for document {document_id}")
            return cached_result

        # Extract features
        feature_task = extract_features.delay(content, metadata)
        features = feature_task.get(timeout=300)  # 5 minute timeout

        # Match patterns
        pattern_task = match_patterns.delay(content, features)
        matches = pattern_task.get(timeout=300)

        # Combine results
        result = {
            'document_id': document_id,
            'features': features,
            'matches': matches,
            'metadata': metadata
        }

        # Cache result
        self.cache_service.set(cache_key, result, expire=3600)

        return result

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

        # Extract features (implement feature extraction logic)
        features = {
            # Feature extraction implementation here
        }

        # Cache result
        self.cache_service.set(cache_key, features, expire=3600)

        return features

    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        self.retry(exc=e, countdown=300)