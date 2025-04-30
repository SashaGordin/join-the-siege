"""Redis-based caching service for the classifier."""

import redis
import json
import hashlib
from typing import Optional, Any, Dict
import logging
from functools import wraps
import os
from dotenv import load_dotenv
from src.classifier.pattern_learning.models import Pattern, PatternMatch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheService:
    """Redis-based caching service with different TTLs for different types of data."""

    # Cache TTLs in seconds
    TTL_MAPPING = {
        'classification': 86400,    # 24 hours
        'patterns': 3600,          # 1 hour
        'industry_config': 3600,   # 1 hour
        'api_response': 43200      # 12 hours
    }

    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )
        self._test_connection()

    def _test_connection(self):
        """Test Redis connection."""
        try:
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def _generate_key(self, key_type: str, identifier: str) -> str:
        """Generate a cache key with prefix."""
        return f"{key_type}:{identifier}"

    def _generate_file_hash(self, file_content: str) -> str:
        """Generate a hash for file content."""
        return hashlib.sha256(file_content.encode('utf-8')).hexdigest()

    def get_classification(self, file_content: bytes) -> Optional[Dict[str, Any]]:
        """
        Get cached classification result for a file.

        Args:
            file_content: Raw file content

        Returns:
            Cached classification result or None
        """
        file_hash = self._generate_file_hash(file_content)
        key = self._generate_key('classification', file_hash)

        try:
            cached = self.redis_client.get(key)
            if cached:
                logger.info(f"Cache hit for classification: {file_hash}")
                return json.loads(cached)
            logger.info(f"Cache miss for classification: {file_hash}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set_classification(self, file_content: bytes, result: Dict[str, Any]) -> bool:
        """
        Cache classification result for a file.

        Args:
            file_content: Raw file content
            result: Classification result to cache

        Returns:
            bool: True if successful
        """
        file_hash = self._generate_file_hash(file_content)
        key = self._generate_key('classification', file_hash)

        try:
            self.redis_client.setex(
                key,
                self.TTL_MAPPING['classification'],
                json.dumps(result)
            )
            logger.info(f"Cached classification result: {file_hash}")
            return True
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
            return False

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get cached pattern."""
        key = self._generate_key('pattern', pattern_id)
        try:
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Error retrieving pattern: {str(e)}")
            return None

    def set_pattern(self, pattern_id: str, pattern: Dict[str, Any]) -> bool:
        """Cache pattern data."""
        key = self._generate_key('pattern', pattern_id)
        try:
            self.redis_client.setex(
                key,
                self.TTL_MAPPING['patterns'],
                json.dumps(pattern)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching pattern: {str(e)}")
            return False

    def get_industry_config(self, industry: str) -> Optional[Dict[str, Any]]:
        """Get cached industry configuration."""
        key = self._generate_key('industry', industry)
        try:
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Error retrieving industry config: {str(e)}")
            return None

    def set_industry_config(self, industry: str, config: Dict[str, Any]) -> bool:
        """Cache industry configuration."""
        key = self._generate_key('industry', industry)
        try:
            self.redis_client.setex(
                key,
                self.TTL_MAPPING['industry_config'],
                json.dumps(config)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching industry config: {str(e)}")
            return False

    def get_api_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached API response."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = self._generate_key('api', prompt_hash)
        try:
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Error retrieving API response: {str(e)}")
            return None

    def set_api_response(self, prompt: str, response: Dict[str, Any]) -> bool:
        """Cache API response."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = self._generate_key('api', prompt_hash)
        try:
            self.redis_client.setex(
                key,
                self.TTL_MAPPING['api_response'],
                json.dumps(response)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching API response: {str(e)}")
            return False

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            cache_type: Optional type of cache to clear (classification, patterns, etc.)
                      If None, clears all cache
        """
        try:
            if cache_type:
                pattern = f"{cache_type}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {cache_type} cache")
            else:
                self.redis_client.flushdb()
                logger.info("Cleared all cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            # Log error but don't fail
            print(f"Cache get error: {str(e)}")
            return None

    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds (default 1 hour)

        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = json.dumps(value)
            return bool(self.redis_client.setex(key, expire, serialized))
        except Exception as e:
            # Log error but don't fail
            print(f"Cache set error: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Cache delete error: {str(e)}")
            return False

    def clear(self, pattern: Optional[str] = None) -> bool:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Optional pattern to match keys (e.g. "prefix:*")

        Returns:
            True if successful, False otherwise
        """
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return bool(self.redis_client.delete(*keys))
                return True
            return bool(self.redis_client.flushdb())
        except Exception as e:
            print(f"Cache clear error: {str(e)}")
            return False

    def _generate_cache_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """
        Generate a cache key from arguments.

        Args:
            prefix: Key prefix
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            Cache key string
        """
        # Convert args and kwargs to strings and sort for consistency
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

        # Join parts and create hash
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()[:16]

        return f"{prefix}:{key_hash}"

def _serialize_arg(arg):
    """Serialize argument for cache key generation."""
    if isinstance(arg, list):
        return json.dumps([_serialize_arg(item) for item in arg], sort_keys=True)
    elif isinstance(arg, dict):
        return json.dumps({k: _serialize_arg(v) for k, v in sorted(arg.items())})
    elif isinstance(arg, Pattern):
        return json.dumps(arg.to_dict(), sort_keys=True)
    else:
        return str(arg)

def cache_decorator(cache_type: str):
    """
    Decorator for caching function results.

    Args:
        cache_type: Type of cache (classification, patterns, etc.)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key from function arguments
            key_parts = [func.__name__]
            # Convert args to strings and handle special types
            for arg in args:
                key_parts.append(_serialize_arg(arg))
            # Add sorted kwargs
            key_parts.extend(f"{k}:{_serialize_arg(v)}" for k, v in sorted(kwargs.items()))
            # Join and hash
            cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_result = self.cache_service.get_classification(cache_key)
            if cached_result:
                # Deserialize matches
                if isinstance(cached_result, list):
                    return [PatternMatch.from_dict(m) for m in cached_result]
                return cached_result

            # If not in cache, execute function and cache result
            result = func(self, *args, **kwargs)

            # Serialize matches for caching
            if isinstance(result, list):
                serialized_result = [m.to_dict() for m in result]
                self.cache_service.set_classification(cache_key, serialized_result)
            else:
                self.cache_service.set_classification(cache_key, result)

            return result
        return wrapper
    return decorator