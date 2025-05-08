import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.cache.redis_cache import RedisCache
from src.utils import load_config

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache manager for caching query results

    Provides a high-level interface to the cache with policies for caching and retrieval
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the semantic cache

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)

        # Initialize Redis cache
        self.redis_cache = RedisCache(config_path)

        # Set up cache policies
        self.cache_complex_queries = True
        self.cache_threshold = 0.8  # Default similarity threshold
        self.enable_exact_matching = True
        self.enable_semantic_matching = True

        logger.info("Semantic cache initialized")

    def get(
        self, query: str, analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query

        Args:
            query: User query
            analysis: Optional query analysis

        Returns:
            Cached result or None if not found
        """
        # Generate cache key for exact matching
        key = self._generate_cache_key(query, analysis)

        # Try exact match first if enabled
        if self.enable_exact_matching:
            exact_result = self.redis_cache.get_exact(key)
            if exact_result:
                logger.info(f"Exact cache hit for query: {query}")
                exact_result["cache_info"] = exact_result.get("cache_info", {})
                exact_result["cache_info"]["cache_type"] = "exact"
                return exact_result

        # Try semantic match if enabled
        if self.enable_semantic_matching:
            semantic_result = self.redis_cache.get_semantic(
                query, threshold=self.cache_threshold
            )
            if semantic_result:
                logger.info(f"Semantic cache hit for query: {query}")
                semantic_result["cache_info"] = semantic_result.get("cache_info", {})
                semantic_result["cache_info"]["cache_type"] = "semantic"
                return semantic_result

        logger.info(f"Cache miss for query: {query}")
        return None

    def set(
        self,
        query: str,
        result: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set result in cache

        Args:
            query: User query
            result: Result to cache
            analysis: Optional query analysis
            ttl: Time-to-live in seconds (optional)

        Returns:
            True if successful, False otherwise
        """
        # Check if result should be cached
        if not self._should_cache(query, result, analysis):
            logger.info(f"Skipping cache for query: {query}")
            return False

        # Generate cache key for exact matching
        key = self._generate_cache_key(query, analysis)

        # Store in exact cache
        exact_success = self.redis_cache.set_exact(key, result, ttl)

        # Store in semantic cache
        semantic_success = self.redis_cache.set_semantic(query, result, ttl)

        return exact_success and semantic_success

    def _generate_cache_key(
        self, query: str, analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a cache key for a query

        Args:
            query: User query
            analysis: Optional query analysis

        Returns:
            Cache key
        """
        # Use query hash as base key
        key = hashlib.md5(query.encode()).hexdigest()

        # If analysis is provided, incorporate relevant fields into the key
        if analysis:
            # Extract relevant fields that would affect the result
            relevant_fields = {}
            for field in [
                "entity_type",
                "entity_name",
                "region",
                "comparison",
                "regions",
            ]:
                if field in analysis:
                    relevant_fields[field] = analysis[field]

            if relevant_fields:
                # Add analysis hash to the key
                analysis_str = json.dumps(relevant_fields, sort_keys=True)
                analysis_hash = hashlib.md5(analysis_str.encode()).hexdigest()
                key = f"{key}_{analysis_hash}"

        return key

    def _should_cache(
        self,
        query: str,
        result: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if a result should be cached

        Args:
            query: User query
            result: Result to potentially cache
            analysis: Optional query analysis

        Returns:
            True if the result should be cached, False otherwise
        """
        # Skip caching if result contains errors
        if "error" in result:
            return False

        # Skip caching for very short queries (likely too generic)
        if len(query.split()) < 3:
            return False

        # Skip caching for complex queries if disabled
        if (
            not self.cache_complex_queries
            and analysis
            and analysis.get("comparison", False)
        ):
            return False

        # Check if there are actual results to cache
        if "results" in result and not result["results"]:
            return False

        # Check answer quality
        if "answer" in result and (not result["answer"] or len(result["answer"]) < 10):
            return False

        return True

    def clear(self) -> bool:
        """
        Clear the cache

        Returns:
            True if successful, False otherwise
        """
        return self.redis_cache.clear()

    def close(self) -> None:
        """
        Close the cache connection
        """
        self.redis_cache.close()

    def __del__(self) -> None:
        """
        Close the cache connection when the object is deleted
        """
        self.close()
