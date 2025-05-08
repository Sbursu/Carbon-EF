import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.utils import load_config
from src.vector_store import EmbeddingGenerator

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based cache for storing query results
    Falls back to in-memory cache if Redis is not available
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Redis cache

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.cache_config = self.config["cache"]["redis"]

        # Connect to Redis
        self.redis_host = self.cache_config["host"]
        self.redis_port = self.cache_config["port"]
        self.default_ttl = self.cache_config["ttl"]
        self.max_size = self.cache_config["max_size"]

        # In-memory cache as fallback
        self.in_memory_mode = False
        self.memory_cache = {"exact": {}, "semantic": {}, "vector": {}, "metadata": {}}

        # Try to connect to Redis if available
        if REDIS_AVAILABLE:
            logger.info(f"Connecting to Redis at {self.redis_host}:{self.redis_port}")
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_host, port=self.redis_port, decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis: {str(e)}. Using in-memory cache instead."
                )
                self.in_memory_mode = True
        else:
            logger.warning(
                "Redis package not available. Using in-memory cache instead."
            )
            self.in_memory_mode = True

        # Initialize embedding generator for semantic similarity
        self.embedding_generator = EmbeddingGenerator(config_path=config_path)

        # Cache namespaces
        self.exact_cache_prefix = "exact:"
        self.semantic_cache_prefix = "semantic:"
        self.vector_prefix = "vector:"
        self.metadata_prefix = "metadata:"

    def get_exact(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from exact cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        cache_key = f"{self.exact_cache_prefix}{key}"

        if self.in_memory_mode:
            cached_value = self.memory_cache["exact"].get(key)
            if cached_value and cached_value.get("expiry", float("inf")) > time.time():
                logger.info(f"Exact cache hit for key: {key} (in-memory)")
                return cached_value.get("value")
            elif cached_value:
                # Remove expired entry
                del self.memory_cache["exact"][key]
            logger.info(f"Exact cache miss for key: {key} (in-memory)")
            return None

        cached_value = self.redis_client.get(cache_key)

        if cached_value:
            logger.info(f"Exact cache hit for key: {key}")
            try:
                return json.loads(cached_value)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode cached value for key: {key}")
                return None

        logger.info(f"Exact cache miss for key: {key}")
        return None

    def set_exact(
        self, key: str, value: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in exact cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)

        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl

        if self.in_memory_mode:
            try:
                self.memory_cache["exact"][key] = {
                    "value": value,
                    "expiry": time.time() + ttl,
                }
                logger.info(f"Set value in exact cache with key: {key} (in-memory)")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to set value in exact cache: {str(e)} (in-memory)"
                )
                return False

        cache_key = f"{self.exact_cache_prefix}{key}"

        try:
            serialized_value = json.dumps(value)
            self.redis_client.setex(cache_key, ttl, serialized_value)
            logger.info(f"Set value in exact cache with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set value in exact cache: {str(e)}")
            return False

    def get_semantic(
        self, query: str, threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Get value from semantic cache based on query similarity

        Args:
            query: Query string
            threshold: Similarity threshold (0-1)

        Returns:
            Cached value or None if no similar query found
        """
        # For in-memory mode, implement a simplified version
        if self.in_memory_mode:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.embed_query(query)

            max_similarity = 0.0
            most_similar_value = None
            original_query = ""

            # Find the most similar cached query
            for key, item in self.memory_cache["semantic"].items():
                if item.get("expiry", 0) < time.time():
                    continue  # Skip expired items

                vector = self.memory_cache["vector"].get(key, {}).get("value")
                if vector:
                    similarity = self._cosine_similarity(query_embedding, vector)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_value = item.get("value")
                        original_query = (
                            self.memory_cache["metadata"]
                            .get(key, {})
                            .get("value", {})
                            .get("query", "")
                        )

            if max_similarity >= threshold and most_similar_value:
                logger.info(
                    f"Semantic cache hit with similarity: {max_similarity:.4f} (in-memory)"
                )
                result = most_similar_value
                if "cache_info" not in result:
                    result["cache_info"] = {}
                result["cache_info"]["similarity"] = max_similarity
                result["cache_info"]["original_query"] = original_query
                return result

            logger.info(
                f"Semantic cache miss (max similarity: {max_similarity:.4f}) (in-memory)"
            )
            return None

        # Implementation for Redis mode
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.embed_query(query)

            # Get all vector keys
            vector_keys = self.redis_client.keys(f"{self.vector_prefix}*")
            if not vector_keys:
                logger.info("Semantic cache is empty")
                return None

            # Calculate similarity for each cached query
            max_similarity = 0.0
            most_similar_key = None

            for vector_key in vector_keys:
                vector_value = self.redis_client.get(vector_key)
                if vector_value:
                    try:
                        cached_embedding = json.loads(vector_value)
                        similarity = self._cosine_similarity(
                            query_embedding, cached_embedding
                        )

                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_key = vector_key.replace(
                                self.vector_prefix, self.semantic_cache_prefix
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error processing vector: {str(e)}")
                        continue

            if max_similarity >= threshold and most_similar_key:
                # Get cached value
                cached_value = self.redis_client.get(most_similar_key)

                if cached_value:
                    logger.info(
                        f"Semantic cache hit with similarity: {max_similarity:.4f}"
                    )
                    # Get metadata
                    metadata_key = most_similar_key.replace(
                        self.semantic_cache_prefix, self.metadata_prefix
                    )
                    metadata_str = self.redis_client.get(metadata_key)
                    metadata = json.loads(metadata_str) if metadata_str else {}

                    # Update access count
                    metadata["access_count"] = metadata.get("access_count", 0) + 1
                    metadata["last_access"] = time.time()
                    self.redis_client.setex(
                        metadata_key, self.default_ttl, json.dumps(metadata)
                    )

                    # Return cached value
                    try:
                        result = json.loads(cached_value)
                        # Add similarity info
                        if "cache_info" not in result:
                            result["cache_info"] = {}
                        result["cache_info"]["similarity"] = max_similarity
                        result["cache_info"]["original_query"] = metadata.get(
                            "query", ""
                        )
                        return result
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to decode cached value for key: {most_similar_key}"
                        )
                        return None

            logger.info(f"Semantic cache miss (max similarity: {max_similarity:.4f})")
            return None
        except Exception as e:
            logger.error(f"Error in semantic cache lookup: {str(e)}")
            return None

    def set_semantic(
        self, query: str, value: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in semantic cache

        Args:
            query: Query string
            value: Value to cache
            ttl: Time-to-live in seconds (optional)

        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.embed_query(query)

            # Create cache keys
            query_hash = hashlib.md5(query.encode()).hexdigest()

            if self.in_memory_mode:
                # Store in memory
                expiry = time.time() + ttl
                self.memory_cache["semantic"][query_hash] = {
                    "value": value,
                    "expiry": expiry,
                }
                self.memory_cache["vector"][query_hash] = {
                    "value": query_embedding,
                    "expiry": expiry,
                }
                self.memory_cache["metadata"][query_hash] = {
                    "value": {
                        "query": query,
                        "created_at": time.time(),
                        "access_count": 0,
                        "last_access": time.time(),
                    },
                    "expiry": expiry,
                }
                logger.info(
                    f"Set value in semantic cache for query: {query} (in-memory)"
                )

                # Check cache size and evict if necessary
                self._check_and_evict()

                return True

            # Redis implementation
            semantic_key = f"{self.semantic_cache_prefix}{query_hash}"
            vector_key = f"{self.vector_prefix}{query_hash}"
            metadata_key = f"{self.metadata_prefix}{query_hash}"

            # Store the value
            serialized_value = json.dumps(value)
            self.redis_client.setex(semantic_key, ttl, serialized_value)

            # Store the vector
            self.redis_client.setex(vector_key, ttl, json.dumps(query_embedding))

            # Store metadata
            metadata = {
                "query": query,
                "created_at": time.time(),
                "access_count": 0,
                "last_access": time.time(),
            }
            self.redis_client.setex(metadata_key, ttl, json.dumps(metadata))

            logger.info(f"Set value in semantic cache for query: {query}")

            # Check cache size and evict if necessary
            self._check_and_evict()

            return True
        except Exception as e:
            logger.error(f"Failed to set value in semantic cache: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear the cache

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.in_memory_mode:
                # Clear in-memory cache
                self.memory_cache = {
                    "exact": {},
                    "semantic": {},
                    "vector": {},
                    "metadata": {},
                }
                logger.info("In-memory cache cleared")
                return True

            # Delete all keys with the cache prefixes
            for prefix in [
                self.exact_cache_prefix,
                self.semantic_cache_prefix,
                self.vector_prefix,
                self.metadata_prefix,
            ]:
                keys = self.redis_client.keys(f"{prefix}*")
                if keys:
                    self.redis_client.delete(*keys)

            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    def _check_and_evict(self) -> None:
        """
        Check cache size and evict least recently used items if necessary
        """
        try:
            if self.in_memory_mode:
                # Get current size
                current_size = len(self.memory_cache["semantic"])

                if current_size <= self.max_size:
                    return

                # Need to evict items
                items_to_evict = current_size - self.max_size
                logger.info(
                    f"Cache size ({current_size}) exceeds limit ({self.max_size}), evicting {items_to_evict} items (in-memory)"
                )

                # Get metadata for all items
                metadata_items = []
                for query_hash in self.memory_cache["semantic"]:
                    metadata = (
                        self.memory_cache["metadata"]
                        .get(query_hash, {})
                        .get("value", {})
                    )
                    last_access = metadata.get("last_access", 0)
                    access_count = metadata.get("access_count", 0)
                    metadata_items.append((query_hash, last_access, access_count))

                # Sort by last access time (older first) and access count (less accessed first)
                metadata_items.sort(key=lambda x: (x[1], x[2]))

                # Evict oldest items
                for i in range(items_to_evict):
                    if i < len(metadata_items):
                        query_hash = metadata_items[i][0]
                        self._evict_item(query_hash)
                return

            # Redis implementation
            # Get current size
            semantic_keys = self.redis_client.keys(f"{self.semantic_cache_prefix}*")
            current_size = len(semantic_keys)

            if current_size <= self.max_size:
                return

            # Need to evict items
            items_to_evict = current_size - self.max_size
            logger.info(
                f"Cache size ({current_size}) exceeds limit ({self.max_size}), evicting {items_to_evict} items"
            )

            # Get metadata for all items
            metadata_items = []
            for semantic_key in semantic_keys:
                query_hash = semantic_key.replace(self.semantic_cache_prefix, "")
                metadata_key = f"{self.metadata_prefix}{query_hash}"
                metadata_str = self.redis_client.get(metadata_key)

                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                        # Use last_access and access_count for LRU-based eviction
                        metadata_items.append(
                            (
                                query_hash,
                                metadata.get("last_access", 0),
                                metadata.get("access_count", 0),
                            )
                        )
                    except json.JSONDecodeError:
                        # If metadata is corrupt, evict this item
                        metadata_items.append((query_hash, 0, 0))
                else:
                    # If no metadata, evict this item
                    metadata_items.append((query_hash, 0, 0))

            # Sort by last access time (older first) and access count (less accessed first)
            metadata_items.sort(key=lambda x: (x[1], x[2]))

            # Evict oldest items
            for i in range(items_to_evict):
                if i < len(metadata_items):
                    query_hash = metadata_items[i][0]
                    self._evict_item(query_hash)
        except Exception as e:
            logger.error(f"Error during cache eviction: {str(e)}")

    def _evict_item(self, query_hash: str) -> None:
        """
        Evict a single item from the cache

        Args:
            query_hash: Hash of the query to evict
        """
        if self.in_memory_mode:
            # Remove from all in-memory stores
            for cache_type in ["semantic", "vector", "metadata"]:
                if query_hash in self.memory_cache[cache_type]:
                    del self.memory_cache[cache_type][query_hash]
            logger.info(f"Evicted cache item with hash: {query_hash} (in-memory)")
            return

        # Redis implementation
        # Delete all keys for this item
        keys_to_delete = [
            f"{self.semantic_cache_prefix}{query_hash}",
            f"{self.vector_prefix}{query_hash}",
            f"{self.metadata_prefix}{query_hash}",
        ]

        for key in keys_to_delete:
            self.redis_client.delete(key)

        logger.info(f"Evicted cache item with hash: {query_hash}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def close(self) -> None:
        """
        Close the Redis connection
        """
        if not self.in_memory_mode:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")

    def __del__(self) -> None:
        """
        Close the Redis connection when the object is deleted
        """
        self.close()
