import logging
from typing import Any, Dict, List, Optional, Union

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PayloadSchemaType,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from src.utils import load_config

logger = logging.getLogger(__name__)

# Track whether Qdrant is actually available in the current environment
QDRANT_REACHABLE = None


class QdrantStore:
    """
    Qdrant vector store for emission factor embeddings
    Falls back to a mock implementation when Qdrant is not available
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Qdrant store

        Args:
            config_path: Path to configuration file
        """
        global QDRANT_REACHABLE

        self.config = load_config(config_path)
        self.qdrant_config = self.config["vector_store"]["qdrant"]

        self.collection_name = self.qdrant_config["collection_name"]
        self.vector_size = self.qdrant_config["vector_size"]
        self.distance = (
            (
                Distance.COSINE
                if self.qdrant_config["distance"] == "Cosine"
                else Distance.EUCLID
            )
            if QDRANT_AVAILABLE
            else "cosine"
        )

        # Fallback in-memory storage
        self.in_memory_mode = False
        self.memory_vectors = []
        self.memory_metadata = []
        self.memory_ids = []

        # Try to connect to Qdrant if available and we haven't determined reachability yet
        if QDRANT_AVAILABLE and QDRANT_REACHABLE is None:
            try:
                self.client = QdrantClient(
                    host=self.qdrant_config.get(
                        "host", "localhost"
                    ),  # Use .get() for safety
                    port=self.qdrant_config.get("port", 6333),
                    timeout=5.0,  # Set a timeout for connection
                )
                # Test connection
                self.client.get_collections()  # Check if connection is successful
                logger.info(
                    f"Successfully connected to Qdrant at {self.qdrant_config.get('host')}:{self.qdrant_config.get('port')}"
                )
                QDRANT_REACHABLE = True

                # Ensure collection exists
                self._ensure_collection()  # Create collection if needed
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Qdrant: {str(e)}. Using in-memory fallback."
                )
                QDRANT_REACHABLE = False
                self.in_memory_mode = True
        elif QDRANT_AVAILABLE and QDRANT_REACHABLE is True:
            # Qdrant is already known to be reachable, initialize client
            try:
                self.client = QdrantClient(
                    host=self.qdrant_config.get("host", "localhost"),
                    port=self.qdrant_config.get("port", 6333),
                    timeout=5.0,  # Set a timeout for connection
                )
                # Ensure collection exists
                self._ensure_collection()
            except Exception as e:
                # If connection fails now, update the reachable status
                logger.warning(
                    f"Failed to connect to Qdrant: {str(e)}. Using in-memory fallback."
                )
                QDRANT_REACHABLE = False
                self.in_memory_mode = True
        else:
            # Library not available or known to be unreachable
            if not QDRANT_AVAILABLE:
                logger.warning(
                    "Qdrant library not available. Using in-memory fallback."
                )
            else:
                logger.warning(
                    "Qdrant is known to be unreachable. Using in-memory fallback."
                )
            self.in_memory_mode = True

        if self.in_memory_mode:
            logger.info("Using in-memory vector store (no persistence)")

    def _ensure_collection(self):
        """
        Ensure the collection exists in Qdrant. Create it if it doesn't.
        """
        if self.in_memory_mode:
            logger.debug("In-memory mode, skipping Qdrant collection check.")
            return

        if not hasattr(self, "client") or self.client is None:
            logger.error("Qdrant client not initialized. Cannot ensure collection.")
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection '{self.collection_name}'...")

                # Define vector params based on config
                vector_params = VectorParams(
                    size=self.vector_size,
                    distance=Distance[
                        self.qdrant_config.get("distance", "Cosine").upper()
                    ],  # Get from config
                )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                    # Add other configurations like optimizers if needed from config
                )
                logger.info(
                    f"Collection '{self.collection_name}' created successfully."
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(
                f"Failed to ensure Qdrant collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            # Depending on the error, you might want to switch to in-memory mode here
            # For now, we just log the error.

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[Union[str, int]]] = None,
    ):
        """
        Upsert vectors and metadata to Qdrant

        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for the vectors
        """
        if ids is None:
            ids = list(range(len(vectors)))

        if len(vectors) != len(metadata) or len(vectors) != len(ids):
            raise ValueError("Vectors, metadata, and IDs must have the same length")

        # For in-memory mode, simply store the data
        if self.in_memory_mode:
            for i in range(len(vectors)):
                # Check if ID already exists
                if ids[i] in self.memory_ids:
                    idx = self.memory_ids.index(ids[i])
                    self.memory_vectors[idx] = vectors[i]
                    self.memory_metadata[idx] = metadata[i]
                else:
                    self.memory_ids.append(ids[i])
                    self.memory_vectors.append(vectors[i])
                    self.memory_metadata.append(metadata[i])
            logger.info(f"Stored {len(vectors)} vectors in memory")
            return

        # Prepare points for batch upload
        points = []
        for i in range(len(vectors)):
            points.append(
                rest.PointStruct(id=ids[i], vector=vectors[i], payload=metadata[i])
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.info(
                f"Upserted batch {i//batch_size + 1}/{len(points)//batch_size + 1}"
            )

        logger.info(
            f"Upserted {len(vectors)} vectors to collection {self.collection_name}"
        )

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional filter conditions

        Returns:
            List of search results with metadata
        """
        # For in-memory mode, implement basic vector search
        if self.in_memory_mode:
            import numpy as np

            # Convert query vector to numpy array
            query_np = np.array(query_vector)

            # If we have no vectors, return empty list
            if not self.memory_vectors:
                logger.warning("No vectors in memory for search")
                return []

            # Calculate cosine similarity for all vectors
            similarities = []
            for i, vec in enumerate(self.memory_vectors):
                vec_np = np.array(vec)
                similarity = np.dot(query_np, vec_np) / (
                    np.linalg.norm(query_np) * np.linalg.norm(vec_np)
                )
                similarities.append((i, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Apply filters if needed
            if filter_conditions:
                filtered_results = []
                for i, similarity in similarities:
                    metadata = self.memory_metadata[i]
                    match = True
                    for field, value in filter_conditions.items():
                        if field not in metadata:
                            match = False
                            break
                        if isinstance(value, list):
                            if metadata[field] not in value:
                                match = False
                                break
                        elif metadata[field] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append((i, similarity))
                similarities = filtered_results

            # Format results (limit to requested number)
            results = []
            for i, similarity in similarities[:limit]:
                results.append(
                    {
                        "id": self.memory_ids[i],
                        "score": float(similarity),
                        **self.memory_metadata[i],
                    }
                )

            return results

        # Qdrant implementation
        search_filter = None
        if filter_conditions:
            must_conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, list):
                    # Handle list of values (any match)
                    should_conditions = []
                    for v in value:
                        should_conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=v))
                        )
                    must_conditions.append(rest.Filter(should=should_conditions))
                else:
                    # Single value match
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )

            search_filter = rest.Filter(must=must_conditions)

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=search_filter,
        )

        # Format results
        results = []
        for res in search_results:
            result = {"id": res.id, "score": res.score, **res.payload}
            results.append(result)

        return results

    def delete_vectors(self, ids: List[Union[str, int]]):
        """
        Delete vectors by IDs

        Args:
            ids: List of vector IDs to delete
        """
        if self.in_memory_mode:
            for id_to_delete in ids:
                if id_to_delete in self.memory_ids:
                    idx = self.memory_ids.index(id_to_delete)
                    del self.memory_ids[idx]
                    del self.memory_vectors[idx]
                    del self.memory_metadata[idx]
            logger.info(f"Deleted {len(ids)} vectors from memory")
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(points=ids),
        )
        logger.info(
            f"Deleted {len(ids)} vectors from collection {self.collection_name}"
        )

    def count_vectors(self) -> int:
        """
        Count vectors in the collection

        Returns:
            Number of vectors in the collection
        """
        if self.in_memory_mode:
            return len(self.memory_ids)

        return self.client.count(collection_name=self.collection_name).count

    def close(self):
        """
        Close connection to Qdrant
        """
        # No need to close anything in memory mode
        if not self.in_memory_mode and hasattr(self, "client"):
            # No explicit close method in QdrantClient, but good practice to include this method
            logger.info("Vector store resources released")

    def __del__(self):
        """
        Clean up resources when the object is destroyed
        """
        self.close()
