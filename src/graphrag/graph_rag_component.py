import gc
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from sentence_transformers import SentenceTransformer

from src.configs.config_loader import ConfigLoader
from src.neo4j.neo4j_connector import Neo4jConnector
from src.vector_store.qdrant_store import QdrantVectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphRAGComponent:
    """Graph RAG component for combining vector search and graph traversal."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to reuse the same instance across imports."""
        if cls._instance is None:
            logger.info("Creating new GraphRAG instance")
            cls._instance = super(GraphRAGComponent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the GraphRAG component with Neo4j and Qdrant connections."""
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            logger.info("Using existing GraphRAG instance")
            return

        logger.info("Initializing GraphRAG component")

        # Load configuration
        self.config = ConfigLoader().load_config()

        # Initialize Neo4j connector
        try:
            logger.info("Connecting to Neo4j")
            self.neo4j = Neo4jConnector(
                uri=self.config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
                user=self.config.get("neo4j", {}).get("user", "neo4j"),
                password=self.config.get("neo4j", {}).get("password", "password"),
            )
            logger.info("Neo4j connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.neo4j = None

        # Initialize Qdrant (with fallback to in-memory mode)
        try:
            logger.info("Connecting to Qdrant")
            qdrant_config = self.config.get("qdrant", {})
            self.qdrant = QdrantVectorStore(
                location=qdrant_config.get("location", ":memory:"),
                collection_name=qdrant_config.get("collection_name", "carbon_ef"),
                embedding_model=self.embedding_model,
            )
            logger.info("Qdrant connection successful")
        except Exception as e:
            logger.warning(
                f"Failed to connect to Qdrant, using in-memory fallback: {e}"
            )
            self.qdrant = QdrantVectorStore(
                location=":memory:",
                collection_name="carbon_ef",
                embedding_model=self.embedding_model,
            )

        # Load embedding model (lazily)
        self._embedding_model = None

        # Flag to track initialization
        self._initialized = True
        logger.info("GraphRAG component initialized successfully")

    @property
    def embedding_model(self):
        """Lazily load embedding model when needed."""
        if self._embedding_model is None:
            try:
                logger.info("Loading embedding model...")
                # Import here to avoid loading model until needed
                from sentence_transformers import SentenceTransformer

                # Free memory before loading
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Load model
                model_name = self.config.get("embedding", {}).get(
                    "model_name", "all-MiniLM-L6-v2"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")

                self._embedding_model = SentenceTransformer(model_name, device=device)
                logger.info(f"Embedding model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using vector search and graph traversal."""
        try:
            logger.info(f"Processing query: {query}")

            # Check if components are available
            if not self.neo4j:
                logger.error("Neo4j connector not available")
                return {"error": "Neo4j connector not available"}

            # Generate query embedding
            try:
                embedding = self.generate_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return {"error": f"Failed to generate embedding: {str(e)}"}

            # Perform vector search in Qdrant
            vector_results = []
            try:
                if self.qdrant:
                    logger.info("Performing vector search")
                    vector_results = self.qdrant.search(embedding, limit=5)
                    logger.info(f"Found {len(vector_results)} vector results")
                else:
                    logger.warning("Qdrant not available, skipping vector search")
            except Exception as e:
                logger.error(f"Error in vector search: {e}")
                # Continue with graph search

            # Perform graph search in Neo4j
            graph_results = []
            try:
                logger.info("Performing graph search")
                # Extract keywords for graph search
                cypher_query = self._generate_cypher_query(query)
                graph_results = self.neo4j.execute_query(cypher_query)
                logger.info(f"Found {len(graph_results)} graph results")
            except Exception as e:
                logger.error(f"Error in graph search: {e}")
                # Continue with whatever results we have

            # Combine results
            results = {
                "query": query,
                "vector_results": vector_results,
                "graph_results": graph_results,
            }

            return results

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {"error": f"Error processing query: {str(e)}"}

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using the embedding model."""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)

            # Convert to list for JSON serialization
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.tolist()
            else:
                embedding = embedding.tolist()

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _generate_cypher_query(self, query: str) -> str:
        """Generate a Cypher query based on the user query."""
        # For now, use a simple query to match nodes containing keywords
        # This can be enhanced with query templates based on the user query
        return """
        MATCH (ef:EmissionFactor)
        WHERE ef.name CONTAINS 'carbon' OR ef.description CONTAINS 'carbon'
        RETURN ef.name AS name, ef.value AS value, ef.unit AS unit, ef.description AS description
        LIMIT 5
        """

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up GraphRAG resources")

        # Close Neo4j connection
        if hasattr(self, "neo4j") and self.neo4j:
            try:
                self.neo4j.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")

        # Clear embedding model
        if hasattr(self, "_embedding_model") and self._embedding_model:
            try:
                del self._embedding_model
                self._embedding_model = None
                logger.info("Embedding model cleared")

                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                logger.error(f"Error clearing embedding model: {e}")
