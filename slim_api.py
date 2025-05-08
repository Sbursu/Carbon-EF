#!/usr/bin/env python3
"""
Slim API Server for Carbon EF

A lightweight version of the API that loads only essential components
with fallbacks to ensure successful startup. This version prioritizes
reliability over full functionality.
"""

import gc
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Define request/response models
class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    memory: Dict[str, float]


# Singleton component instances (loaded lazily)
components = {
    "neo4j": None,
    "embedding_model": None,
    "qdrant": None,
    "graph_rag": None,
}


def load_config():
    """Load configuration with fallbacks"""
    try:
        # Try to import the project's config loader
        try:
            from src.utils import load_config as project_load_config

            return project_load_config()
        except ImportError:
            # Fallback to basic config
            import yaml

            config_path = os.path.join(project_root, "configs", "config.yaml")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        return {
            "api": {"host": "0.0.0.0", "port": 8000},
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "carbon_ef_secure",
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "emission_factors",
            },
        }


def get_neo4j_client():
    """Get Neo4j client with lazy loading and fallbacks"""
    global components

    if components["neo4j"] is None:
        try:
            # Try loading the project's Neo4j client
            try:
                from src.graph_rag.neo4j_client import Neo4jClient

                components["neo4j"] = Neo4jClient()
                logger.info("Loaded project Neo4j client")
            except ImportError:
                # Fallback to basic Neo4j client
                from neo4j import GraphDatabase

                config = load_config()
                neo4j_config = config.get("graph_db", {}).get("neo4j", {})

                uri = neo4j_config.get("uri", "bolt://localhost:7687")
                user = neo4j_config.get("user", "neo4j")
                password = neo4j_config.get("password", "")

                driver = GraphDatabase.driver(uri, auth=(user, password))

                class BasicNeo4jClient:
                    def __init__(self, driver):
                        self.driver = driver

                    def execute_query(self, query, params=None):
                        with self.driver.session() as session:
                            result = session.run(query, params or {})
                            return [dict(record) for record in result]

                    def close(self):
                        self.driver.close()

                components["neo4j"] = BasicNeo4jClient(driver)
                logger.info("Loaded basic Neo4j client")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j client: {e}")
            # Return None but don't store in components
            return None

    return components["neo4j"]


def get_embedding_model():
    """Get embedding model with lazy loading"""
    global components

    if components["embedding_model"] is None:
        try:
            # Only load if explicitly requested to avoid memory usage
            from sentence_transformers import SentenceTransformer

            # Use a small, fast model
            model_name = "all-MiniLM-L6-v2"
            device = "cpu"  # Always use CPU for reliability

            logger.info(f"Loading embedding model {model_name} on {device}...")
            model = SentenceTransformer(model_name, device=device)

            components["embedding_model"] = model
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None

    return components["embedding_model"]


def get_qdrant_client():
    """Get Qdrant client with lazy loading"""
    global components

    if components["qdrant"] is None:
        try:
            from qdrant_client import QdrantClient

            config = load_config()
            qdrant_config = config.get("vector_store", {}).get("qdrant", {})

            host = qdrant_config.get("host", "localhost")
            port = qdrant_config.get("port", 6333)

            client = QdrantClient(host=host, port=port)

            class BasicQdrantStore:
                def __init__(self, client):
                    self.client = client
                    self.collection_name = qdrant_config.get(
                        "collection_name", "emission_factors"
                    )

                def search(self, query_vector, limit=5, filter_conditions=None):
                    try:
                        results = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=query_vector,
                            limit=limit,
                            query_filter=filter_conditions,
                        )
                        return [
                            {"id": hit.id, "score": hit.score, "payload": hit.payload}
                            for hit in results
                        ]
                    except Exception as e:
                        logger.error(f"Search error: {e}")
                        return []

                def close(self):
                    # No specific cleanup needed for client
                    pass

            components["qdrant"] = BasicQdrantStore(client)
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            return None

    return components["qdrant"]


def get_graph_rag():
    """Get GraphRAG component with lazy loading."""
    global components
    if components.get("graph_rag") is None:
        try:
            logger.info("Attempting to import GraphRAG...")
            # Import directly from the source file
            from src.graph_rag.graph_rag import GraphRAG

            logger.info("GraphRAG imported successfully.")

            # Initialize GraphRAG (which now doesn't load Phi2)
            logger.info("Initializing GraphRAG...")
            components["graph_rag"] = GraphRAG()
            logger.info("GraphRAG initialized successfully.")
        except ImportError as e:
            logger.error(f"Could not import GraphRAG: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG: {e}", exc_info=True)
            return None

    return components.get("graph_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """API lifecycle manager with minimal initialization"""
    try:
        logger.info("Starting slim GraphRAG API server...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize Neo4j and GraphRAG at startup (or lazy load GraphRAG too if preferred)
        get_neo4j_client()
        get_graph_rag()

    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    # Cleanup
    try:
        logger.info("Shutting down API server...")
        # Close Neo4j connection
        if components.get("neo4j"):
            try:
                components["neo4j"].close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j: {e}")

        # Close GraphRAG component
        if components.get("graph_rag"):
            try:
                components["graph_rag"].close()
                logger.info("GraphRAG component closed")
            except Exception as e:
                logger.error(f"Error closing GraphRAG: {e}")

        # Free up memory
        for key in list(components.keys()):
            components[key] = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="Carbon EF Slim GraphRAG API",
    description="Lightweight API for Carbon Emission Factor data using GraphRAG",
    version="0.2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Carbon EF Slim API - Running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with component status"""
    neo4j_available = components.get("neo4j") is not None
    embedding_available = components.get("embedding_model") is not None
    qdrant_available = components.get("qdrant") is not None
    graph_rag_available = components.get("graph_rag") is not None

    import psutil

    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    cuda_memory = 0
    if torch.cuda.is_available():
        cuda_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    components_status = {
        "neo4j": neo4j_available,
        "embedding_model": embedding_available,
        "qdrant": qdrant_available,
        "graph_rag": graph_rag_available,
    }

    # Status depends on core components being available
    status = "healthy" if neo4j_available and graph_rag_available else "degraded"

    return {
        "status": status,
        "components": components_status,
        "memory": {"ram_mb": ram_usage, "cuda_mb": cuda_memory},
    }


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a user query using the AgentFramework (loads LLM)."""
    logger.info(f"Received query in /query endpoint: {request.query}")
    try:
        query = request.query
        logger.info(f"Processing query: {query}")

        # Try to use the Agent Framework first if available
        logger.info("Attempting to get Agent Framework...")
        agent = get_agent_framework()

        if agent:
            logger.info("Agent Framework instance obtained.")
            try:
                logger.info("Processing query with Agent Framework...")
                # Use the agent's process_query method
                result = agent.process_query(query, use_cache=request.use_cache)
                logger.info("Agent Framework processing successful.")
                # Return the full agent result structure
                return result
            except Exception as e:
                logger.error(
                    f"Agent Framework processing failed: {e}. Aborting query.",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500, detail=f"Agent processing error: {str(e)}"
                )
        else:
            logger.error("Agent Framework not available or failed to load.")
            raise HTTPException(
                status_code=503, detail="Agent Framework not available."
            )

    except HTTPException as http_exc:
        # Re-raise HTTPException to ensure proper status codes
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing query: {str(e)}"
        )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    """Run the slim API server"""
    # Get port from environment or config
    config = load_config()
    port = int(os.environ.get("API_PORT", config.get("api", {}).get("port", 8000)))

    print(f"\n{'='*60}")
    print(f" Carbon EF Slim API Server")
    print(f"{'='*60}")
    print(f" - Running on port: {port}")
    print(f" - CUDA available: {torch.cuda.is_available()}")
    print(f" - Memory-efficient mode: enabled")
    print(f"{'='*60}\n")

    # Run the server
    uvicorn.run(
        "slim_api:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
