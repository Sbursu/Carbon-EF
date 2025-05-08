import gc
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from src.agent.agent_framework import AgentFramework
from src.cache.semantic_cache import SemanticCache

# Import local modules
from src.graphrag.graph_rag_component import GraphRAGComponent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model singleton instances
graph_rag_instance = None
agent_framework_instance = None
semantic_cache_instance = None


class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the API server."""
    # Initialize components on startup
    try:
        # Use global variables to maintain singleton instances
        global graph_rag_instance, agent_framework_instance, semantic_cache_instance

        logger.info("Initializing components...")

        # Free memory before loading models
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Load GraphRAG component (uses singleton pattern internally)
        logger.info("Initializing GraphRAG component...")
        graph_rag_instance = GraphRAGComponent()

        # Initialize semantic cache
        logger.info("Initializing Semantic Cache...")
        semantic_cache_instance = SemanticCache()

        # Initialize agent framework (after GraphRAG is loaded)
        logger.info("Initializing Agent Framework...")
        agent_framework_instance = AgentFramework()

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Continue with partial functionality if possible

    yield

    # Cleanup on shutdown
    try:
        logger.info("Cleaning up resources...")

        if graph_rag_instance:
            logger.info("Cleaning up GraphRAG component...")
            graph_rag_instance.cleanup()

        if semantic_cache_instance:
            logger.info("Cleaning up Semantic Cache...")
            if hasattr(semantic_cache_instance, "cleanup"):
                semantic_cache_instance.cleanup()

        if agent_framework_instance:
            logger.info("Cleaning up Agent Framework...")
            if hasattr(agent_framework_instance, "cleanup"):
                agent_framework_instance.cleanup()

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info("Cleanup complete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)


# Initialize FastAPI app
app = FastAPI(
    title="Carbon Emission Factor API",
    description="API for carbon emission factor retrieval using Graph RAG",
    version="0.1.0",
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API and its components."""
    components_status = {
        "graph_rag": graph_rag_instance is not None,
        "agent_framework": agent_framework_instance is not None,
        "semantic_cache": semantic_cache_instance is not None,
    }

    all_healthy = all(components_status.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components_status,
    }


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a user query using GraphRAG and the agent framework."""
    try:
        query = request.query
        use_cache = request.use_cache
        logger.info(f"Received query: {query}")

        # Check if components are initialized
        if not graph_rag_instance:
            raise HTTPException(
                status_code=503,
                detail="GraphRAG component not initialized. Try again later.",
            )

        # Try to get result from cache if enabled
        cache_hit = None
        if use_cache and semantic_cache_instance:
            try:
                logger.info("Checking semantic cache...")
                cache_hit = semantic_cache_instance.get(query)
                if cache_hit:
                    logger.info("Cache hit!")
                    return cache_hit
                logger.info("Cache miss")
            except Exception as e:
                logger.error(f"Error accessing cache: {e}")
                # Continue without cache

        # Process with GraphRAG
        logger.info("Processing query with GraphRAG...")
        rag_results = graph_rag_instance.process_query(query)

        # Process with agent if available
        agent_response = None
        if agent_framework_instance:
            try:
                logger.info("Processing with agent framework...")
                agent_response = agent_framework_instance.process(
                    query=query, context=rag_results
                )
            except Exception as e:
                logger.error(f"Error in agent processing: {e}")
                agent_response = "Agent processing failed. Using raw results instead."

        # Prepare response
        response = {
            "query": query,
            "vector_results": rag_results.get("vector_results", []),
            "graph_results": rag_results.get("graph_results", []),
        }

        if agent_response:
            response["agent_response"] = agent_response

        # Cache the result if needed
        if use_cache and semantic_cache_instance and not cache_hit:
            try:
                semantic_cache_instance.add(query, response)
                logger.info("Added result to cache")
            except Exception as e:
                logger.error(f"Error adding to cache: {e}")

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the query: {str(e)}",
        )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time to response headers."""
    import time

    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)
        raise

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request processed in {process_time:.4f} seconds")

    return response


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("API_PORT", 8000))

    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
