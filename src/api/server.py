import argparse
import gc
import importlib
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import psutil
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

# Use AgentFramework instead of the non-existent EFAgent
# from src.agent.agent import EFAgent
from src.agent.agent_framework import AgentFramework

# Commenting out imports from non-existent module
# from src.models.models import load_embedding_model, load_llm
from src.utils import load_config

# The import below seems incorrect, GraphRAG is likely imported differently
# from src.graphrag.graphrag import GraphRAG


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Singleton pattern for models and components
SHARED_RESOURCES = {}


# Memory monitoring
def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logger.info(
            f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB"
        )
        logger.info(
            f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models and resources
    try:
        logger.info("Starting API server and loading resources...")
        log_memory_usage()

        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Import components lazily to avoid circular imports
        from src.agent.rag_agent import GraphRAGAgent
        from src.graphrag.graph_rag_component import GraphRAGComponent

        # Initialize shared resources once
        SHARED_RESOURCES["graph_rag"] = GraphRAGComponent()
        SHARED_RESOURCES["agent"] = GraphRAGAgent(SHARED_RESOURCES["graph_rag"])

        logger.info("Resources loaded successfully")
        log_memory_usage()

    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        logger.error(traceback.format_exc())
        # We don't exit here - allow the API to start without models
        # so we can at least serve health endpoints and debugging info

    yield

    # Cleanup on shutdown
    logger.info("Shutting down API server...")
    for resource_name, resource in SHARED_RESOURCES.items():
        logger.info(f"Cleaning up {resource_name}...")
        try:
            # Call dispose/cleanup method if available
            if hasattr(resource, "cleanup") and callable(resource.cleanup):
                resource.cleanup()
            elif hasattr(resource, "dispose") and callable(resource.dispose):
                resource.dispose()
            elif hasattr(resource, "close") and callable(resource.close):
                resource.close()
        except Exception as e:
            logger.error(f"Error cleaning up {resource_name}: {str(e)}")

    # Clear memory
    SHARED_RESOURCES.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete")


# Request and response models
class QueryRequest(BaseModel):
    query: str
    temperature: float = Field(0.7, ge=0, le=1.0)
    max_tokens: int = Field(256, ge=64, le=512)


class ResponseItem(BaseModel):
    response: str
    confidence: Optional[float] = None
    reference: Optional[str] = None


class QueryResponse(BaseModel):
    result: List[ResponseItem]
    query_time: float


# Initialize FastAPI app
app = FastAPI(title="Carbon Emission Factor API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handling middleware
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Internal server error"},
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Carbon Emission Factor API"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    status = "healthy"
    details = {}

    # Check if models are loaded
    if not SHARED_RESOURCES:
        status = "degraded"
        details["reason"] = "Models not loaded"

    # Add more health checks as needed
    return {
        "status": status,
        "timestamp": time.time(),
        "details": details,
        "memory": {
            "ram_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            "cuda_allocated_mb": (
                torch.cuda.memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
        },
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query and return results"""

    # Check if models are available
    if "agent" not in SHARED_RESOURCES:
        raise HTTPException(
            status_code=503, detail="Models not loaded. Please check server logs."
        )

    start_time = time.time()

    try:
        agent = SHARED_RESOURCES["agent"]

        # Process the query
        result = agent.process_query(
            request.query,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Create response
        if isinstance(result, str):
            # Simple string result
            response = QueryResponse(
                result=[ResponseItem(response=result)],
                query_time=time.time() - start_time,
            )
        elif isinstance(result, dict):
            # Dictionary result
            response = QueryResponse(
                result=[
                    ResponseItem(
                        response=result.get("response", "No response"),
                        confidence=result.get("confidence"),
                        reference=result.get("reference"),
                    )
                ],
                query_time=time.time() - start_time,
            )
        elif isinstance(result, list):
            # List of results
            response = QueryResponse(
                result=[
                    (
                        ResponseItem(**item)
                        if isinstance(item, dict)
                        else ResponseItem(response=str(item))
                    )
                    for item in result
                ],
                query_time=time.time() - start_time,
            )
        else:
            # Fallback for unexpected types
            response = QueryResponse(
                result=[ResponseItem(response=str(result))],
                query_time=time.time() - start_time,
            )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    finally:
        # Clean up to prevent memory leaks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.get("/debug/memory")
async def debug_memory():
    """Debug endpoint to check memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "process_memory_mb": memory_info.rss / 1024 / 1024,
        "python_objects": {
            "gc_count": gc.get_count(),
            "gc_objects": len(gc.get_objects()),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "allocated_mb": (
                torch.cuda.memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
            "reserved_mb": (
                torch.cuda.memory_reserved() / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
        },
        "resources_loaded": list(SHARED_RESOURCES.keys()),
    }


def main():
    """Main entry point for the API server"""
    parser = argparse.ArgumentParser(description="Carbon Emission Factor API Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file", default=None
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    api_config = config.get("api", {})

    # Override with command line arguments if provided
    host = args.host or api_config.get("host", "0.0.0.0")
    port = args.port or api_config.get("port", 8000)
    reload = args.reload or api_config.get("reload", False)

    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Auto-reload: {reload}")

    # Clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run the server
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=api_config.get(
            "workers", 1
        ),  # Multiple workers can cause memory issues
    )


if __name__ == "__main__":
    main()
