import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent import AgentFramework
from src.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Carbon Emission Factor API",
    description="API for retrieving carbon emission factors data",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Dependency to get the agent
def get_agent():
    try:
        agent = AgentFramework()
        yield agent
        agent.close()
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing agent: {str(e)}"
        )


# Response models
class QueryResponse(BaseModel):
    query: str
    answer: str
    explanation: Optional[str] = None
    cache_used: Optional[bool] = None


class EmissionFactor(BaseModel):
    entity_type: str
    entity_name: str
    region: str
    ef_value: float
    ef_unit: str
    source: Optional[str] = None
    year: Optional[int] = None
    confidence: Optional[float] = None


class EmissionFactorsResponse(BaseModel):
    emission_factors: List[EmissionFactor]
    metadata: Optional[Dict[str, Any]] = None


class GraphRAGRequest(BaseModel):
    query: str
    entity_type: Optional[str] = None
    region: Optional[str] = None


class GraphRAGResponse(BaseModel):
    answer: str
    graph_context: str
    vector_results: List[Dict[str, Any]]
    subgraph: Optional[Dict[str, Any]] = None


# Endpoints
@app.get("/")
async def root():
    return {"message": "Carbon Emission Factor API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def process_query(
    query: str, use_cache: bool = True, agent: AgentFramework = Depends(get_agent)
):
    """
    Process a natural language query about carbon emission factors
    """
    try:
        result = agent.process_query(query, use_cache=use_cache)

        response = QueryResponse(
            query=result["query"],
            answer=result["answer"],
            explanation=result.get("explanation"),
            cache_used=result.get("cache_info", {}).get("used_cache", False),
        )

        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/emission-factors", response_model=EmissionFactorsResponse)
async def get_emission_factors(
    entity_type: str = Query(
        ..., description="Type of entity (e.g., electricity, fuel, transport)"
    ),
    entity_name: str = Query(..., description="Name of the entity"),
    region: Optional[str] = Query(
        None, description="Region code (e.g., US, EU, global)"
    ),
    agent: AgentFramework = Depends(get_agent),
):
    """
    Get emission factors for a specific entity and region
    """
    try:
        # Use the graph RAG component directly for this query
        result = agent.graph_rag.execute_query(
            "basic_ef_lookup",
            {"entity_type": entity_type, "entity_name": entity_name, "region": region},
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        emission_factors = result.get("result", {}).get("emission_factors", [])

        # Transform to response model format
        response_efs = []
        for ef in emission_factors:
            response_efs.append(
                EmissionFactor(
                    entity_type=ef.get("entity_type", ""),
                    entity_name=ef.get("entity_name", ""),
                    region=ef.get("region", ""),
                    ef_value=float(ef.get("ef_value", 0)),
                    ef_unit=ef.get("ef_unit", ""),
                    source=ef.get("source", "graph"),
                    year=ef.get("year", None),
                    confidence=ef.get("confidence", None),
                )
            )

        return EmissionFactorsResponse(
            emission_factors=response_efs,
            metadata={
                "count": len(response_efs),
                "warning": result.get("result", {}).get("warning"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving emission factors: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving emission factors: {str(e)}"
        )


@app.get("/regions/compare", response_model=Dict[str, Any])
async def compare_regions(
    entity_type: str = Query(..., description="Type of entity"),
    entity_name: str = Query(..., description="Name of the entity"),
    regions: List[str] = Query(..., description="List of region codes to compare"),
    agent: AgentFramework = Depends(get_agent),
):
    """
    Compare emission factors across multiple regions
    """
    try:
        result = agent.graph_rag.execute_query(
            "regional_comparison",
            {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "regions": regions,
            },
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result.get("result", {})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing regions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error comparing regions: {str(e)}"
        )


@app.post("/graphrag", response_model=GraphRAGResponse)
async def graphrag_query(
    request: GraphRAGRequest, agent: AgentFramework = Depends(get_agent)
):
    """
    Process a query using GraphRAG (hybrid Qdrant vector search + Neo4j graph traversal)
    """
    try:
        # Execute the GraphRAG query
        result = agent.graph_rag.execute_hybrid_query(
            query=request.query, entity_type=request.entity_type, region=request.region
        )

        # Prepare the response
        response = GraphRAGResponse(
            answer=result.get("answer", "No answer generated"),
            graph_context=result.get("graph_context", ""),
            vector_results=result.get("vector_results", []),
            subgraph=result.get("subgraph", None),
        )

        return response
    except Exception as e:
        logger.error(f"Error in GraphRAG query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in GraphRAG query: {str(e)}"
        )


@app.get("/cache/clear")
async def clear_cache(agent: AgentFramework = Depends(get_agent)):
    """
    Clear the semantic cache
    """
    try:
        agent.cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")
