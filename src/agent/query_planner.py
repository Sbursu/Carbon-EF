import json
import logging
from typing import Any, Dict, List, Optional, Union

from src.agent.phi2_model import Phi2Model
from src.utils import load_config

logger = logging.getLogger(__name__)


class QueryPlanner:
    """
    Query planner for decomposing complex queries into simpler subqueries
    """

    def __init__(self, phi2_model: Phi2Model, config_path: Optional[str] = None):
        """
        Initialize the query planner

        Args:
            phi2_model: Phi-2 model instance
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.phi2_model = phi2_model

        # Define query types and templates
        self.query_types = {
            "basic_ef_lookup": {
                "description": "Basic emission factor lookup by entity and region",
                "params": ["entity_type", "entity_name", "region"],
                "template": "What is the emission factor for {entity_name} in {region}?",
            },
            "regional_comparison": {
                "description": "Compare emission factors across regions",
                "params": ["entity_type", "entity_name", "regions"],
                "template": "Compare the emission factor for {entity_name} between {regions}.",
            },
            "entity_subgraph": {
                "description": "Get subgraph around an entity",
                "params": ["entity_type", "entity_name", "max_depth"],
                "template": "Show me the relationships for {entity_name}.",
            },
            "hybrid_lookup": {
                "description": "Use both graph and vector search for lookup",
                "params": ["entity_type", "entity_name", "region"],
                "template": "Find detailed information about {entity_name} in {region}.",
            },
        }

        logger.info("Query planner initialized")

    def decompose_query(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Decompose a complex query into subqueries

        Args:
            query: User query
            analysis: Query analysis

        Returns:
            List of subquery specifications
        """
        # For simple comparison queries, handle directly
        if (
            analysis.get("comparison", False)
            and "regions" in analysis
            and len(analysis["regions"]) > 1
        ):
            entity_type = analysis.get("entity_type", "unknown")
            entity_name = analysis.get("entity_name", "unknown")
            regions = analysis["regions"]

            # Create a regional comparison subquery
            return [
                {
                    "query_type": "regional_comparison",
                    "subquery": f"Compare the emission factor for {entity_name} between {', '.join(regions)}",
                    "params": {
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "regions": regions,
                    },
                }
            ]

        # For more complex queries, use Phi-2 to decompose
        return self._decompose_with_phi2(query, analysis)

    def _decompose_with_phi2(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Use Phi-2 model to decompose complex queries

        Args:
            query: User query
            analysis: Query analysis

        Returns:
            List of subquery specifications
        """
        # Create prompt with context about available query types
        query_types_info = "\n".join(
            [
                f"- {query_type}: {info['description']}"
                for query_type, info in self.query_types.items()
            ]
        )

        prompt = f"""[INST] You are an AI assistant specialized in environmental emission factors. 
For complex queries, break them down into simpler subqueries that can be answered independently.

Available query types:
{query_types_info}

For each subquery, specify:
1. The query_type to use
2. A clear subquery text
3. The required parameters for that query type

The input query analysis shows:
- Entity type: {analysis.get('entity_type', 'unknown')}
- Entity name: {analysis.get('entity_name', 'unknown')}
- Region: {analysis.get('region', 'global')}
- Is comparison: {analysis.get('comparison', False)}
- Regions to compare: {', '.join(analysis.get('regions', []))}

Format your response as a JSON array of objects, where each object has these fields:
- query_type: The type of query (from the available types listed above)
- subquery: The text of the subquery
- params: Dictionary of parameters needed for this query type

Complex Query: {query} [/INST]"""

        response = self.phi2_model.generate(prompt)

        # Try to extract JSON from the response
        try:
            subqueries = json.loads(response)
            logger.info(f"Query decomposition: {len(subqueries)} subqueries")
            return subqueries
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {response}")

            # Fallback: create a simple subquery based on analysis
            entity_type = analysis.get("entity_type", "unknown")
            entity_name = analysis.get("entity_name", "unknown")
            region = analysis.get("region", "global")

            return [
                {
                    "query_type": "hybrid_lookup",
                    "subquery": query,
                    "params": {
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "region": region,
                    },
                }
            ]

    def select_retrieval_strategy(self, query: str, analysis: Dict[str, Any]) -> str:
        """
        Select the best retrieval strategy for a query

        Args:
            query: User query
            analysis: Query analysis

        Returns:
            Retrieval strategy name ("graph", "vector", or "hybrid")
        """
        # For comparison queries, prefer graph
        if analysis.get("comparison", False):
            return "graph"

        # For specific entity lookups, prefer hybrid
        if (
            analysis.get("entity_type", "unknown") != "unknown"
            and analysis.get("entity_name", "unknown") != "unknown"
        ):
            return "hybrid"

        # For general queries, prefer vector
        return "vector"

    def generate_cypher_query(
        self, query_type: str, params: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a Cypher query for a query type and parameters

        Args:
            query_type: Query type
            params: Query parameters

        Returns:
            Cypher query or None if not applicable
        """
        if query_type == "basic_ef_lookup":
            entity_type = params.get("entity_type", "unknown")
            entity_name = params.get("entity_name", "unknown")
            region = params.get("region", "global")

            if region and region != "global":
                return f"""
                MATCH (ef:EF {{entity_type: "{entity_type}", entity_name: "{entity_name}"}})
                MATCH (ef)-[:PRODUCED_IN]->(r:Region {{country_code: "{region}"}})
                RETURN ef {{.*}} as emission_factor
                LIMIT 5
                """
            else:
                return f"""
                MATCH (ef:EF {{entity_type: "{entity_type}", entity_name: "{entity_name}"}})
                RETURN ef {{.*}} as emission_factor
                LIMIT 5
                """

        elif query_type == "regional_comparison":
            entity_type = params.get("entity_type", "unknown")
            entity_name = params.get("entity_name", "unknown")
            regions = params.get("regions", [])

            if regions:
                regions_str = ", ".join([f'"{r}"' for r in regions])
                return f"""
                MATCH (ef:EF {{entity_type: "{entity_type}", entity_name: "{entity_name}"}})
                MATCH (ef)-[:PRODUCED_IN]->(r:Region)
                WHERE r.country_code IN [{regions_str}]
                RETURN ef {{.*}} as emission_factor, r.country_code as region
                """

        return None
