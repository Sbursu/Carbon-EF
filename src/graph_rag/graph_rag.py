import logging
import re  # Import re here for use in execute_hybrid_query
from typing import Any, Dict, List, Optional, Tuple, Union

from src.graph_rag.neo4j_client import Neo4jClient
from src.utils import load_config
from src.vector_store import EmbeddingGenerator, QdrantStore

logger = logging.getLogger(__name__)


class GraphRAG:
    """
    Graph RAG component for retrieving information from Neo4j and Qdrant
    Implements a hybrid GraphRAG pattern combining vector search and graph traversal
    (Simplified: LLM agent dependency removed)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Graph RAG

        Args:
            config_path: Path to configuration file
        """
        logger.info("GraphRAG: Initializing...")
        self.config = load_config(config_path)
        self.neo4j_client = Neo4jClient(config_path)
        logger.info("GraphRAG: Neo4jClient initialized.")

        # Initialize vector store components
        try:
            logger.info("GraphRAG: Initializing EmbeddingGenerator...")
            self.embedding_generator = EmbeddingGenerator(config_path=config_path)
            logger.info("GraphRAG: EmbeddingGenerator initialized.")
        except Exception as e:
            logger.error(
                f"GraphRAG: Failed to initialize EmbeddingGenerator: {e}", exc_info=True
            )
            self.embedding_generator = None

        try:
            logger.info("GraphRAG: Initializing QdrantStore...")
            self.qdrant_store = QdrantStore(config_path=config_path)
            self.vector_search_available = True
            logger.info("GraphRAG: QdrantStore initialized.")
        except Exception as e:
            logger.warning(
                f"GraphRAG: Failed to initialize QdrantStore: {str(e)}. Vector search will be unavailable.",
                exc_info=True,
            )
            self.qdrant_store = None
            self.vector_search_available = False

        # Removed Phi-2 model initialization
        # self.phi2_model = Phi2Model(config_path)
        logger.info("GraphRAG: Phi2Model dependency removed.")

        # Load query templates (simplified)
        self.query_templates = {
            "basic_ef_lookup": {
                "description": "Basic emission factor lookup by entity and region",
                "params": ["entity_type", "entity_name", "region"],
                "method": self._basic_ef_lookup,
            },
            "regional_comparison": {
                "description": "Compare emission factors across regions",
                "params": ["entity_type", "entity_name", "regions"],
                "method": self._regional_comparison,
            },
            "entity_subgraph": {
                "description": "Get subgraph around an entity",
                "params": ["entity_type", "entity_name", "max_depth"],
                "method": self._entity_subgraph,
            },
            # Removed graphrag_query template as execute_hybrid_query is now the main method
        }
        logger.info("GraphRAG: Query templates loaded.")
        logger.info("GraphRAG: Initialization complete.")

    def execute_query(self, query_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query using a template

        Args:
            query_type: Type of query to execute
            params: Parameters for the query

        Returns:
            Query results
        """
        if query_type not in self.query_templates:
            logger.error(f"Unknown query type: {query_type}")
            return {"error": f"Unknown query type: {query_type}"}

        template = self.query_templates[query_type]
        method = template["method"]

        # Validate required parameters
        missing_params = []
        for required_param in template["params"]:
            if required_param not in params and not required_param.endswith("?"):
                missing_params.append(required_param)

        if missing_params:
            logger.error(f"Missing required parameters: {', '.join(missing_params)}")
            return {
                "error": f"Missing required parameters: {', '.join(missing_params)}"
            }

        # Execute the query method
        try:
            result = method(**params)
            return {"query_type": query_type, "params": params, "result": result}
        except Exception as e:
            logger.error(f"Error executing query {query_type}: {str(e)}")
            return {"query_type": query_type, "params": params, "error": str(e)}

    def execute_hybrid_query(
        self,
        query: str,
        entity_type: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute a hybrid GraphRAG query using both Qdrant and Neo4j.
        (Simplified: Does not use LLM for synthesis)

        Args:
            query: User query
            entity_type: Optional entity type filter
            region: Optional region filter
            limit: Maximum number of results

        Returns:
            Dictionary containing combined vector and graph search results.
        """
        logger.info(f"Executing simplified hybrid GraphRAG query: {query}")

        # Initialize results structure
        results = {
            "query": query,
            "vector_results": [],  # Renamed for clarity
            "graph_results": [],  # Renamed for clarity
            "combined_factors": [],  # Merged results
            "graph_context": "",
            "vector_error": None,
            "graph_error": None,
            "subgraph_error": None,
        }

        # Collected entity IDs for graph traversal
        entity_ids = []
        vector_results_raw = []

        # 1. Vector search with Qdrant if available
        if (
            self.vector_search_available
            and self.qdrant_store
            and self.embedding_generator
        ):
            try:
                query_embedding = self.embedding_generator.embed_query(query)
                filter_conditions = {}
                if entity_type and entity_type != "unknown":
                    filter_conditions["entity_type"] = entity_type
                if region and region != "global":
                    filter_conditions["region"] = region

                vector_results_raw = self.qdrant_store.search(
                    query_vector=query_embedding,
                    limit=limit,
                    filter_conditions=filter_conditions if filter_conditions else None,
                )
                logger.info(f"Vector search returned {len(vector_results_raw)} results")

                # Store raw vector results and extract IDs
                results["vector_results"] = vector_results_raw
                for res in vector_results_raw:
                    res["source"] = "vector"  # Add source marker
                    if "id" in res:
                        entity_ids.append(res["id"])
                    elif "entity_id" in res:
                        entity_ids.append(res["entity_id"])

            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}", exc_info=True)
                results["vector_error"] = str(e)
        else:
            logger.info("Vector search prerequisites not met. Skipping.")

        # 2. Graph search with Neo4j (using specific templates or fallback)
        graph_results_raw = []
        try:
            logger.info("Attempting graph search...")
            # Try basic lookup first if entity name seems present
            entity_name_match = re.search(
                r"for\s+([a-zA-Z0-9\s]+)(?:\s+in|\s*\?|$)", query
            )
            entity_name = (
                entity_name_match.group(1).strip() if entity_name_match else None
            )

            if entity_name:
                logger.info(f"Attempting basic EF lookup for: {entity_name}")
                basic_lookup_params = {"entity_name": entity_name}
                if entity_type and entity_type != "unknown":
                    basic_lookup_params["entity_type"] = entity_type
                if region and region != "global":
                    basic_lookup_params["region"] = region

                graph_lookup_result = self._basic_ef_lookup(**basic_lookup_params)
                graph_results_raw = graph_lookup_result.get("emission_factors", [])
                logger.info(
                    f"Graph basic lookup returned {len(graph_results_raw)} results"
                )

            # Fallback or additional search if needed (e.g., by region)
            if not graph_results_raw and region and region != "global":
                logger.info(f"Attempting graph search by region: {region}")
                region_lookup_result = self.get_emission_factors_by_region(
                    region, entity_type
                )
                # Check if it's a list of dicts
                if isinstance(region_lookup_result, list):
                    graph_results_raw.extend(region_lookup_result)
                logger.info(
                    f"Graph region search added {len(region_lookup_result)} results"
                )

            # Add graph results to the main results dict and extract IDs
            results["graph_results"] = graph_results_raw
            for res in graph_results_raw:
                res["source"] = "graph"  # Add source marker
                ef_id = res.get("id") or res.get("ef_id")
                if ef_id and ef_id not in entity_ids:
                    entity_ids.append(ef_id)

        except Exception as e:
            logger.error(f"Graph search failed: {str(e)}", exc_info=True)
            results["graph_error"] = str(e)

        # 3. Combine results (simple merge and deduplicate based on a key like 'id' or 'ef_id')
        combined_factors = {}
        for item in results["vector_results"] + results["graph_results"]:
            key = item.get("id") or item.get("ef_id")
            if (
                key and key not in combined_factors
            ):  # Prioritize first seen (vector or graph)
                combined_factors[key] = item
            elif (
                not key
            ):  # Handle items without a clear ID, maybe add based on name/region? For now, skip.
                logger.warning(f"Skipping result without ID: {item.get('entity_name')}")

        results["combined_factors"] = list(combined_factors.values())[
            : limit * 2
        ]  # Limit combined results

        # 4. If we have entity IDs, fetch related graph context from Neo4j
        if entity_ids:
            try:
                logger.info(f"Fetching subgraph for {len(entity_ids)} entities...")
                subgraph = self.neo4j_client.get_subgraph_for_entities(entity_ids)
                results["graph_context"] = self._format_graph_context(subgraph)
                logger.info("Subgraph context generated.")
            except Exception as e:
                logger.error(f"Error fetching subgraph: {str(e)}", exc_info=True)
                results["subgraph_error"] = str(e)

        # Remove the now redundant emission_factors key if desired, keep separate vector/graph results
        # results.pop("emission_factors", None)

        logger.info(
            f"Hybrid query finished. Returning {len(results['combined_factors'])} combined factors."
        )
        return results

    def _format_graph_context(self, subgraph: Dict[str, Any]) -> str:
        """
        Format the subgraph as context for the LLM

        Args:
            subgraph: Subgraph from Neo4j

        Returns:
            Formatted context string
        """
        if not subgraph:
            return "No graph context available."

        context = "Graph Context:\n"

        # Format nodes
        if "nodes" in subgraph and subgraph["nodes"]:
            context += "\nNodes:\n"
            for node_type, nodes in subgraph["nodes"].items():
                context += f"  {node_type}:\n"
                for node in nodes[:5]:  # Limit to 5 nodes per type
                    context += f"    - "
                    # Show the most relevant properties
                    if "name" in node:
                        context += f"{node['name']}"
                    elif "entity_name" in node:
                        context += f"{node['entity_name']}"
                    elif "type_name" in node:
                        context += f"{node['type_name']}"
                    elif "region_code" in node:
                        context += f"{node.get('name', '')} ({node['region_code']})"

                    # Add additional important properties
                    if "ef_value" in node:
                        context += (
                            f", Value: {node['ef_value']} {node.get('ef_unit', '')}"
                        )
                    if "confidence" in node:
                        context += f", Confidence: {node['confidence']}"

                    context += "\n"

                if len(nodes) > 5:
                    context += f"    ... and {len(nodes) - 5} more\n"

        # Format relationships
        if "relationships" in subgraph and subgraph["relationships"]:
            context += "\nRelationships:\n"
            for rel_type, rels in subgraph["relationships"].items():
                context += f"  {rel_type}:\n"
                for rel in rels[:5]:  # Limit to 5 relationships per type
                    context += f"    - {rel['source']} -> {rel['target']}"
                    # Add properties if any
                    if "properties" in rel and rel["properties"]:
                        context += " ("
                        props = []
                        for key, value in rel["properties"].items():
                            props.append(f"{key}: {value}")
                        context += ", ".join(props)
                        context += ")"
                    context += "\n"

                if len(rels) > 5:
                    context += f"    ... and {len(rels) - 5} more\n"

        # Add a note about limitations
        if subgraph.get("truncated", False):
            context += "\nNote: The graph context has been truncated due to size limitations.\n"

        return context

    def get_entity_by_name(
        self, entity_name: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get entity by name from Neo4j

        Args:
            entity_name: Name of the entity
            entity_type: Optional entity type

        Returns:
            List of entities matching the name
        """
        # Use Neo4j client to find entities by name
        cypher_query = """
        MATCH (ef:EmissionFactor)
        WHERE toLower(ef.entity_name) CONTAINS toLower($entity_name)
        """

        if entity_type and entity_type != "unknown":
            cypher_query += """
            AND (ef)-[:HAS_ENTITY_TYPE]->(et:EntityType)
            WHERE toLower(et.type_name) CONTAINS toLower($entity_type)
            """

        cypher_query += """
        RETURN ef.ef_id as id, ef.entity_name as name, ef.ef_value as value, 
               ef.ef_unit as unit, ef.confidence as confidence
        LIMIT 10
        """

        params = {"entity_name": entity_name}
        if entity_type and entity_type != "unknown":
            params["entity_type"] = entity_type

        results = self.neo4j_client.run_query(cypher_query, params)
        return results

    def get_emission_factors_by_region(
        self, region_code: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get emission factors for a specific region

        Args:
            region_code: Region code
            entity_type: Optional entity type filter

        Returns:
            List of emission factors for the region
        """
        # Use Neo4j client to find emission factors by region
        cypher_query = """
        MATCH (ef:EmissionFactor)-[:APPLIES_TO_REGION]->(r:Region)
        WHERE toLower(r.region_code) = toLower($region_code)
        """

        if entity_type and entity_type != "unknown":
            cypher_query += """
            AND (ef)-[:HAS_ENTITY_TYPE]->(et:EntityType)
            WHERE toLower(et.type_name) = toLower($entity_type)
            """

        cypher_query += """
        RETURN ef.ef_id as id, ef.entity_name as name, ef.entity_id as entity_id,
               ef.ef_value as value, ef.ef_unit as unit, r.name as region,
               ef.confidence as confidence
        ORDER BY ef.confidence DESC
        LIMIT 20
        """

        params = {"region_code": region_code}
        if entity_type and entity_type != "unknown":
            params["entity_type"] = entity_type

        results = self.neo4j_client.run_query(cypher_query, params)
        return results

    def _basic_ef_lookup(
        self, entity_type: str, entity_name: str, region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Basic emission factor lookup

        Args:
            entity_type: Entity type
            entity_name: Entity name
            region: Optional region code

        Returns:
            Emission factor data
        """
        filters = {"entity_type": entity_type, "entity_name": entity_name}

        if region:
            filters["region"] = region

        emission_factors = self.neo4j_client.find_emission_factors(filters, limit=5)

        if not emission_factors:
            # Try without region constraint if no results
            if region and "region" in filters:
                del filters["region"]
                emission_factors = self.neo4j_client.find_emission_factors(
                    filters, limit=5
                )

                if emission_factors:
                    return {
                        "warning": f"No exact match for region '{region}', showing global or alternative regions",
                        "emission_factors": emission_factors,
                    }

        return {"emission_factors": emission_factors}

    def _regional_comparison(
        self, entity_type: str, entity_name: str, regions: List[str]
    ) -> Dict[str, Any]:
        """
        Compare emission factors across regions

        Args:
            entity_type: Entity type
            entity_name: Entity name
            regions: List of region codes

        Returns:
            Emission factors by region
        """
        regional_efs = self.neo4j_client.get_regional_comparison(
            entity_type, entity_name, regions
        )

        # Calculate statistics
        values = []
        for ef in regional_efs:
            if "ef_value" in ef and "error" not in ef:
                values.append(float(ef["ef_value"]))

        stats = {}
        if values:
            stats["min"] = min(values)
            stats["max"] = max(values)
            stats["avg"] = sum(values) / len(values)
            stats["range"] = stats["max"] - stats["min"]
            stats["count"] = len(values)

        return {"emission_factors": regional_efs, "statistics": stats}

    def _entity_subgraph(
        self, entity_type: str, entity_name: Optional[str] = None, max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get subgraph around an entity

        Args:
            entity_type: Entity type
            entity_name: Optional entity name
            max_depth: Maximum traversal depth

        Returns:
            Subgraph data
        """
        subgraph = self.neo4j_client.get_subgraph_for_entity(
            entity_type, entity_name, max_depth
        )

        # Create a text representation of the graph
        if "nodes" in subgraph and "relationships" in subgraph:
            nodes = subgraph["nodes"]
            relationships = subgraph["relationships"]

            node_count = len(nodes)
            rel_count = len(relationships)

            node_types = {}
            for node in nodes:
                for label in node["labels"]:
                    node_types[label] = node_types.get(label, 0) + 1

            text_representation = (
                f"Subgraph with {node_count} nodes and {rel_count} relationships.\n"
                f"Node types: {', '.join([f'{label} ({count})' for label, count in node_types.items()])}\n"
            )

            # Add notable nodes
            if nodes:
                text_representation += "Notable nodes:\n"
                for i, node in enumerate(nodes[:5]):  # Show at most 5 nodes
                    props = node["properties"]
                    notable_props = {
                        k: v
                        for k, v in props.items()
                        if k in ["entity_name", "entity_type", "ef_value", "region"]
                    }
                    text_representation += f"- {', '.join([f'{k}: {v}' for k, v in notable_props.items()])}\n"

                if len(nodes) > 5:
                    text_representation += f"... and {len(nodes) - 5} more nodes\n"

            subgraph["text_representation"] = text_representation

        return subgraph

    def close(self):
        """
        Close connections and clean up resources
        """
        try:
            self.neo4j_client.close()
        except Exception as e:
            logger.error(f"Error closing Neo4j client: {str(e)}")

        if self.vector_search_available and self.qdrant_store:
            try:
                self.qdrant_store.close()
            except Exception as e:
                logger.error(f"Error closing Qdrant store: {str(e)}")

        logger.info("GraphRAG resources released")
