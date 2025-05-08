import json
import logging
from typing import Any, Dict, List, Optional, Union

from src.agent.phi2_model import Phi2Model
from src.agent.query_planner import QueryPlanner
from src.cache import SemanticCache
from src.graph_rag.graph_rag import GraphRAG
from src.utils import load_config
from src.vector_store import EmbeddingGenerator, QdrantStore

logger = logging.getLogger(__name__)


class AgentFramework:
    """
    Agent framework for handling emission factor queries

    Coordinates between Phi-2 model, vector search, and graph search
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the agent framework

        Args:
            config_path: Path to configuration file
        """
        logger.info("AgentFramework: Initializing...")  # Log start
        self.config = load_config(config_path)
        logger.info("AgentFramework: Config loaded.")  # Log step

        # Initialize components
        logger.info("AgentFramework: Initializing components...")

        # Initialize cache
        try:
            logger.info("AgentFramework: Initializing SemanticCache...")
            self.cache = SemanticCache(config_path)
            logger.info("AgentFramework: SemanticCache initialized.")
        except Exception as e:
            logger.error(
                f"AgentFramework: Failed to initialize SemanticCache: {e}",
                exc_info=True,
            )
            self.cache = None  # Allow partial init

        # Initialize Phi-2 model
        try:
            logger.info("AgentFramework: Initializing Phi2Model...")
            self.phi2_model = Phi2Model(config_path)
            logger.info("AgentFramework: Phi2Model initialized.")
        except Exception as e:
            logger.error(
                f"AgentFramework: Failed to initialize Phi2Model: {e}", exc_info=True
            )
            raise Exception(
                "Phi2Model is critical, cannot continue without it."
            )  # Critical component

        # Initialize query planner
        try:
            logger.info("AgentFramework: Initializing QueryPlanner...")
            self.query_planner = QueryPlanner(self.phi2_model, config_path)
            logger.info("AgentFramework: QueryPlanner initialized.")
        except Exception as e:
            logger.error(
                f"AgentFramework: Failed to initialize QueryPlanner: {e}", exc_info=True
            )
            raise Exception(
                "QueryPlanner is critical, cannot continue without it."
            )  # Critical component

        # Initialize vector search components
        try:
            logger.info("AgentFramework: Initializing EmbeddingGenerator...")
            self.embedding_generator = EmbeddingGenerator(config_path=config_path)
            logger.info("AgentFramework: EmbeddingGenerator initialized.")
        except Exception as e:
            logger.error(
                f"AgentFramework: Failed to initialize EmbeddingGenerator: {e}",
                exc_info=True,
            )
            self.embedding_generator = None  # Allow partial init

        try:
            logger.info("AgentFramework: Initializing QdrantStore...")
            self.vector_store = QdrantStore(config_path=config_path)
            logger.info("AgentFramework: QdrantStore initialized.")
        except Exception as e:
            logger.warning(
                f"AgentFramework: Failed to initialize QdrantStore: {e}. Some vector search features may be unavailable.",
                exc_info=True,
            )
            self.vector_store = None

        # Initialize graph search component
        try:
            logger.info("AgentFramework: Initializing GraphRAG...")
            self.graph_rag = GraphRAG(config_path=config_path)
            logger.info("AgentFramework: GraphRAG initialized.")
        except Exception as e:
            logger.warning(
                f"AgentFramework: Failed to initialize GraphRAG: {e}. Graph search features may be unavailable.",
                exc_info=True,
            )
            self.graph_rag = None

        logger.info("AgentFramework: All components initialized successfully")

    def process_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a user query

        Args:
            query: User query
            use_cache: Whether to use the cache

        Returns:
            Processed results
        """
        logger.info(f"Processing query: {query}")

        # 1. Analyze the query
        analysis = self.phi2_model.analyze_query(query)
        logger.info(f"Query analysis: {analysis}")

        # 2. Check cache if enabled
        if use_cache:
            cached_result = self.cache.get(query, analysis)
            if cached_result:
                logger.info(f"Cache hit for query: {query}")
                # Add cache information to the result
                if "cache_info" not in cached_result:
                    cached_result["cache_info"] = {}
                cached_result["cache_info"]["used_cache"] = True
                return cached_result

        # 3. Determine if decomposition is needed
        if self._is_complex_query(query, analysis):
            # Process complex query using query planner
            subqueries = self.query_planner.decompose_query(query, analysis)
            results = self._process_subqueries(subqueries)
        else:
            # Process simple query
            results = self._process_simple_query(query, analysis)

        # 4. Synthesize results
        answer = self.phi2_model.synthesize_results(query, results)

        # 5. Generate explanation (optional)
        explanation = self.phi2_model.explain_reasoning(query, results, answer)

        # 6. Prepare final result
        result = {
            "query": query,
            "analysis": analysis,
            "results": results,
            "answer": answer,
            "explanation": explanation,
            "cache_info": {"used_cache": False},
        }

        # 7. Cache the result if enabled
        if use_cache:
            self.cache.set(query, result, analysis)

        return result

    def _process_subqueries(
        self, subqueries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of subqueries

        Args:
            subqueries: List of subquery specifications

        Returns:
            List of query results
        """
        results = []

        for subquery in subqueries:
            subquery_type = subquery.get("query_type", "basic_ef_lookup")
            subquery_params = subquery.get("params", {})
            subquery_text = subquery.get("subquery", "")

            # Execute the subquery based on type
            if subquery_type.startswith("graph_") or subquery_type in [
                "basic_ef_lookup",
                "regional_comparison",
                "entity_subgraph",
            ]:
                # Graph-based query
                graph_type = subquery_type.replace("graph_", "")
                subquery_result = self.graph_rag.execute_query(
                    graph_type, subquery_params
                )

                # Extract result portion if available
                if "result" in subquery_result:
                    result_data = subquery_result["result"]
                else:
                    result_data = subquery_result

                results.append(
                    {
                        "query_type": subquery_type,
                        "subquery": subquery_text,
                        "params": subquery_params,
                        "result": result_data,
                    }
                )

            elif subquery_type == "hybrid_lookup":
                # Use both graph and vector search with weighting
                graph_result = self._try_graph_lookup(subquery_params)
                vector_result = self._vector_search(
                    subquery_text,
                    subquery_params.get("entity_type", None),
                    subquery_params.get("region", None),
                )

                # Combine results with ranking
                combined_result = self._combine_results(graph_result, vector_result)

                results.append(
                    {
                        "query_type": "hybrid_lookup",
                        "subquery": subquery_text,
                        "params": subquery_params,
                        "result": combined_result,
                    }
                )

            else:
                # Default to vector search for other query types
                vector_result = self._vector_search(
                    subquery_text,
                    subquery_params.get("entity_type", None),
                    subquery_params.get("region", None),
                )

                results.append(
                    {
                        "query_type": "vector_search",
                        "subquery": subquery_text,
                        "params": subquery_params,
                        "result": vector_result,
                    }
                )

        return results

    def _process_simple_query(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a simple query that doesn't need decomposition

        Args:
            query: User query
            analysis: Query analysis

        Returns:
            List of query results
        """
        entity_type = analysis.get("entity_type", "unknown")
        entity_name = analysis.get("entity_name", "unknown")
        region = analysis.get("region", "global")

        # Determine if this is a comparison query
        if (
            analysis.get("comparison", False)
            and "regions" in analysis
            and analysis["regions"]
        ):
            # Regional comparison query
            graph_result = self.graph_rag.execute_query(
                "regional_comparison",
                {
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "regions": analysis["regions"],
                },
            )

            return [
                {
                    "query_type": "regional_comparison",
                    "params": {
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "regions": analysis["regions"],
                    },
                    "result": graph_result.get("result", {}),
                }
            ]
        else:
            # Select retrieval strategy
            strategy = self.query_planner.select_retrieval_strategy(query, analysis)

            if strategy == "graph":
                # Use graph-based lookup
                graph_result = self._try_graph_lookup(
                    {
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "region": region,
                    }
                )

                return [
                    {
                        "query_type": "graph_lookup",
                        "params": {
                            "entity_type": entity_type,
                            "entity_name": entity_name,
                            "region": region,
                        },
                        "result": graph_result,
                    }
                ]

            elif strategy == "vector":
                # Use vector search
                vector_result = self._vector_search(query, entity_type, region)

                return [
                    {
                        "query_type": "vector_search",
                        "params": {
                            "query": query,
                            "entity_type": entity_type,
                            "region": region,
                        },
                        "result": vector_result,
                    }
                ]

            else:  # hybrid
                # Try hybrid approach (graph + vector)
                graph_result = self._try_graph_lookup(
                    {
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "region": region,
                    }
                )

                vector_result = self._vector_search(query, entity_type, region)

                # Combine the results
                combined_result = self._combine_results(graph_result, vector_result)

                return [
                    {
                        "query_type": "hybrid_lookup",
                        "params": {
                            "entity_type": entity_type,
                            "entity_name": entity_name,
                            "region": region,
                        },
                        "result": combined_result,
                    }
                ]

    def _try_graph_lookup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try to look up emission factors using graph-based approach

        Args:
            params: Query parameters

        Returns:
            Graph query results
        """
        entity_type = params.get("entity_type", "unknown")
        entity_name = params.get("entity_name", "unknown")
        region = params.get("region", "global")

        try:
            graph_result = self.graph_rag.execute_query(
                "basic_ef_lookup",
                {
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "region": region,
                },
            )

            # Extract result portion
            if "result" in graph_result:
                return graph_result["result"]
            return graph_result
        except Exception as e:
            logger.error(f"Error in graph lookup: {str(e)}")
            return {"error": str(e), "emission_factors": []}

    def _is_good_graph_result(self, result: Dict[str, Any]) -> bool:
        """
        Check if graph result is good enough to use

        Args:
            result: Graph query result

        Returns:
            True if the result is good, False otherwise
        """
        if "error" in result:
            return False

        if "emission_factors" not in result:
            return False

        emission_factors = result["emission_factors"]
        return len(emission_factors) > 0

    def _is_good_vector_result(self, result: Dict[str, Any]) -> bool:
        """
        Check if vector result is good enough to use

        Args:
            result: Vector search result

        Returns:
            True if the result is good, False otherwise
        """
        if "vector_results" not in result:
            return False

        vector_results = result["vector_results"]
        if not vector_results or len(vector_results) == 0:
            return False

        # Check if top result has high score
        if "score" in vector_results[0] and vector_results[0]["score"] < 0.7:
            return False

        return True

    def _combine_results(
        self, graph_result: Dict[str, Any], vector_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine graph and vector results

        Args:
            graph_result: Graph query result
            vector_result: Vector search result

        Returns:
            Combined result
        """
        combined_result = {
            "emission_factors": [],
            "sources": {
                "graph": bool(self._is_good_graph_result(graph_result)),
                "vector": bool(self._is_good_vector_result(vector_result)),
            },
            "graph_context": vector_result.get("graph_context", ""),
        }

        # Add graph results
        if self._is_good_graph_result(graph_result):
            for ef in graph_result.get("emission_factors", []):
                ef["source"] = "graph"
                combined_result["emission_factors"].append(ef)

        # Add vector results
        if self._is_good_vector_result(vector_result):
            for vr in vector_result.get("vector_results", []):
                # Skip duplicates (if already added from graph)
                is_duplicate = False
                for ef in combined_result["emission_factors"]:
                    if (
                        vr.get("entity_name") == ef.get("entity_name")
                        and vr.get("entity_type") == ef.get("entity_type")
                        and vr.get("region") == ef.get("region")
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    vr["source"] = "vector"
                    combined_result["emission_factors"].append(vr)

        # Rerank the results (graph results first, then by score/relevance)
        combined_result["emission_factors"].sort(
            key=lambda x: (
                0 if x.get("source") == "graph" else 1,  # Graph results first
                -x.get("score", 0) if "score" in x else 0,  # Then by score
            )
        )

        # Limit to top 10
        combined_result["emission_factors"] = combined_result["emission_factors"][:10]

        return combined_result

    def _is_complex_query(self, query: str, analysis: Dict[str, Any]) -> bool:
        """
        Determine if a query is complex and needs decomposition

        Args:
            query: User query
            analysis: Query analysis

        Returns:
            True if the query is complex, False otherwise
        """
        # Check for comparison queries
        if (
            analysis.get("comparison", False)
            and "regions" in analysis
            and len(analysis["regions"]) > 1
        ):
            return True

        # Check for multiple entities or specific keywords
        complex_keywords = [
            "compare",
            "difference",
            "versus",
            "vs",
            "multiple",
            "relationship",
            "trends",
        ]
        for keyword in complex_keywords:
            if keyword in query.lower():
                return True

        # Check for query length (longer queries tend to be more complex)
        if len(query.split()) > 15:
            return True

        return False

    def _vector_search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform vector search for a query

        Args:
            query: User query
            entity_type: Optional entity type filter
            region: Optional region filter
            limit: Maximum number of results

        Returns:
            Vector search results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Prepare filters
        filter_conditions = {}
        if entity_type and entity_type != "unknown":
            filter_conditions["entity_type"] = entity_type

        if region and region != "global":
            filter_conditions["region"] = region

        # Perform search
        search_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        # Get graph context for top result if available
        graph_context = ""
        if search_results and len(search_results) > 0:
            top_result = search_results[0]
            if "entity_type" in top_result and "entity_name" in top_result:
                try:
                    subgraph = self.graph_rag.execute_query(
                        "entity_subgraph",
                        {
                            "entity_type": top_result["entity_type"],
                            "entity_name": top_result["entity_name"],
                            "max_depth": 1,
                        },
                    )

                    if (
                        "result" in subgraph
                        and "text_representation" in subgraph["result"]
                    ):
                        graph_context = subgraph["result"]["text_representation"]
                except Exception as e:
                    logger.error(f"Error getting graph context: {str(e)}")

        return {"vector_results": search_results, "graph_context": graph_context}

    def close(self):
        """
        Close connections and clean up resources
        """
        if hasattr(self, "graph_rag") and self.graph_rag:
            try:
                self.graph_rag.close()
            except Exception as e:
                logger.error(f"Error closing GraphRAG: {str(e)}")

        if hasattr(self, "vector_store") and self.vector_store:
            try:
                self.vector_store.close()
            except Exception as e:
                logger.error(f"Error closing vector store: {str(e)}")

        if hasattr(self, "cache"):
            try:
                self.cache.close()
            except Exception as e:
                logger.error(f"Error closing cache: {str(e)}")

        logger.info("Agent framework resources released")
