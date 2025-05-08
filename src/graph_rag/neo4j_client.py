import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from neo4j import GraphDatabase
from src.utils import load_config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j client for connecting to the emission factor knowledge graph
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Neo4j client

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.graph_config = self.config["graph_db"]["neo4j"]

        self.uri = self.graph_config["uri"]
        self.user = self.graph_config["user"]
        self.password = self.graph_config.get(
            "password", ""
        )  # Password might be passed separately
        self.database = self.graph_config["database"]

        logger.info(f"Connecting to Neo4j at {self.uri}")
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password) if self.password else None
        )

        # Test connection
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        """
        Close the Neo4j connection
        """
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_emission_factor_by_id(self, ef_id: str) -> Optional[Dict[str, Any]]:
        """
        Get emission factor by ID

        Args:
            ef_id: Emission factor ID

        Returns:
            Emission factor data or None if not found
        """
        query = """
        MATCH (ef:EF {ef_id: $ef_id})
        RETURN ef {.*} as emission_factor
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, ef_id=ef_id)
            record = result.single()

            if record:
                return record["emission_factor"]
            return None

    def get_emission_factors_by_region(
        self, region: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get emission factors for a specific region

        Args:
            region: Region/country code
            limit: Maximum number of results

        Returns:
            List of emission factors
        """
        query = """
        MATCH (ef:EF)-[:PRODUCED_IN]->(r:Region {country_code: $region})
        RETURN ef {.*} as emission_factor
        LIMIT $limit
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, region=region, limit=limit)
            return [record["emission_factor"] for record in result]

    def get_emission_factors_by_entity(
        self, entity_type: str, entity_name: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get emission factors by entity type and optional name

        Args:
            entity_type: Entity type (product, sector, etc.)
            entity_name: Optional entity name
            limit: Maximum number of results

        Returns:
            List of emission factors
        """
        params = {"entity_type": entity_type, "limit": limit}

        if entity_name:
            query = """
            MATCH (ef:EF {entity_type: $entity_type, entity_name: $entity_name})
            RETURN ef {.*} as emission_factor
            LIMIT $limit
            """
            params["entity_name"] = entity_name
        else:
            query = """
            MATCH (ef:EF {entity_type: $entity_type})
            RETURN ef {.*} as emission_factor
            LIMIT $limit
            """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [record["emission_factor"] for record in result]

    def get_emission_factors_by_industry(
        self, industry_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get emission factors for a specific industry

        Args:
            industry_id: Industry ID
            limit: Maximum number of results

        Returns:
            List of emission factors
        """
        query = """
        MATCH (i:Industry {industry_id: $industry_id})-[:HAS_IMPACT]->(ef:EF)
        RETURN ef {.*} as emission_factor
        LIMIT $limit
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, industry_id=industry_id, limit=limit)
            return [record["emission_factor"] for record in result]

    def find_emission_factors(
        self, filters: Dict[str, Any], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find emission factors using filters

        Args:
            filters: Dictionary of filters (field: value)
            limit: Maximum number of results

        Returns:
            List of emission factors
        """
        # Build filter conditions
        filter_conditions = []
        params = {"limit": limit}

        for field, value in filters.items():
            if field == "region":
                # Handle region filter separately (relationship)
                filter_region = True
                params["region"] = value
            else:
                # Add property filter
                filter_conditions.append(f"ef.{field} = ${field}")
                params[field] = value

        # Construct query
        filter_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"

        if filter_region:
            query = f"""
            MATCH (ef:EF)-[:PRODUCED_IN]->(r:Region {{country_code: $region}})
            WHERE {filter_clause}
            RETURN ef {{.*}} as emission_factor
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (ef:EF)
            WHERE {filter_clause}
            RETURN ef {{.*}} as emission_factor
            LIMIT $limit
            """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [record["emission_factor"] for record in result]

    def get_related_emission_factors(
        self, ef_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get related emission factors

        Args:
            ef_id: Emission factor ID
            limit: Maximum number of results

        Returns:
            List of related emission factors
        """
        query = """
        MATCH (ef:EF {ef_id: $ef_id})-[:PRODUCED_IN]->(r:Region)
        MATCH (other:EF)-[:PRODUCED_IN]->(r)
        WHERE other.ef_id <> ef.ef_id
        AND other.entity_type = ef.entity_type
        RETURN other {.*} as emission_factor
        LIMIT $limit
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, ef_id=ef_id, limit=limit)
            return [record["emission_factor"] for record in result]

    def get_regional_comparison(
        self, entity_type: str, entity_name: str, regions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get emission factors for entity across multiple regions

        Args:
            entity_type: Entity type
            entity_name: Entity name
            regions: List of region codes

        Returns:
            List of emission factors for the regions
        """
        query = """
        MATCH (ef:EF {entity_type: $entity_type, entity_name: $entity_name})-[:PRODUCED_IN]->(r:Region)
        WHERE r.country_code IN $regions
        RETURN ef {.*} as emission_factor, r.country_code as region
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query, entity_type=entity_type, entity_name=entity_name, regions=regions
            )
            ef_by_region = {}

            for record in result:
                ef = record["emission_factor"]
                ef["region"] = record["region"]  # Ensure region is included
                ef_by_region[record["region"]] = ef

            # Return emission factors in the order of requested regions
            return [
                ef_by_region.get(
                    region, {"region": region, "error": "No data available"}
                )
                for region in regions
            ]

    def get_subgraph_for_entity(
        self, entity_type: str, entity_name: Optional[str] = None, max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get subgraph for an entity

        Args:
            entity_type: Entity type
            entity_name: Optional entity name
            max_depth: Maximum traversal depth

        Returns:
            Subgraph data
        """
        params = {"entity_type": entity_type, "max_depth": max_depth}
        if entity_name:
            where_clause = (
                "WHERE ef.entity_type = $entity_type AND ef.entity_name = $entity_name"
            )
            params["entity_name"] = entity_name
        else:
            where_clause = "WHERE ef.entity_type = $entity_type"

        query = f"""
        MATCH (ef:EF) {where_clause}
        CALL apoc.path.expandConfig(ef, {{
            relationshipFilter: "*",
            minLevel: 1,
            maxLevel: $max_depth
        }})
        YIELD path
        RETURN path
        LIMIT 100
        """

        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, **params)

                # Process paths into a subgraph
                nodes = {}
                relationships = []

                for record in result:
                    path = record["path"]

                    # Add nodes
                    for node in path.nodes:
                        if node.id not in nodes:
                            nodes[node.id] = {
                                "id": node.id,
                                "labels": list(node.labels),
                                "properties": dict(node),
                            }

                    # Add relationships
                    for rel in path.relationships:
                        relationships.append(
                            {
                                "id": rel.id,
                                "type": rel.type,
                                "start_node": rel.start_node.id,
                                "end_node": rel.end_node.id,
                                "properties": dict(rel),
                            }
                        )

                return {"nodes": list(nodes.values()), "relationships": relationships}
            except Exception as e:
                logger.error(f"Error getting subgraph: {str(e)}")
                # Return the attempt with error
                return {"nodes": [], "relationships": [], "error": str(e)}

    def get_subgraph_for_entities(
        self, entity_ids: List[str], max_depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get subgraph centered around multiple entities

        Args:
            entity_ids: List of entity IDs
            max_depth: Maximum traversal depth

        Returns:
            Subgraph data
        """
        if not entity_ids:
            return {"nodes": [], "relationships": []}

        query = """
        MATCH (n)
        WHERE id(n) IN $entity_ids
        CALL apoc.path.expandConfig(n, {
            relationshipFilter: "*",
            labelFilter: "+EmissionFactor",
            minLevel: 0,
            maxLevel: $max_depth,
            limit: 100
        })
        YIELD path
        WITH DISTINCT nodes(path) as nodes, relationships(path) AS rels
        UNWIND nodes AS node
        WITH COLLECT(DISTINCT node) AS allNodes, rels
        UNWIND rels AS rel
        RETURN allNodes, COLLECT(DISTINCT rel) AS allRels
        """

        try:
            # Convert string IDs to integers if needed
            numeric_ids = []
            for id_str in entity_ids:
                try:
                    if isinstance(id_str, str) and id_str.isdigit():
                        numeric_ids.append(int(id_str))
                    else:
                        numeric_ids.append(id_str)
                except ValueError:
                    # Keep as is if conversion fails
                    numeric_ids.append(id_str)

            with self.driver.session() as session:
                result = session.run(query, entity_ids=numeric_ids, max_depth=max_depth)
                record = result.single()

                if not record:
                    return {"nodes": [], "relationships": []}

                nodes = []
                for node in record["allNodes"]:
                    node_data = {
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node),
                    }
                    nodes.append(node_data)

                relationships = []
                for rel in record["allRels"]:
                    rel_data = {
                        "id": rel.id,
                        "type": rel.type,
                        "start_node_id": rel.start_node.id,
                        "end_node_id": rel.end_node.id,
                        "properties": dict(rel),
                    }
                    relationships.append(rel_data)

                # Enrich relationships with node data
                for rel in relationships:
                    start_node = next(
                        (n for n in nodes if n["id"] == rel["start_node_id"]), None
                    )
                    end_node = next(
                        (n for n in nodes if n["id"] == rel["end_node_id"]), None
                    )
                    if start_node and end_node:
                        rel["start_node"] = start_node
                        rel["end_node"] = end_node

                return {"nodes": nodes, "relationships": relationships}
        except Exception as e:
            logger.error(f"Error getting subgraph for entities: {str(e)}")
            return {"nodes": [], "relationships": []}
