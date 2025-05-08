"""Configuration loader module."""

import os
from pathlib import Path

import yaml


def load_config(config_path=None):
    """
    Load configuration from a YAML file or from environment variables.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dict containing configuration
    """
    if config_path is None:
        # Try to load from default location
        project_root = Path(__file__).parent.parent.parent.absolute()
        config_path = os.path.join(project_root, "configs", "config.yaml")

    # Try to load from file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        # Fallback to environment variables
        config = {
            "neo4j": {
                "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.environ.get("NEO4J_USER", "neo4j"),
                "password": os.environ.get("NEO4J_PASSWORD", "carbon_ef_secure"),
            },
            "qdrant": {
                "location": os.environ.get("QDRANT_LOCATION", "localhost"),
                "port": int(os.environ.get("QDRANT_PORT", "6333")),
                "collection_name": os.environ.get("QDRANT_COLLECTION", "carbon_ef"),
            },
            "embedding": {
                "model_name": os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            },
        }

    return config
