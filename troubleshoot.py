#!/usr/bin/env python3
"""
Troubleshooting script for Carbon EF components
"""

import importlib
import logging
import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "neo4j",
        "qdrant-client",
        "sentence-transformers",
        "torch",
        "transformers",
        "numpy",
        "psutil",
        "pyyaml",
    ]

    print("\nChecking dependencies:")
    print("======================")

    all_deps_ok = True
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError as e:
            all_deps_ok = False
            print(f"❌ {package}: {str(e)}")

    if not all_deps_ok:
        print("\n⚠️  Missing dependencies detected. Install them with:")
        print("pip install -r requirements.txt")


def check_imports():
    """Check if critical modules can be imported"""
    modules = [
        # Core config
        "src.utils.config",
        # Neo4j
        "src.graph_rag.neo4j_client",
        "src.neo4j.neo4j_connector",
        # Vector store
        "src.vector_store.embedding_generator",
        "src.vector_store.qdrant_store",
        # Phi2 model
        "src.agent.phi2_model",
        # API components
        "src.api.main",
        "src.api.app",
    ]

    print("\nChecking module imports:")
    print("=======================")

    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
        except Exception as e:
            print(f"❌ {module} (ERROR): {str(e)}")


def check_neo4j():
    """Test the Neo4j connection"""
    print("\nTesting Neo4j connection:")
    print("========================")

    try:
        from neo4j import GraphDatabase

        # Try to get credentials from config
        try:
            from src.utils import load_config

            config = load_config()
            uri = (
                config.get("graph_db", {})
                .get("neo4j", {})
                .get("uri", "bolt://localhost:7687")
            )
            user = config.get("graph_db", {}).get("neo4j", {}).get("user", "neo4j")
            password = (
                config.get("graph_db", {})
                .get("neo4j", {})
                .get("password", "carbon_ef_secure")
            )
        except:
            # Fallback to environment variables or defaults
            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "carbon_ef_secure")

        print(f"Connecting to: {uri}")

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            driver.close()
            print("✅ Neo4j connection successful")

        except Exception as e:
            print(f"❌ Neo4j connection failed: {str(e)}")
            print("\nPossible solutions:")
            print("1. Ensure Neo4j is running on the correct port")
            print("2. Check username/password in config.yaml")
            print("3. Try connecting to Neo4j Browser at http://localhost:7474")

    except ImportError:
        print("❌ Neo4j driver not installed. Install with: pip install neo4j")


def check_model_loading():
    """Test loading the embedding model"""
    print("\nTesting embedding model loading:")
    print("==============================")

    try:
        from sentence_transformers import SentenceTransformer

        model_name = "all-MiniLM-L6-v2"  # Small model for testing
        print(f"Loading model: {model_name}")

        try:
            model = SentenceTransformer(model_name)
            print(f"✅ Model loaded successfully")

            # Test embedding generation
            text = "Test embedding generation"
            embedding = model.encode(text)
            print(f"✅ Embedding generated successfully (dim={len(embedding)})")

        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            print("\nPossible solutions:")
            print("1. Check internet connection for model download")
            print("2. Ensure PyTorch is installed correctly")
            print("3. Check for available disk space")

    except ImportError:
        print(
            "❌ sentence-transformers not installed. Install with: pip install sentence-transformers"
        )


def main():
    """Main troubleshooting function"""
    print("\n" + "=" * 60)
    print("Carbon EF Troubleshooting Tool")
    print("=" * 60)

    check_dependencies()
    check_imports()
    check_neo4j()
    check_model_loading()

    print("\n" + "=" * 60)
    print("Troubleshooting complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
