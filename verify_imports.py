#!/usr/bin/env python3
"""
Verification script to check that all required modules are importable
"""

import importlib
import os
import sys

# Configure the environment
os.environ["TEST_MODE"] = "1"  # Prevents full initialization of some components

# Define modules to check
modules_to_check = [
    # API components
    "src.api.app",
    "src.api.server",
    # Streamlit components
    "src.streamlit.app",
    "src.streamlit.run",
    # Agent components
    "src.agent",
    "src.agent.agent_framework",
    "src.agent.phi2_model",
    "src.agent.query_planner",
    # Graph RAG components
    "src.graph_rag",
    "src.graph_rag.graph_rag",
    "src.graph_rag.neo4j_client",
    # Vector store components
    "src.vector_store",
    # Cache components
    "src.cache",
    # Utilities
    "src.utils",
]

# Try importing each module
print("Verifying imports...")
print("=====================")

all_ok = True
for module in modules_to_check:
    try:
        imported_module = importlib.import_module(module)
        print(f"✅ {module}")
    except Exception as e:
        all_ok = False
        print(f"❌ {module}: {str(e)}")

# Check for Neo4j connectivity
print("\nVerifying Neo4j connection...")
print("===========================")
try:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "carbon_ef_secure")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        test_value = result.single()["test"]
        if test_value == 1:
            print(f"✅ Neo4j connection successful: {uri}")
        else:
            print(f"❌ Neo4j connection test returned unexpected value")
            all_ok = False
    driver.close()
except Exception as e:
    all_ok = False
    print(f"❌ Neo4j connection failed: {str(e)}")

# Check for Qdrant connectivity (optional)
print("\nVerifying Qdrant connection...")
print("============================")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    # Just check if we can get a response from the server
    client.get_collections()
    print(f"✅ Qdrant connection successful: {qdrant_host}:{qdrant_port}")
except Exception as e:
    # This is not a critical error if Qdrant is not yet set up
    print(f"⚠️ Qdrant connection not verified: {str(e)}")
    print("   This is OK if Qdrant is not yet set up.")

# Summary
print("\nVerification Summary")
print("===================")
if all_ok:
    print("✅ All critical checks passed!")
    sys.exit(0)
else:
    print(
        "❌ Some checks failed. Please fix the issues before running the application."
    )
    sys.exit(1)
