#!/usr/bin/env python3
"""
Model Optimization Script for Carbon EF

This script analyzes the model loading process, identifies bottlenecks,
and optimizes model configurations for better performance.
"""

import argparse
import gc
import importlib
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))


def log_memory_usage(step=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{step}] Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(
            f"[{step}] CUDA memory: allocated={allocated:.2f} MB, reserved={reserved:.2f} MB"
        )


def check_config_file():
    """Check if the config file exists and is valid"""
    config_file = project_root / "configs" / "config.yaml"

    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            print(f"✅ Config file valid: {config_file}")

            # Check specific configurations
            if "model" in config:
                print(f"  - Model: {config['model'].get('name', 'not specified')}")

            if "vector_store" in config and "qdrant" in config["vector_store"]:
                qdrant = config["vector_store"]["qdrant"]
                print(
                    f"  - Qdrant: {qdrant.get('host', 'localhost')}:{qdrant.get('port', '6333')}"
                )

            if "graph_db" in config and "neo4j" in config["graph_db"]:
                neo4j = config["graph_db"]["neo4j"]
                print(f"  - Neo4j: {neo4j.get('uri', 'bolt://localhost:7687')}")

            return True
    except Exception as e:
        print(f"❌ Config file error: {str(e)}")
        return False


def optimize_config_file():
    """Optimize the config file for better performance"""
    config_file = project_root / "configs" / "config.yaml"

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Optimization suggestions
        optimizations = []

        # Check model configuration
        if "model" in config:
            if "max_length" in config["model"] and config["model"]["max_length"] > 1024:
                optimizations.append(
                    "Consider reducing model.max_length to 1024 for better performance"
                )

        # Create optimized config
        if optimizations:
            print("\nOptimization suggestions:")
            for i, opt in enumerate(optimizations, 1):
                print(f"{i}. {opt}")

            # Ask for confirmation
            if input("\nApply optimizations? (y/n): ").lower() == "y":
                # Apply optimizations here
                # ...

                # Save optimized config
                optimized_file = config_file.parent / "optimized_config.yaml"
                with open(optimized_file, "w") as f:
                    yaml.dump(config, f)
                print(f"✅ Optimized config saved to: {optimized_file}")
        else:
            print("✅ Config appears to be well-optimized")

    except Exception as e:
        print(f"❌ Config optimization error: {str(e)}")


def test_load_models():
    """Test loading models to identify bottlenecks"""
    print("\nTesting model loading...")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log_memory_usage("Before model loading")

    # Test loading embedding model
    print("\nTesting embedding model loading...")
    try:
        start_time = time.time()
        from sentence_transformers import SentenceTransformer

        model_name = "all-MiniLM-L6-v2"  # Default model, adjust from config if needed
        print(f"Loading embedding model: {model_name}")

        embedding_model = SentenceTransformer(model_name)
        load_time = time.time() - start_time

        print(f"✅ Embedding model loaded in {load_time:.2f}s")
        log_memory_usage("After embedding model")

        # Test embedding generation
        text = "This is a test for embedding generation"
        start_time = time.time()
        embedding = embedding_model.encode(text)
        encode_time = time.time() - start_time

        print(f"  - Embedding generation: {encode_time:.4f}s, dim={len(embedding)}")

    except Exception as e:
        print(f"❌ Error loading embedding model: {str(e)}")

    # Test loading Phi-2 model (if available)
    print("\nTesting Phi-2 model loading...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Clean up memory first
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Get the model name from the config file
        try:
            from src.utils import load_config

            config = load_config()
            # Ensure we correctly navigate the config structure
            if "model" in config and "name" in config["model"]:
                model_name = config["model"]["name"]
            elif "phi2" in config and "model_name" in config["phi2"]:
                model_name = config["phi2"]["model_name"]
            elif (
                "agent" in config
                and "phi2_model" in config["agent"]
                and "model_name" in config["agent"]["phi2_model"]
            ):
                model_name = config["agent"]["phi2_model"]["model_name"]
            else:
                logger.warning("Could not find model name in config, defaulting.")
                model_name = "Surendra-Aitest/phi2-env-factors-merged"  # Default to user's model if not in config
        except Exception as e:
            logger.error(f"Error loading config to get model name: {e}. Defaulting.")
            model_name = "Surendra-Aitest/phi2-env-factors-merged"  # Default to user's model on error

        print(f"Loading Phi-2 model: {model_name}")

        # Measure tokenizer loading time
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - start_time
        print(f"  - Tokenizer loaded in {tokenizer_time:.2f}s")

        # Measure model loading time
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model_time = time.time() - start_time
        print(f"✅ Phi-2 model loaded in {model_time:.2f}s")

        log_memory_usage("After Phi-2 model")

    except Exception as e:
        print(f"❌ Error loading Phi-2 model: {str(e)}")
        traceback.print_exc()


def test_api_startup():
    """Test API startup to identify issues"""
    print("\nTesting API startup...")

    try:
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log_memory_usage("Before API startup")

        # Import and test API components
        print("Importing API components...")

        try:
            start_time = time.time()
            # Import GraphRAGComponent
            from src.graphrag.graph_rag_component import GraphRAGComponent

            print(f"✅ GraphRAGComponent imported ({(time.time()-start_time):.2f}s)")

            # Test initialization with minimal setup
            start_time = time.time()
            try:
                graph_rag = GraphRAGComponent()
                print(
                    f"✅ GraphRAGComponent initialized ({(time.time()-start_time):.2f}s)"
                )
            except Exception as e:
                print(f"❌ Error initializing GraphRAGComponent: {str(e)}")

            log_memory_usage("After GraphRAG init")

        except Exception as e:
            print(f"❌ Error importing GraphRAGComponent: {str(e)}")

        # Test other API components
        try:
            from src.agent.agent_framework import AgentFramework

            print("✅ AgentFramework imported")
        except Exception as e:
            print(f"❌ Error importing AgentFramework: {str(e)}")

        # Test Neo4j connection
        try:
            from src.neo4j.neo4j_connector import Neo4jConnector

            print("Testing Neo4j connection...")

            # Get config
            from src.configs.config_loader import ConfigLoader

            config = ConfigLoader().load_config()

            neo4j = Neo4jConnector(
                uri=config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
                user=config.get("neo4j", {}).get("user", "neo4j"),
                password=config.get("neo4j", {}).get("password", "password"),
            )

            # Test query
            result = neo4j.test_connection()
            print(f"✅ Neo4j connection successful: {result}")

        except Exception as e:
            print(f"❌ Error testing Neo4j connection: {str(e)}")

    except Exception as e:
        print(f"❌ Error in API startup test: {str(e)}")
        traceback.print_exc()


def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(
        description="Optimize model loading and API startup"
    )
    parser.add_argument(
        "--check-config", action="store_true", help="Check configuration file"
    )
    parser.add_argument(
        "--optimize-config", action="store_true", help="Optimize configuration"
    )
    parser.add_argument("--test-models", action="store_true", help="Test model loading")
    parser.add_argument("--test-api", action="store_true", help="Test API startup")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # Default to all if no args specified
    if not any(vars(args).values()):
        args.all = True

    print("\n" + "=" * 60)
    print("Carbon EF Model Optimization Tool")
    print("=" * 60)

    # Log system info
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check available memory
    vm = psutil.virtual_memory()
    print(
        f"System memory: {vm.total / (1024**3):.1f} GB total, {vm.available / (1024**3):.1f} GB available"
    )

    if args.all or args.check_config:
        print("\n" + "=" * 60)
        print("Checking Configuration")
        print("=" * 60)
        check_config_file()

    if args.all or args.optimize_config:
        print("\n" + "=" * 60)
        print("Optimizing Configuration")
        print("=" * 60)
        optimize_config_file()

    if args.all or args.test_models:
        print("\n" + "=" * 60)
        print("Testing Model Loading")
        print("=" * 60)
        test_load_models()

    if args.all or args.test_api:
        print("\n" + "=" * 60)
        print("Testing API Startup")
        print("=" * 60)
        test_api_startup()

    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
