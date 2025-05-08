import argparse
import logging

from src.agent import AgentFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(description="Carbon Emission Factor RAG System")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file", default=None
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cache before starting"
    )

    args = parser.parse_args()

    # Initialize the agent framework
    logger.info("Initializing the agent framework...")
    agent = AgentFramework(config_path=args.config)

    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing cache...")
        agent.cache.clear()
        logger.info("Cache cleared")

    # Process the query or run in interactive mode
    if args.query:
        # Process a single query
        result = agent.process_query(args.query, use_cache=not args.no_cache)
        print("\n--- QUERY RESULT ---")
        print(f"Query: {result['query']}")
        print(f"\nAnswer: {result['answer']}")
        print("\n--- EXPLANATION ---")
        print(f"{result['explanation']}")

        # Show cache info if available
        if "cache_info" in result:
            cache_used = result["cache_info"].get("used_cache", False)
            print(f"\nCache used: {cache_used}")

            if cache_used and "cache_type" in result["cache_info"]:
                print(f"Cache type: {result['cache_info']['cache_type']}")
                if "similarity" in result["cache_info"]:
                    print(f"Similarity: {result['cache_info']['similarity']:.4f}")
                if "original_query" in result["cache_info"]:
                    print(f"Original query: {result['cache_info']['original_query']}")
    else:
        # Interactive mode
        print("Carbon Emission Factor RAG System")
        print("Type 'exit' to quit")
        print("Type 'clear' to clear the cache")
        print("Commands:")
        print("  !cache on/off - Enable/disable cache")
        print("  !clear - Clear cache")

        use_cache = not args.no_cache

        while True:
            query = input("\nEnter your query: ")

            # Check for special commands
            if query.lower() in ["exit", "quit", "q"]:
                break
            elif query.lower() == "clear" or query.lower() == "!clear":
                agent.cache.clear()
                print("Cache cleared")
                continue
            elif query.lower() == "!cache on":
                use_cache = True
                print("Cache enabled")
                continue
            elif query.lower() == "!cache off":
                use_cache = False
                print("Cache disabled")
                continue

            try:
                result = agent.process_query(query, use_cache=use_cache)
                print("\n--- RESULT ---")
                print(f"Answer: {result['answer']}")

                # Show cache info
                if "cache_info" in result:
                    cache_used = result["cache_info"].get("used_cache", False)
                    print(f"\nCache used: {cache_used}")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"Error: {str(e)}")

    # Clean up
    agent.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
