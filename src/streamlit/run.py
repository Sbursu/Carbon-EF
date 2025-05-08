import argparse
import logging
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Streamlit app launcher."""
    parser = argparse.ArgumentParser(description="Carbon EF Streamlit Interface")
    parser.add_argument(
        "--port", type=int, default=8501, help="Port to run Streamlit on"
    )
    parser.add_argument(
        "--api-host", type=str, default="localhost", help="API host address"
    )
    parser.add_argument("--api-port", type=int, default=8000, help="API port")

    args = parser.parse_args()

    # Set environment variables for API connection
    os.environ["API_HOST"] = args.api_host
    os.environ["API_PORT"] = str(args.api_port)

    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Prepare the command
    cmd = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.serverAddress",
        "localhost",
        "--theme.primaryColor",
        "#4CAF50",
        "--theme.backgroundColor",
        "#FFFFFF",
        "--theme.secondaryBackgroundColor",
        "#F0F2F6",
        "--theme.textColor",
        "#262730",
    ]

    logger.info(f"Starting Streamlit app on port {args.port}")
    logger.info(f"Connecting to API at {args.api_host}:{args.api_port}")

    try:
        # Run the Streamlit process
        process = subprocess.Popen(cmd)
        logger.info(f"Streamlit app is running. Press Ctrl+C to stop.")
        process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping Streamlit app...")
        process.terminate()
    except Exception as e:
        logger.error(f"Error running Streamlit app: {str(e)}")


if __name__ == "__main__":
    main()
