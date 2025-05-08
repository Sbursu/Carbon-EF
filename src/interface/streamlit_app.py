import logging
import os
import time
import traceback
from typing import Dict, Optional

import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
API_TIMEOUT = 30  # seconds


def load_config():
    """Load configuration for the Streamlit app"""
    # For simplicity, we're hard-coding some values
    return {
        "api": {
            "url": os.environ.get("API_URL", "http://localhost:8000"),
        }
    }


def call_api(
    endpoint: str, data: Dict = None, method: str = "GET", retries: int = MAX_RETRIES
) -> Optional[Dict]:
    """
    Call the API with retry and backoff logic

    Args:
        endpoint: API endpoint to call
        data: Data to send in the request
        method: HTTP method to use
        retries: Number of retries to attempt

    Returns:
        API response as a dictionary or None if failed
    """
    config = load_config()
    api_url = f"{config['api']['url']}/{endpoint.lstrip('/')}"

    for attempt in range(retries):
        try:
            if method.upper() == "GET":
                response = requests.get(api_url, timeout=API_TIMEOUT)
            else:  # POST
                response = requests.post(api_url, json=data, timeout=API_TIMEOUT)

            response.raise_for_status()
            return response.json()

        except ConnectionError as e:
            logger.error(f"Connection error on attempt {attempt+1}/{retries}: {str(e)}")
            if attempt == retries - 1:
                st.error(
                    f"Unable to connect to the API server. Please check if the server is running."
                )
                return None

        except ReadTimeout:
            logger.error(f"Request timed out on attempt {attempt+1}/{retries}")
            if attempt == retries - 1:
                st.error(
                    "Request to the API server timed out. The server might be overloaded."
                )
                return None

        except RequestException as e:
            logger.error(f"Request error on attempt {attempt+1}/{retries}: {str(e)}")
            if attempt == retries - 1:
                st.error(f"Error communicating with the API: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            if attempt == retries - 1:
                st.error(f"An unexpected error occurred: {str(e)}")
                return None

        # Wait before retrying with exponential backoff
        if attempt < retries - 1:
            wait_time = BACKOFF_FACTOR**attempt
            logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
            time.sleep(wait_time)

    return None


def health_check() -> bool:
    """Check if the API is healthy"""
    response = call_api("health", retries=1)
    return response is not None and response.get("status") == "healthy"


def main():
    st.set_page_config(
        page_title="Carbon Emission Factor Assistant",
        page_icon="🌍",
        layout="wide",
    )

    st.title("Carbon Emission Factor Assistant")

    # API health check
    api_status = health_check()
    if api_status:
        st.success("✅ API server is running", icon="✅")
    else:
        st.warning(
            "⚠️ API server is not responding. Some features may not work.", icon="⚠️"
        )

    # User input
    query = st.text_area("Ask about carbon emission factors:", height=100)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
        )
    with col2:
        max_tokens = st.slider(
            "Max response tokens", min_value=64, max_value=512, value=256, step=32
        )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query.")
            return

        if not api_status and not health_check():
            st.error("Cannot process query: API server is not available.")
            return

        with st.spinner("Processing your query..."):
            try:
                # Call the API
                payload = {
                    "query": query,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                start_time = time.time()
                result = call_api("query", data=payload, method="POST")
                total_time = time.time() - start_time

                if result:
                    # Display results
                    for item in result.get("result", []):
                        st.markdown("### Response")
                        st.write(item.get("response", "No response available"))

                        if item.get("reference"):
                            st.markdown("#### References")
                            st.info(item.get("reference"))

                        if item.get("confidence") is not None:
                            st.markdown("#### Confidence")
                            st.progress(item.get("confidence"))

                    st.success(
                        f"Query processed in {total_time:.2f} seconds (API time: {result.get('query_time', 0):.2f} seconds)"
                    )
                else:
                    st.error(
                        "Failed to get a response from the API. Please try again later."
                    )

            except Exception as e:
                logger.error(f"Error in Streamlit app: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
