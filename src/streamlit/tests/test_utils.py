import json
import logging
import os

# Import from the Streamlit app
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the streamlit directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Mock the API call function
@patch("requests.get")
def test_call_api_get_success(mock_get):
    """Test the call_api function with GET method success."""
    from src.streamlit.app import call_api

    # Configure mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": "success"}
    mock_get.return_value = mock_response

    # Call the function
    result = call_api("/test", method="GET", params={"param1": "value1"})

    # Assertions
    assert result == {"message": "success"}
    mock_get.assert_called_once()


@patch("requests.get")
def test_call_api_get_error(mock_get):
    """Test the call_api function with GET method error."""
    from src.streamlit.app import call_api

    # Configure mock to raise an exception
    mock_get.side_effect = Exception("Connection error")

    # Call the function
    result = call_api("/test", method="GET")

    # Assertions
    assert "error" in result
    assert "Connection error" in result["error"]
    mock_get.assert_called_once()


@patch("requests.post")
def test_call_api_post_success(mock_post):
    """Test the call_api function with POST method success."""
    from src.streamlit.app import call_api

    # Configure mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_post.return_value = mock_response

    # Call the function
    result = call_api("/query", method="POST", data={"query": "test query"})

    # Assertions
    assert result == {"result": "success"}
    mock_post.assert_called_once()


@patch("requests.post")
def test_call_api_post_error_response(mock_post):
    """Test the call_api function with POST method error response."""
    from src.streamlit.app import call_api

    # Configure mock
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    mock_post.return_value = mock_response

    # Call the function
    result = call_api("/query", method="POST", data={"query": "test query"})

    # Assertions
    assert "error" in result
    assert "500" in result["error"]
    assert "Internal server error" in result["error"]
    mock_post.assert_called_once()


def test_call_api_invalid_method():
    """Test the call_api function with an invalid method."""
    from src.streamlit.app import call_api

    # Call the function with an invalid method
    result = call_api("/test", method="PUT")

    # Assertions
    assert "error" in result
    assert "Unsupported method: PUT" in result["error"]


if __name__ == "__main__":
    pytest.main(["-xvs", "test_utils.py"])
