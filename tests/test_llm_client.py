
import pytest
import requests
from unittest.mock import Mock, patch

from llm_client import generate_text


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generate_text("   ")


@patch("llm_client.requests.post")
def test_successful_response(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "RAG retrieves relevant information."
    }
    mock_post.return_value = mock_response

    answer = generate_text("Explain RAG")

    assert answer == "RAG retrieves relevant information."
    mock_post.assert_called_once()


@patch("llm_client.requests.post")
def test_http_error_is_propagated(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = (
        requests.HTTPError("Server error")
    )
    mock_post.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        generate_text("Explain RAG")


@patch("llm_client.requests.post")
def test_timeout_is_propagated(mock_post):
    mock_post.side_effect = requests.Timeout("Request timed out")

    with pytest.raises(requests.Timeout):
        generate_text("Explain RAG")
