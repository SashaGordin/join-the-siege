import pytest
from unittest.mock import MagicMock, patch
import httpx

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI client for all tests."""
    with patch('httpx.Client') as mock_http_client, \
         patch('openai.OpenAI') as mock_openai:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""Document Type: invoice
Confidence: 0.95

Features:
1. Dates:
2024-03-20

2. Amounts:
$500.00

3. Document Numbers:
Invoice Number: INV-2024-001

4. Key Phrases:
INVOICE
Total Amount

5. Document Fields:
Bill To: John Doe"""
                )
            )
        ]

        # Configure the mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Configure the mock http client with a basic MagicMock
        mock_http_client.return_value = MagicMock()
        mock_http_client.return_value.follow_redirects = True
        mock_http_client.return_value.timeout = 60.0

        yield mock_openai