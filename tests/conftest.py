import pytest
from unittest.mock import MagicMock, patch
import httpx

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI client for all tests."""
    with patch('openai.OpenAI') as mock_openai:
        # Configure the mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="""Document Type: invoice
Confidence: 0.95

Features:
1. Document Information
- Invoice Number: INV-2024-001
- Date: 03/15/2024

2. Financial Details
- Amount: $500.00
- Total Amount: $500.00

3. Document Fields
- Bill To: John Doe"""
                    )
                )
            ]
        )
        mock_openai.return_value = mock_client
        yield mock_openai

@pytest.fixture
def mock_openai_base_response():
    """Base mock for OpenAI API responses."""
    def create_mock_response(content):
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=content
                    )
                )
            ]
        )
    return create_mock_response

@pytest.fixture
def mock_healthcare_response(mock_openai_base_response):
    """Mock response for healthcare document classification."""
    return mock_openai_base_response("""
Document Type: medical_claim
Confidence: 0.95

Features:
1. Patient Information
- Patient ID: PAT-2024-001
- Date: 03/15/2024

2. Medical Codes
- CPT Code: 99213 (Office Visit)
- CPT Code: 80053 (Blood Test)
- ICD-10: J45.901

3. Financial Information
- Amount: $150.00
- Amount: $75.00
- Total Amount: $225.00

4. Provider Information
- Provider: Dr. Jane Smith
- NPI: 1234567890

5. Insurance Details
- Insurance Claim: CLM-2024-456
""")

@pytest.fixture
def mock_financial_response(mock_openai_base_response):
    """Mock response for financial document classification."""
    return mock_openai_base_response("""
Document Type: invoice
Confidence: 0.92

Features:
1. Document Information
- Invoice Number: INV-2024-001
- Date: 03/15/2024

2. Financial Details
- Amount: $1,000.00
- Amount: $500.00
- Total Amount: $1,500.00

3. Payment Information
- Payment Terms: Net 30
- Due Date: 04/14/2024

4. Client Information
- Name: John Doe
- Address: 123 Main St
""")

@pytest.fixture
def mock_mixed_response(mock_openai_base_response):
    """Mock response for mixed industry document."""
    return mock_openai_base_response("""
Document Type: medical_claim
Confidence: 0.6

Features:
1. Medical Features
- Patient ID: PAT-2024-001
- CPT Code: 99213 (Medical Consultation)

2. Financial Features
- Invoice Number: INV-2024-001
- Amount: $150.00

3. Validation Warnings
- Warning: Document contains mixed industry signals
- Warning: Unable to determine primary document type
""")

@pytest.fixture
def mock_incomplete_response(mock_openai_base_response):
    """Mock response for document with missing required features."""
    return mock_openai_base_response("""
Document Type: medical_claim
Confidence: 0.4

Features:
1. Present Features
- Date: 03/15/2024
- Amount: $150.00

2. Missing Features
- Warning: Missing required patient identification
- Warning: Missing required procedure codes

3. Validation Results
- Warning: Document missing critical healthcare fields
- Warning: Low confidence due to missing required information
""")

@pytest.fixture
def mock_error_response():
    """Mock for API error responses."""
    def raise_error(*args, **kwargs):
        raise Exception("API Error")
    mock = MagicMock()
    mock.create.side_effect = raise_error
    return mock