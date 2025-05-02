import pytest
from pathlib import Path
import io
from PIL import Image
from src.app import app
import logging

@pytest.fixture
def client():
    """Create a test client for our Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF file for testing."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch

    pdf_path = tmp_path / "test.pdf"

    # Create PDF with proper header
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.7\n")  # PDF header
        f.write(b"% Testing PDF\n")  # Comment

    # Now add content using reportlab
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Add metadata
    c.setTitle("Test Invoice")
    c.setAuthor("Test System")
    c.setSubject("Test Document")

    # Add content with proper formatting
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, 10*inch, "INVOICE")

    c.setFont("Helvetica", 12)
    c.drawString(1*inch, 9*inch, "Invoice Number: INV-2024-001")
    c.drawString(1*inch, 8.5*inch, "Amount: $1500")

    c.save()

    return pdf_path

@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    file_path = tmp_path / "test.txt"
    content = """BANK STATEMENT
    Account Number: 1234567890
    Statement Period: March 1-31, 2024
    Opening Balance: $5000
    Closing Balance: $5800
    """
    file_path.write_text(content)
    return file_path

@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image file for testing."""
    img_path = tmp_path / "test.png"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(img_path)
    return img_path

def test_classify_file_success_pdf(client, sample_pdf):
    """Test successful classification of a PDF file."""
    with open(sample_pdf, 'rb') as f:
        data = {'file': (f, 'test.pdf')}
        response = client.post('/classify_file', data=data)

    assert response.status_code == 200
    json_data = response.get_json()

    assert 'classification' in json_data
    assert 'document_type' in json_data['classification']
    assert 'confidence' in json_data['classification']
    assert 'features' in json_data['classification']
    assert json_data['file_info']['mime_type'] == 'application/pdf'

def test_classify_file_success_text(client, sample_text_file):
    """Test successful classification of a text file."""
    with open(sample_text_file, 'rb') as f:
        data = {'file': (f, 'test.txt')}
        response = client.post('/classify_file', data=data)

    print(f"[TEST LOG] Response status: {response.status_code}")
    json_data = response.get_json()
    print(f"[TEST LOG] Full JSON response: {json_data}")
    print(f"[TEST LOG] Document type: {json_data['classification'].get('document_type')}")
    print(f"[TEST LOG] Confidence: {json_data['classification'].get('confidence')}")
    print(f"[TEST LOG] File info: {json_data.get('file_info')}")

    assert response.status_code == 200
    assert json_data['classification']['document_type'] == 'bank_statement'
    assert json_data['classification']['confidence'] > 0.8
    assert json_data['file_info']['mime_type'].startswith('text/')

def test_classify_file_no_file(client):
    """Test classification endpoint with no file."""
    response = client.post('/classify_file')
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_classify_file_empty_filename(client):
    """Test classification endpoint with empty filename."""
    data = {'file': (io.BytesIO(b''), '')}
    response = client.post('/classify_file', data=data)
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_preview_file_success(client, sample_pdf):
    """Test successful file preview generation."""
    with open(sample_pdf, 'rb') as f:
        data = {'file': (f, 'test.pdf')}
        response = client.post('/preview_file', data=data)
    print("[TEST LOG] Response status:", response.status_code)
    print("[TEST LOG] Response data:", response.data)
    if response.status_code != 200:
        try:
            print("[TEST LOG] Response JSON:", response.get_json())
        except Exception as e:
            print(f"[TEST LOG] Could not parse JSON: {e}")
    assert response.status_code == 200

def test_preview_file_no_file(client):
    """Test preview endpoint with no file."""
    response = client.post('/preview_file')
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_preview_file_empty_filename(client):
    """Test preview endpoint with empty filename."""
    data = {'file': (io.BytesIO(b''), '')}
    response = client.post('/preview_file', data=data)
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_invalid_file_type(client, tmp_path):
    """Test handling of invalid file types."""
    # Create a file with invalid extension
    invalid_file = tmp_path / "test.xyz"
    invalid_file.write_text("some content")

    with open(invalid_file, 'rb') as f:
        data = {'file': (f, 'test.xyz')}
        response = client.post('/classify_file', data=data)

    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_large_file_handling(client, tmp_path):
    """Test handling of large files."""
    # Create a large file (just over typical limit)
    large_file = tmp_path / "large.txt"
    large_file.write_bytes(b'x' * (10 * 1024 * 1024 + 1))  # 10MB + 1 byte

    with open(large_file, 'rb') as f:
        data = {'file': (f, 'large.txt')}
        response = client.post('/classify_file', data=data)

    assert response.status_code == 413  # Request Entity Too Large
    assert 'error' in response.get_json()