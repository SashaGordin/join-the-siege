import pytest
from pathlib import Path
import io
from PIL import Image
import pypdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import unicodedata

from src.classifier.content_classifier import ContentClassifier
from src.classifier.exceptions import (
    ClassificationError,
    TextExtractionError
)

@pytest.fixture
def classifier():
    """Create a classifier instance for testing."""
    return ContentClassifier()

@pytest.fixture
def empty_file(tmp_path):
    """Create an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.touch()
    return file_path

@pytest.fixture
def mixed_language_document(tmp_path):
    """Create a document with mixed language content."""
    content = """INVOICE - FACTURA - 請求書
    Invoice Number: INV-2024-001
    Date: 2024年3月15日

    Bill To:
    John Doe - ジョン・ドウ
    123 Main St - 本町123

    Items:
    1. Consulting Services - コンサルティング: €1000
    2. Software License - ソフトウェア: £500

    Total Amount: ¥150000
    """
    file_path = tmp_path / "mixed_language.txt"
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def multiple_currency_document(tmp_path):
    """Create a document with multiple currency symbols."""
    content = """INVOICE
    Invoice Number: INV-2024-002

    Items:
    1. EU Services: €1000,00
    2. UK License: £500.00
    3. Japan Office: ¥50000
    4. US Consulting: $750.00
    5. Bitcoin Payment: ₿0.025

    Total Amount: $2,500.00 (USD equivalent)
    """
    file_path = tmp_path / "multiple_currency.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def corrupted_text_document(tmp_path):
    """Create a document with corrupted/invalid text encoding."""
    # Create some invalid UTF-8 sequences
    content = b"INVOICE\xFF\xFE\n" + b"Amount: $500\x80\x90\n"
    file_path = tmp_path / "corrupted.txt"
    file_path.write_bytes(content)
    return file_path

@pytest.fixture
def minimal_invoice(tmp_path):
    """Create a document with minimal valid invoice features."""
    content = """Invoice #12345
    Amount: $500
    """
    file_path = tmp_path / "minimal_invoice.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def password_protected_pdf(tmp_path):
    """Create a password-protected PDF."""
    pdf_path = tmp_path / "protected.pdf"

    # Create PDF with content first
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(1*inch, 10*inch, "INVOICE")
    c.drawString(1*inch, 9*inch, "Amount: $500")
    c.save()

    # Now encrypt it
    reader = pypdf.PdfReader(pdf_path)
    writer = pypdf.PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("password123")

    with open(pdf_path, "wb") as f:
        writer.write(f)

    return pdf_path

@pytest.fixture
def ambiguous_document(tmp_path):
    """Create a document with multiple type indicators."""
    content = """INVOICE / BANK STATEMENT

    Statement Period: March 1-31, 2024
    Account: 1234567890

    Invoice Number: INV-2024-003

    Opening Balance: $1000

    Items:
    1. Service Fee: $500

    Closing Balance: $500
    """
    file_path = tmp_path / "ambiguous.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def alternative_dates_document(tmp_path):
    """Create a document with various date formats."""
    content = """INVOICE

    Standard Date: March 15, 2024
    ISO Date: 2024-03-15
    Short Date: 3/15/24
    European Date: 15.03.2024
    Julian Date: 2024-074
    Unix Timestamp: 1710428400

    Amount: $500
    """
    file_path = tmp_path / "dates.txt"
    file_path.write_text(content)
    return file_path

def test_empty_document(classifier, empty_file):
    """Test classification of empty document."""
    with pytest.raises(ClassificationError, match="Empty file"):
        classifier.classify_file(empty_file)

def test_mixed_language_document(classifier, mixed_language_document):
    """Test classification with mixed language content."""
    result = classifier.classify_file(mixed_language_document)

    assert result["class"] == "invoice"
    assert result["confidence"] > 0.8
    # Verify amount detection works with Japanese text
    amounts = next(f for f in result["features"] if f["type"] == "amount")
    assert any("1000" in v for v in amounts["values"])
    assert any("500" in v for v in amounts["values"])
    assert any("150000" in v for v in amounts["values"])

def test_multiple_currency_symbols(classifier, multiple_currency_document):
    """Test amount detection with various currency symbols."""
    result = classifier.classify_file(multiple_currency_document)

    assert result["class"] == "invoice"
    amounts = next(f for f in result["features"] if f["type"] == "amount")

    # Check detection of different currencies
    assert any("€1000,00" in v for v in amounts["values"])
    assert any("£500.00" in v for v in amounts["values"])
    assert any("¥50000" in v for v in amounts["values"])
    assert any("$750.00" in v for v in amounts["values"])

def test_corrupted_text_handling(classifier, corrupted_text_document):
    """Test handling of documents with encoding issues."""
    with pytest.raises(TextExtractionError, match="encoding"):
        classifier.extract_text(corrupted_text_document)

def test_minimal_valid_document(classifier, minimal_invoice):
    """Test classification with minimal required features."""
    result = classifier.classify_file(minimal_invoice)

    assert result["class"] == "invoice"
    # Should have lower confidence due to minimal features
    assert 0.5 < result["confidence"] < 0.8

    # Check that basic features are detected
    features = result["features"]
    assert any(f["type"] == "invoice_number" for f in features)
    assert any(f["type"] == "amount" for f in features)

def test_password_protected_pdf(classifier, password_protected_pdf):
    """Test handling of password-protected PDFs."""
    with pytest.raises(TextExtractionError, match="encrypted"):
        classifier.extract_text(password_protected_pdf)

def test_ambiguous_document(classifier, ambiguous_document):
    """Test document with multiple type indicators."""
    result = classifier.classify_file(ambiguous_document)

    # Should either be classified as unknown or have very low confidence
    if result["class"] != "unknown":
        assert result["confidence"] < 0.6

    # Should detect features from both types
    features = result["features"]
    assert any(f["type"] == "invoice_number" for f in features)
    assert any("opening balance" in str(f).lower() for f in features)

def test_alternative_date_formats(classifier, alternative_dates_document):
    """Test recognition of various date formats."""
    result = classifier.classify_file(alternative_dates_document)

    dates = next(f for f in result["features"] if f["type"] == "date")
    assert dates["present"]
    assert len(dates["values"]) >= 2  # Should detect at least standard and ISO formats

def test_large_document_performance(classifier, tmp_path):
    """Test classification performance with large documents."""
    # Create a large invoice with repeated content
    base_content = "INVOICE\nInvoice Number: INV-2024-001\nAmount: $500\n"
    large_content = base_content * 1000  # Create ~50KB file

    file_path = tmp_path / "large.txt"
    file_path.write_text(large_content)

    import time
    start_time = time.time()

    result = classifier.classify_file(file_path)

    processing_time = time.time() - start_time

    assert result["class"] == "invoice"
    assert processing_time < 5.0  # Should process in under 5 seconds

def test_multiple_classifications(classifier, minimal_invoice, ambiguous_document):
    """Test multiple classifications in sequence."""
    import time
    start_time = time.time()

    # Classify same documents multiple times
    for _ in range(10):
        classifier.classify_file(minimal_invoice)
        classifier.classify_file(ambiguous_document)

    processing_time = time.time() - start_time

    # 20 classifications should take less than 10 seconds
    assert processing_time < 10.0