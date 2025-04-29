import pytest
from pathlib import Path
import json
from PIL import Image
import io
import pypdf  # Updated from PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.classifier.content_classifier import ContentClassifier
from src.classifier.exceptions import (
    ClassificationError,
    TextExtractionError
)

class TestContentClassifier:
    """
    Test suite for content-based document classification.
    Tests text extraction, classification, and confidence scoring.
    """

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for testing."""
        return ContentClassifier()

    @pytest.fixture
    def sample_invoice_text(self):
        """Sample text content for an invoice."""
        return """INVOICE

        Invoice Number: INV-2024-001
        Date: March 15, 2024

        Bill To:
        John Doe
        123 Main St

        Items:
        1. Consulting Services - $1000
        2. Software License - $500

        Total Amount: $1500
        """

    @pytest.fixture
    def sample_bank_statement_text(self):
        """Sample text content for a bank statement."""
        return """BANK STATEMENT

        Account Number: 1234567890
        Statement Period: March 1-31, 2024

        Opening Balance: $5000

        Transactions:
        03/15 Deposit $1000
        03/16 Withdrawal $200

        Closing Balance: $5800
        """

    def test_text_extraction_from_pdf(self, tmp_path, sample_invoice_text):
        """Test extracting text from a PDF file."""
        pdf_path = tmp_path / "test_invoice.pdf"

        # Create PDF using reportlab
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        # Add text to the PDF
        y_position = 750  # Start from top
        for line in sample_invoice_text.split('\n'):
            c.drawString(72, y_position, line.strip())
            y_position -= 15  # Move down for next line
        c.save()

        classifier = ContentClassifier()
        extracted_text = classifier.extract_text(pdf_path)

        # Check for key phrases rather than exact matches due to PDF formatting
        assert "INVOICE" in extracted_text
        assert "INV-2024-001" in extracted_text
        assert "$1500" in extracted_text

    def test_text_extraction_from_image(self, tmp_path):
        """Test extracting text from an image using OCR."""
        # Create an image with text
        img = Image.new('RGB', (800, 400), color='white')
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        classifier = ContentClassifier()
        with pytest.raises(NotImplementedError, match="OCR not implemented yet"):
            classifier.extract_text(img_path)

    def test_classify_invoice(self, tmp_path, sample_invoice_text, classifier):
        """Test classification of an invoice document."""
        # Create a text file with invoice content
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(sample_invoice_text)

        result = classifier.classify_file(invoice_file)

        assert result["class"] == "invoice"
        assert result["confidence"] > 0.8  # High confidence for clear invoice
        assert "features" in result

        # Check specific features
        features = result["features"]
        assert any(f["type"] == "document_type" and f["value"] == "invoice" for f in features)
        assert any(f["type"] == "amount" for f in features)
        assert any(f["type"] == "invoice_number" for f in features)

    def test_classify_bank_statement(self, tmp_path, sample_bank_statement_text, classifier):
        """Test classification of a bank statement."""
        # Create a text file with bank statement content
        statement_file = tmp_path / "statement.txt"
        statement_file.write_text(sample_bank_statement_text)

        result = classifier.classify_file(statement_file)

        assert result["class"] == "bank_statement"
        assert result["confidence"] > 0.8
        assert "features" in result

        # Check specific features
        features = result["features"]
        assert any(f["type"] == "document_type" and f["value"] == "bank_statement" for f in features)
        assert any(f["type"] == "amount" for f in features)
        assert any(f["type"] == "date" for f in features)

    def test_classify_unknown_document(self, tmp_path, classifier):
        """Test classification of a document that doesn't match known types."""
        # Create a text file with random content
        random_file = tmp_path / "random.txt"
        random_file.write_text("This is some random text that doesn't match any category.")

        result = classifier.classify_file(random_file)

        assert result["class"] == "unknown"
        assert result["confidence"] < 0.5  # Low confidence for unknown type
        assert "features" in result

    def test_classify_empty_file(self, tmp_path, classifier):
        """Test classification of an empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        with pytest.raises(ClassificationError, match="Empty file"):
            classifier.classify_file(empty_file)

    def test_confidence_scoring(self, tmp_path, sample_invoice_text, classifier):
        """Test that confidence scores are reasonable."""
        # Create variations of invoice with different confidence levels

        # Clear invoice (high confidence)
        clear_invoice = tmp_path / "clear_invoice.txt"
        clear_invoice.write_text(sample_invoice_text)

        # Partial invoice (medium confidence)
        partial_text = "Invoice #12345\nAmount: $500"
        partial_invoice = tmp_path / "partial_invoice.txt"
        partial_invoice.write_text(partial_text)

        # Ambiguous content (low confidence)
        ambiguous_text = "Payment received. Thank you."
        ambiguous_file = tmp_path / "ambiguous.txt"
        ambiguous_file.write_text(ambiguous_text)

        # Test confidence levels
        clear_result = classifier.classify_file(clear_invoice)
        partial_result = classifier.classify_file(partial_invoice)
        ambiguous_result = classifier.classify_file(ambiguous_file)

        assert clear_result["confidence"] > partial_result["confidence"]
        assert partial_result["confidence"] > ambiguous_result["confidence"]
        assert clear_result["confidence"] > 0.8  # High confidence for clear match
        assert ambiguous_result["confidence"] < 0.5  # Low confidence for ambiguous

    def test_feature_extraction(self, tmp_path, sample_invoice_text, classifier):
        """Test that relevant features are extracted from documents."""
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(sample_invoice_text)

        result = classifier.classify_file(invoice_file)

        assert "features" in result
        features = result["features"]

        # Check for expected invoice features
        assert any(feature["type"] == "amount" for feature in features)
        assert any(feature["type"] == "date" for feature in features)
        assert any(feature["type"] == "invoice_number" for feature in features)
        assert any(feature["type"] == "document_type" and feature["value"] == "invoice" for feature in features)

    def test_multiple_classifications(self, tmp_path, classifier):
        """Test that classifier maintains consistency across multiple calls."""
        # Create test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"invoice_{i}.txt"
            file_path.write_text(f"INVOICE #{i}\nAmount: ${i}00")
            files.append(file_path)

        # Classify all files
        results = [classifier.classify_file(f) for f in files]

        # Check consistency
        assert all(r["class"] == "invoice" for r in results)
        # Confidence should be similar for similar documents
        confidences = [r["confidence"] for r in results]
        assert max(confidences) - min(confidences) < 0.1  # Similar confidence