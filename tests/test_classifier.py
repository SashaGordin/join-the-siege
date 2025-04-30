import pytest
from pathlib import Path
import json
from PIL import Image
import io
import pypdf  # Updated from PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import time

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

    @pytest.fixture
    def sample_complex_document(self):
        """Sample text content for a complex document."""
        return """MEDICAL BILLING STATEMENT

        Patient ID: PAT-2024-789
        Insurance Policy: INS-456-789
        Date of Service: March 15, 2024

        Provider: Dr. Jane Smith
        License: MD-12345

        Services:
        1. Initial Consultation (CPT: 99213) - $150
        2. Blood Test Panel (CPT: 80053) - $75
        3. Vaccination (CPT: 90715) - $95

        Insurance Adjustment: -$125
        Patient Responsibility: $195

        Payment Due: April 15, 2024

        Please reference claim number CLM-2024-456 for all inquiries.
        """

    @pytest.fixture
    def sample_industry_documents(self):
        """Sample documents from different industries."""
        return {
            "healthcare": """
            MEDICAL CLAIM FORM
            Patient: John Smith
            Insurance ID: INS-123456
            Provider: Dr. Jane Doe
            CPT Code: 99213 - Office Visit
            Diagnosis: ICD-10 J45.901
            Amount Billed: $150.00
            Date of Service: 03/15/2024
            """,
            "legal": """
            LEGAL CONTRACT
            Agreement Date: March 15, 2024
            Parties:
            1. ABC Corporation ("Client")
            2. XYZ Law Firm ("Provider")

            Terms:
            1. Legal services to be provided...
            2. Fee structure: $300/hour

            Contract ID: CNT-2024-789
            """,
            "real_estate": """
            PROPERTY APPRAISAL REPORT
            Property Address: 123 Main St
            Appraisal Date: March 15, 2024
            Appraiser: Jane Smith
            License: APP-123-NY

            Market Value: $500,000
            Square Footage: 2,000
            Report ID: APR-2024-456
            """,
            "education": """
            STUDENT TRANSCRIPT
            Student: Alice Johnson
            Student ID: STU-2024-123
            Program: Computer Science

            Courses:
            CS101 - A
            CS102 - B+

            GPA: 3.75
            Semester: Spring 2024
            """
        }

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

        print("\n=== Confidence Scoring Test Results ===")
        print(f"Clear Invoice:")
        print(f"  Type: {clear_result['class']}")
        print(f"  Confidence: {clear_result['confidence']}")
        print(f"  Features: {[f['type'] for f in clear_result['features'] if f['present']]}")

        print(f"\nPartial Invoice:")
        print(f"  Type: {partial_result['class']}")
        print(f"  Confidence: {partial_result['confidence']}")
        print(f"  Features: {[f['type'] for f in partial_result['features'] if f['present']]}")

        print(f"\nAmbiguous Document:")
        print(f"  Type: {ambiguous_result['class']}")
        print(f"  Confidence: {ambiguous_result['confidence']}")
        print(f"  Features: {[f['type'] for f in ambiguous_result['features'] if f['present']]}")

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

    def test_show_llm_response(self, tmp_path, sample_complex_document, classifier):
        """Test to demonstrate LLM classification response."""
        # Create a text file with complex content
        doc_file = tmp_path / "complex_doc.txt"
        doc_file.write_text(sample_complex_document)

        result = classifier.classify_file(doc_file)

        print("\n=== LLM Classification Result ===")
        print(f"Document Type: {result['class']}")
        print(f"Confidence: {result['confidence']}")
        print("\nFeatures:")
        for feature in result['features']:
            print(f"\n{feature['type'].upper()}:")
            print(f"Present: {feature['present']}")
            if feature['present']:
                print(f"Values: {', '.join(feature['values'])}")
            if 'category' in feature:
                print(f"Category: {feature['category']}")

        # Basic assertions to ensure response structure
        assert isinstance(result['class'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['features'], list)

        # This should be a medical billing statement
        assert 'medical' in result['class'].lower()
        assert result['confidence'] > 0.8  # Should be high confidence

        # Should find specific medical billing features
        features = result['features']
        assert any('patient' in str(v).lower() for f in features for v in f.get('values', []))
        assert any('cpt' in str(v).lower() for f in features for v in f.get('values', []))

    def test_classify_with_flexible_types(self, tmp_path, classifier):
        """Test classification with various document types."""
        # Test cases with different document types
        test_cases = [
            {
                "content": """
                PURCHASE ORDER
                PO Number: PO-2024-123
                Date: March 15, 2024
                Vendor: ABC Corp
                Items:
                1. Widget A - $100
                2. Widget B - $200
                Total: $300
                """,
                "expected_type": "purchase order",
            },
            {
                "content": """
                MEDICAL PRESCRIPTION
                Patient: John Doe
                Date: March 15, 2024
                Rx: Amoxicillin 500mg
                Sig: Take 1 capsule 3 times daily
                Quantity: 30
                Refills: 2
                Dr. Smith, MD
                License: 12345
                """,
                "expected_type": "prescription",
            },
            {
                "content": """
                TAX RETURN 2023
                Form 1040
                Taxpayer: Jane Doe
                SSN: XXX-XX-1234
                Income: $75,000
                Deductions: $12,000
                Tax Due: $15,000
                """,
                "expected_type": "tax return",
            }
        ]

        print("\n=== Flexible Types Test Results ===")
        for i, case in enumerate(test_cases):
            # Create and classify test file
            test_file = tmp_path / f"test_doc_{i}.txt"
            test_file.write_text(case["content"])

            result = classifier.classify_file(test_file)

            print(f"\nDocument {i+1} - {case['expected_type']}:")
            print(f"  Raw type from LLM: {result['class']}")
            print(f"  Expected type: {case['expected_type']}")
            print(f"  Features: {[f['type'] for f in result['features'] if f['present']]}")
            print(f"  Confidence: {result['confidence']}")

            # Normalize both strings for comparison (convert spaces to underscores)
            expected_normalized = '_'.join(case['expected_type'].lower().split())
            result_normalized = '_'.join(result['class'].lower().split())

            # Verify classification is reasonable
            assert expected_normalized == result_normalized, \
                f"Expected '{case['expected_type']}' but got '{result['class']}'"
            assert result["confidence"] > 0.7  # Should be reasonably confident
            assert len(result["features"]) > 0  # Should find some features

    def test_cross_industry_classification(self, tmp_path, classifier, sample_industry_documents):
        """Test classification across different industries."""
        print("\n=== Cross-Industry Classification Test ===")

        for industry, content in sample_industry_documents.items():
            # Create test file
            test_file = tmp_path / f"test_{industry}.txt"
            test_file.write_text(content)

            result = classifier.classify_file(test_file)

            print(f"\n{industry.upper()} Document:")
            print(f"  Type: {result['class']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Features: {[f['type'] for f in result['features'] if f['present']]}")

            # Basic assertions
            assert result["confidence"] > 0.6  # Should be reasonably confident
            assert len(result["features"]) > 0
            assert any(f["type"] == "document_type" for f in result["features"])

            # Industry-specific assertions
            if industry == "healthcare":
                assert any(f["type"] == "cpt_code" for f in result["features"])
                assert "medical" in result["class"].lower() or "claim" in result["class"].lower()
            elif industry == "legal":
                assert "contract" in result["class"].lower()
                assert any("contract" in str(f).lower() for f in result["features"])
            elif industry == "real_estate":
                assert "appraisal" in result["class"].lower()
                assert any(f["type"] == "amount" for f in result["features"])

    def test_poorly_named_files(self, tmp_path, classifier):
        """Test that classification works regardless of filename."""
        print("\n=== Poorly Named Files Test ===")

        test_cases = [
            {
                "filename": "document1.txt",  # Generic name
                "content": """INVOICE #12345
                            Date: March 15, 2024
                            Amount: $500.00""",
                "expected_type": "invoice"
            },
            {
                "filename": "scan.pdf",  # Misleading name
                "content": """MEDICAL PRESCRIPTION
                            Patient: John Doe
                            Rx: Amoxicillin""",
                "expected_type": "prescription"
            },
            {
                "filename": "file_123.doc",  # Random name
                "content": """TAX RETURN 2023
                            Income: $75,000
                            Deductions: $12,000""",
                "expected_type": "tax_return"
            }
        ]

        for case in test_cases:
            test_file = tmp_path / case["filename"]
            test_file.write_text(case["content"])

            result = classifier.classify_file(test_file)

            print(f"\nFile: {case['filename']}")
            print(f"  Expected Type: {case['expected_type']}")
            print(f"  Actual Type: {result['class']}")
            print(f"  Confidence: {result['confidence']}")

            assert case["expected_type"] in result["class"].lower()
            assert result["confidence"] > 0.6

    def test_large_document_handling(self, tmp_path, classifier):
        """Test handling of large documents."""
        print("\n=== Large Document Test ===")

        # Create a large invoice with many line items
        large_content = ["INVOICE #12345\nDate: March 15, 2024\n"]
        for i in range(1, 101):  # 100 line items
            large_content.append(f"Item {i}: Product {i} - ${i*10}.00")
        large_content.append("\nTotal Amount: $50,500.00")

        large_file = tmp_path / "large_invoice.txt"
        large_file.write_text("\n".join(large_content))

        start_time = time.time()
        result = classifier.classify_file(large_file)
        processing_time = time.time() - start_time

        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Document Type: {result['class']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Number of Features: {len([f for f in result['features'] if f['present']])}")

        assert result["class"] == "invoice"
        assert result["confidence"] > 0.8
        assert processing_time < 5.0  # Should process in under 5 seconds
        assert any(f["type"] == "amount" and len(f["values"]) > 50 for f in result["features"])

    def test_robustness_and_edge_cases(self, tmp_path, classifier):
        """Test robustness with various edge cases."""
        print("\n=== Robustness and Edge Cases Test ===")

        edge_cases = [
            {
                "name": "mixed_languages.txt",
                "content": """INVOICE - FACTURA
                            Date/Fecha: 15/03/2024
                            Amount/Monto: $500.00
                            Thank you/Gracias""",
                "expected_type": "invoice"
            },
            {
                "name": "multiple_currencies.txt",
                "content": """INVOICE #12345
                            Amount: $500.00
                            Amount: €450.00
                            Amount: £400.00""",
                "expected_type": "invoice"
            },
            {
                "name": "special_chars.txt",
                "content": """INVOICE #️⃣ 12345
                            Date: 📅 15/03/2024
                            Amount: 💰 $500.00""",
                "expected_type": "invoice"
            }
        ]

        for case in edge_cases:
            test_file = tmp_path / case["name"]
            test_file.write_text(case["content"])

            result = classifier.classify_file(test_file)

            print(f"\nCase: {case['name']}")
            print(f"  Expected Type: {case['expected_type']}")
            print(f"  Actual Type: {result['class']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Features: {[f['type'] for f in result['features'] if f['present']]}")

            assert case["expected_type"] in result["class"].lower()
            assert result["confidence"] > 0.6
            assert len(result["features"]) > 0

    def test_industry_specific_features(self, tmp_path, classifier):
        """Test extraction of industry-specific features."""
        print("\n=== Industry-Specific Features Test ===")

        # Medical document with CPT codes
        medical_content = """
        MEDICAL CLAIM
        Patient: John Doe
        Date: March 15, 2024

        Services:
        1. Office Visit (CPT: 99213) - $150
        2. Lab Work (CPT: 80053) - $100
        3. X-Ray (CPT: 71045) - $200

        Diagnosis: ICD-10 J45.901
        Provider: Dr. Smith (NPI: 1234567890)
        """

        medical_file = tmp_path / "medical_claim.txt"
        medical_file.write_text(medical_content)

        result = classifier.classify_file(medical_file)

        print("\nMedical Document Features:")
        for feature in result["features"]:
            if feature["present"]:
                print(f"  {feature['type']}: {feature['values']}")

        # Check for medical-specific features
        assert any(f["type"] == "cpt_code" and len(f["values"]) >= 3 for f in result["features"])
        assert any("diagnosis" in str(f).lower() for f in result["features"])
        assert any("provider" in str(f).lower() for f in result["features"])
        assert result["confidence"] > 0.8

        # Legal document with specific clauses
        legal_content = """
        LEGAL CONTRACT
        Contract ID: LC-2024-001
        Date: March 15, 2024

        WHEREAS, the parties agree as follows:

        1. DEFINITIONS
        2. TERM AND TERMINATION
        3. CONFIDENTIALITY
        4. GOVERNING LAW

        IN WITNESS WHEREOF, the parties have executed this Agreement.
        """

        legal_file = tmp_path / "legal_contract.txt"
        legal_file.write_text(legal_content)

        result = classifier.classify_file(legal_file)

        print("\nLegal Document Features:")
        for feature in result["features"]:
            if feature["present"]:
                print(f"  {feature['type']}: {feature['values']}")

        # Check for legal-specific features
        assert "contract" in result["class"].lower()
        assert any("clause" in str(f).lower() or "section" in str(f).lower() for f in result["features"])
        assert result["confidence"] > 0.8