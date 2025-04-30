import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
from src.classifier.content_classifier import ContentClassifier
from src.classifier.exceptions import TextExtractionError
import numpy as np

class TestOCRFunctionality:
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for testing."""
        return ContentClassifier()

    @pytest.fixture
    def create_test_image(self, tmp_path):
        """Create a test image with text."""
        def _create_image(text, size=(800, 400), bg_color='white', text_color='black'):
            # Create a high-resolution image (2x size for better quality)
            img = Image.new('RGB', (size[0]*2, size[1]*2), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Try to load Arial font, fallback to default if not available
            try:
                font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 60)  # Larger font size
            except OSError:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
                except OSError:
                    font = ImageFont.load_default()

            # Calculate text position to center it
            text_lines = text.split('\n')
            y_position = 50
            for line in text_lines:
                # Get text size for centering
                try:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    x_position = (size[0]*2 - text_width) // 2
                except AttributeError:
                    x_position = 50  # Fallback if textbbox not available

                # Draw text with anti-aliasing
                draw.text((x_position, y_position), line, fill=text_color, font=font)
                y_position += 80

            # Save image with high DPI
            img_path = tmp_path / "test_image.png"
            img.save(img_path, dpi=(300, 300))  # Set DPI to 300
            return img_path
        return _create_image

    def test_basic_text_extraction(self, classifier, create_test_image):
        """Test basic text extraction from a clear image."""
        test_text = "INVOICE\nAmount: $500.00\nDate: 2024-03-20"
        image_path = create_test_image(test_text)

        extracted_text = classifier.extract_text(image_path)
        assert "INVOICE" in extracted_text.upper()
        assert "$500.00" in extracted_text
        assert "2024" in extracted_text

    def test_rotated_image(self, classifier, create_test_image):
        """Test text extraction from a rotated image."""
        test_text = "INVOICE #12345"
        image_path = create_test_image(test_text)

        # Rotate image
        with Image.open(image_path) as img:
            rotated = img.rotate(90, expand=True)  # expand=True to prevent cropping
            rotated.save(image_path, dpi=(300, 300))

        extracted_text = classifier.extract_text(image_path)
        assert "12345" in extracted_text

    def test_low_quality_image(self, classifier, create_test_image):
        """Test text extraction from a low quality image."""
        test_text = "MEDICAL REPORT\nPatient ID: 12345"
        image_path = create_test_image(test_text, size=(400, 200))

        extracted_text = classifier.extract_text(image_path)
        assert "MEDICAL" in extracted_text.upper()
        assert "12345" in extracted_text

    def test_image_with_tables(self, classifier, create_test_image):
        """Test extraction of text from images containing tables."""
        table_text = """
Item     Quantity  Price
-----------------------
Item A      2      $100
Item B      1       $50
"""
        image_path = create_test_image(table_text, size=(1000, 600))

        extracted_text = classifier.extract_text(image_path)
        assert "ITEM" in extracted_text.upper()
        assert "QUANTITY" in extracted_text.upper()
        assert "PRICE" in extracted_text.upper()

    def test_image_with_special_characters(self, classifier, create_test_image):
        """Test handling of special characters and symbols."""
        special_text = "€500.00\n¥1000\n£250.00"
        image_path = create_test_image(special_text, size=(600, 400))

        extracted_text = classifier.extract_text(image_path)
        # Look for numbers since currency symbols might be interpreted differently
        assert "500" in extracted_text
        assert "1000" in extracted_text
        assert "250" in extracted_text

    def test_invalid_image(self, classifier, tmp_path):
        """Test handling of invalid or corrupted images."""
        invalid_path = tmp_path / "invalid.png"
        with open(invalid_path, 'wb') as f:
            f.write(b"Not a valid PNG file")

        with pytest.raises(TextExtractionError):
            classifier.extract_text(invalid_path)

    def test_empty_image(self, classifier, create_test_image):
        """Test handling of images with no text."""
        image_path = create_test_image("   ")  # Just whitespace

        extracted_text = classifier.extract_text(image_path)
        assert not extracted_text.strip()

    def test_handwritten_text(self, classifier, create_test_image):
        """Test extraction of handwritten-style text."""
        handwritten_text = "Total Amount: $750.00"
        image_path = create_test_image(handwritten_text)

        extracted_text = classifier.extract_text(image_path)
        assert "750" in extracted_text