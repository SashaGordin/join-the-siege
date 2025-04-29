import pytest
from pathlib import Path
import magic

from src.utils.file_validator import FileValidator, InvalidFileTypeError

class TestFileHandling:
    """
    Test suite for file handling functionality.
    Tests file type detection, validation, and basic content checking.
    """

    def test_valid_pdf_detection(self, tmp_path):
        """
        Test that a valid PDF file is correctly identified.
        Uses a temporary file to ensure test isolation.
        """
        # Create a simple PDF file for testing
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.5\n%\x93\x8C\x8B\x9E")  # Basic PDF header

        validator = FileValidator()
        assert validator.is_valid_file_type(test_file) == True
        assert validator.get_file_type(test_file) == "application/pdf"

    def test_invalid_file_type(self, tmp_path):
        """
        Test that files with unsupported types are rejected.
        """
        # Create a text file with .pdf extension
        test_file = tmp_path / "fake.pdf"
        test_file.write_text("This is not a PDF file")

        validator = FileValidator()
        with pytest.raises(InvalidFileTypeError):
            validator.is_valid_file_type(test_file)

    def test_empty_file(self, tmp_path):
        """
        Test handling of empty files.
        Empty files should be rejected.
        """
        test_file = tmp_path / "empty.pdf"
        test_file.touch()  # Creates an empty file

        validator = FileValidator()
        with pytest.raises(InvalidFileTypeError, match="Empty file"):
            validator.is_valid_file_type(test_file)

    def test_file_size_limit(self, tmp_path):
        """
        Test that files exceeding size limit are rejected.
        """
        test_file = tmp_path / "large.pdf"
        # Create a file that exceeds our limit (let's say 10MB)
        test_file.write_bytes(b"%PDF-1.5" + b"0" * (11 * 1024 * 1024))  # 11MB file

        validator = FileValidator()
        with pytest.raises(InvalidFileTypeError, match="File too large"):
            validator.is_valid_file_type(test_file)