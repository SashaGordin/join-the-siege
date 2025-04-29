import magic
from pathlib import Path
from typing import Union, Set

from .exceptions import (
    InvalidFileTypeError,
    FileTooLargeError,
    FileCorruptError,
    FileAccessError,
)

class FileValidator:
    """
    Validates files for type, size, and content.
    """

    # Maximum file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Allowed MIME types and their corresponding extensions
    ALLOWED_MIME_TYPES = {
        'application/pdf': '.pdf',
        'image/jpeg': '.jpg',
        'image/png': '.png'
    }

    def __init__(self):
        """Initialize the file validator."""
        self.magic = magic.Magic(mime=True)

    def get_file_type(self, file_path: Union[str, Path]) -> str:
        """
        Get the MIME type of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: MIME type of the file

        Raises:
            FileAccessError: If file cannot be accessed
        """
        file_path = Path(file_path)
        try:
            return self.magic.from_file(str(file_path))
        except Exception as e:
            raise FileAccessError(f"Cannot access file: {e}")

    def is_valid_file_type(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the file is of a valid type and size.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file is valid

        Raises:
            InvalidFileTypeError: If file type is not supported
            FileTooLargeError: If file is too large
            FileAccessError: If file cannot be accessed
        """
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            raise FileAccessError("File does not exist")

        # Check if file is empty
        if file_path.stat().st_size == 0:
            raise InvalidFileTypeError("Empty file")

        # Check file size
        if file_path.stat().st_size > self.MAX_FILE_SIZE:
            raise InvalidFileTypeError("File too large")

        # Get and validate MIME type
        mime_type = self.get_file_type(file_path)
        if mime_type not in self.ALLOWED_MIME_TYPES:
            raise InvalidFileTypeError(f"Unsupported file type: {mime_type}")

        # Check if extension matches MIME type
        expected_ext = self.ALLOWED_MIME_TYPES[mime_type]
        if file_path.suffix.lower() != expected_ext:
            raise InvalidFileTypeError(
                f"File extension does not match content type. "
                f"Expected {expected_ext} for {mime_type}"
            )

        return True