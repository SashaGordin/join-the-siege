import magic
from pathlib import Path
from typing import Union, Dict, Any
import io
from PIL import Image

from .exceptions import (
    InvalidFileTypeError,
    FileTooLargeError,
    FileCorruptError,
    FileAccessError,
)

class FileValidator:
    """
    Validates files for type, size, and content.
    Provides preview functionality for supported file types.
    """

    # Maximum file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Maximum preview size for images
    MAX_PREVIEW_SIZE = (300, 300)

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

        # Get actual MIME type from file content
        actual_mime_type = self.get_file_type(file_path)

        # Check if the MIME type is supported
        if actual_mime_type not in self.ALLOWED_MIME_TYPES:
            raise InvalidFileTypeError(f"Unsupported file type: {actual_mime_type}")

        # Get the expected extension for this MIME type
        expected_ext = self.ALLOWED_MIME_TYPES[actual_mime_type]
        # Get the actual file extension (converted to lowercase)
        actual_ext = file_path.suffix.lower()

        # Check if extension matches the actual content type
        if actual_ext != expected_ext:
            raise InvalidFileTypeError(
                f"File extension does not match content type. "
                f"Got {actual_ext} but expected {expected_ext} for {actual_mime_type}"
            )

        return True

    def get_file_preview(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get a preview of the file contents.

        Args:
            file_path: Path to the file

        Returns:
            dict: Preview data including mime_type and preview-specific data

        Raises:
            FileAccessError: If file cannot be accessed
            InvalidFileTypeError: If file type is not supported
        """
        file_path = Path(file_path)

        # Validate file first
        self.is_valid_file_type(file_path)

        mime_type = self.get_file_type(file_path)
        preview_data = {'mime_type': mime_type}

        try:
            if mime_type.startswith('image/'):
                # Handle image preview
                with Image.open(file_path) as img:
                    # Create thumbnail
                    img.thumbnail(self.MAX_PREVIEW_SIZE)
                    # Save thumbnail to bytes
                    thumb_io = io.BytesIO()
                    img.save(thumb_io, format=img.format)
                    preview_data['thumbnail'] = thumb_io.getvalue()
                    preview_data['dimensions'] = img.size

            elif mime_type == 'application/pdf':
                # For now, just return the first page info
                # TODO: Implement PDF preview
                preview_data['preview_available'] = False

            return preview_data

        except Exception as e:
            raise FileCorruptError(f"Could not generate preview: {e}")