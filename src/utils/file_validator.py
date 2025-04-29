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
        'image/png': '.png',
        'text/plain': '.txt',
        # Common variations of text MIME types
        'text/x-python': '.txt',  # Python files are text
        'text/x-java': '.txt',    # Java files are text
        'text/x-c': '.txt',       # C files are text
        'text/x-cpp': '.txt',     # C++ files are text
        'text/x-script': '.txt',  # Script files are text
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
            raise FileTooLargeError("File exceeds maximum allowed size")

        # Get actual MIME type from file content
        actual_mime_type = self.get_file_type(file_path)
        actual_ext = file_path.suffix.lower()

        # For text files, normalize the MIME type and check extension
        if actual_mime_type.startswith('text/'):
            if actual_ext in ['.txt', '.text', '.log']:
                return True
            # If it's a text file with wrong extension, treat it as unsupported
            raise InvalidFileTypeError(f"Unsupported file type: text file with invalid extension {actual_ext}")

        # For images, allow them through validation (they'll be handled in classification)
        if actual_mime_type in ['image/jpeg', 'image/png']:
            if actual_ext in ['.jpg', '.jpeg', '.png']:
                return True
            raise InvalidFileTypeError(f"File extension does not match content type")

        # For PDFs, check both MIME type and extension
        if actual_mime_type == 'application/pdf':
            if actual_ext == '.pdf':
                return True
            raise InvalidFileTypeError(f"File extension does not match content type")

        # If we get here, the file type is not supported
        raise InvalidFileTypeError(f"Unsupported file type: {actual_mime_type}")

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
        mime_type = self.get_file_type(file_path)

        # Initialize preview data
        preview_data = {
            'mime_type': mime_type,
            'preview_available': False
        }

        try:
            # Validate file type (but allow images through)
            if not mime_type.startswith('image/'):
                self.is_valid_file_type(file_path)

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
                    preview_data['preview_available'] = True

            elif mime_type == 'application/pdf':
                # For now, just return the first page info
                preview_data['preview_available'] = False

            elif mime_type.startswith('text/'):
                # For text files, return first few lines
                try:
                    with open(file_path, 'r') as f:
                        lines = [next(f) for _ in range(10)]
                        preview_data['text_preview'] = ''.join(lines)
                        preview_data['preview_available'] = True
                except (StopIteration, UnicodeDecodeError):
                    # File has fewer lines or is not readable as text
                    preview_data['preview_available'] = False

            return preview_data

        except Exception as e:
            raise FileCorruptError(f"Could not generate preview: {e}")