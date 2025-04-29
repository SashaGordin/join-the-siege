class FileValidationError(Exception):
    """Base exception for file validation errors."""
    pass

class InvalidFileTypeError(FileValidationError):
    """Raised when file type is not supported or doesn't match its extension."""
    pass

class FileTooLargeError(FileValidationError):
    """Raised when file size exceeds the maximum allowed size."""
    pass

class FileCorruptError(FileValidationError):
    """Raised when file content is corrupted or unreadable."""
    pass

class FileAccessError(FileValidationError):
    """Raised when file cannot be accessed due to permissions or other IO issues."""
    pass