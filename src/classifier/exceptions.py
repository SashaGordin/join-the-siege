class ClassifierError(Exception):
    """Base exception for classifier errors."""
    pass

class ClassificationError(ClassifierError):
    """Raised when classification fails."""
    pass

class TextExtractionError(ClassifierError):
    """Raised when text extraction fails."""
    pass

class FeatureExtractionError(ClassifierError):
    """Raised when feature extraction fails."""
    pass