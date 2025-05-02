from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ClassificationResult:
    """Result of document classification."""
    doc_type: str
    confidence: float
    features: List[Dict[str, Any]]