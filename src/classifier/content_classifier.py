from pathlib import Path
from typing import Dict, Any, Union, List
import pypdf
from PIL import Image
import magic
import re

from .exceptions import (
    ClassificationError,
    TextExtractionError,
    FeatureExtractionError
)

class ContentClassifier:
    """
    A content-based document classifier that uses actual document content
    rather than just filenames to determine document type.
    """

    def __init__(self):
        """Initialize the content classifier."""
        self.mime_magic = magic.Magic(mime=True)

    def extract_text(self, file_path: Union[str, Path]) -> str:
        """
        Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            str: Extracted text content

        Raises:
            TextExtractionError: If text extraction fails
            NotImplementedError: For unimplemented file types
        """
        file_path = Path(file_path)
        mime_type = self.mime_magic.from_file(str(file_path))

        try:
            if mime_type == 'application/pdf':
                return self._extract_text_from_pdf(file_path)
            elif mime_type.startswith('image/'):
                raise NotImplementedError("OCR not implemented yet")
            else:
                # For text files or unknown types, try reading as text
                return file_path.read_text()
        except NotImplementedError:
            raise  # Re-raise NotImplementedError as is
        except Exception as e:
            raise TextExtractionError(f"Failed to extract text: {str(e)}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        text = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise TextExtractionError(f"Failed to extract PDF text: {str(e)}")

    def classify_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Classify a file based on its content.

        Args:
            file_path: Path to the file

        Returns:
            dict: Classification result with class, confidence, and features

        Raises:
            ClassificationError: If classification fails
        """
        file_path = Path(file_path)

        # Check if file exists and is not empty
        if not file_path.exists():
            raise ClassificationError("File does not exist")
        if file_path.stat().st_size == 0:
            raise ClassificationError("Empty file")

        try:
            # Extract text content
            text = self.extract_text(file_path)

            # Extract features from text
            features = self._extract_features(text)

            # Classify based on features
            classification = self._classify_features(features)

            return {
                "class": classification["class"],
                "confidence": classification["confidence"],
                "features": features
            }

        except TextExtractionError as e:
            raise ClassificationError(f"Text extraction failed: {str(e)}")
        except Exception as e:
            raise ClassificationError(f"Classification failed: {str(e)}")

    def _extract_features(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relevant features from text content.

        Args:
            text: The text content to analyze

        Returns:
            list: List of extracted features
        """
        features = []
        text_lower = text.lower()

        # Extract amounts using regex
        amount_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, text)
        if amounts:
            features.append({
                "type": "amount",
                "present": True,
                "count": len(amounts),
                "values": amounts
            })

        # Extract dates using multiple patterns
        date_patterns = [
            # Month name formats
            r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4}',
            # Numeric dates
            r'\d{2}[/-]\d{2}[/-]\d{4}',
            # Date ranges
            r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:\s*-\s*\d{1,2})?,?\s+\d{4}'
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            dates.extend(matches)

        if dates:
            features.append({
                "type": "date",
                "present": True,
                "count": len(dates),
                "values": dates
            })

        # Extract invoice numbers with more patterns
        invoice_patterns = [
            r'invoice\s*#?\s*\d+',
            r'inv[- ]?\d+',
            r'invoice number:?\s*\w+[-]?\d+'
        ]
        for pattern in invoice_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                features.append({
                    "type": "invoice_number",
                    "present": True,
                    "value": matches[0]
                })
                break

        # Document type indicators with more context
        doc_type = None
        bank_statement_indicators = 0

        # Check for bank statement specific features
        if 'opening balance' in text_lower:
            bank_statement_indicators += 1
        if 'closing balance' in text_lower:
            bank_statement_indicators += 1
        if 'account number' in text_lower:
            bank_statement_indicators += 1
        if 'statement period' in text_lower:
            bank_statement_indicators += 1

        if bank_statement_indicators >= 2:  # If we have at least 2 strong indicators
            doc_type = "bank_statement"
        elif 'bank statement' in text_lower or 'account statement' in text_lower:
            doc_type = "bank_statement"
        elif 'invoice' in text_lower:
            doc_type = "invoice"
        elif any(term in text_lower for term in ['drivers license', "driver's license"]):
            doc_type = "drivers_license"

        if doc_type:
            features.append({
                "type": "document_type",
                "value": doc_type,
                "strength": bank_statement_indicators if doc_type == "bank_statement" else 1
            })

        return features

    def _classify_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify document based on extracted features.

        Args:
            features: List of extracted features

        Returns:
            dict: Classification result with class and confidence
        """
        feature_types = {f["type"] for f in features}
        doc_type_feature = next((f for f in features if f["type"] == "document_type"), None)
        doc_type = doc_type_feature["value"] if doc_type_feature else None

        # Calculate base confidence from feature quality
        base_confidence = 0.0

        # Add confidence for each feature type
        if "amount" in feature_types:
            amount_feature = next(f for f in features if f["type"] == "amount")
            base_confidence += min(0.3, amount_feature["count"] * 0.1)  # More amounts = higher confidence

        if "date" in feature_types:
            date_feature = next(f for f in features if f["type"] == "date")
            base_confidence += min(0.2, date_feature["count"] * 0.1)  # More dates = higher confidence

        if "invoice_number" in feature_types:
            base_confidence += 0.3  # Strong indicator

        if doc_type:
            # For bank statements, add more confidence based on number of specific indicators
            if doc_type == "bank_statement" and doc_type_feature.get("strength", 0) > 1:
                base_confidence += min(0.4, 0.1 * doc_type_feature["strength"])  # Up to 0.4 for strong indicators
            else:
                base_confidence += 0.2  # Standard confidence for document type

        # Classify based on feature combinations
        if doc_type == "invoice" and "invoice_number" in feature_types and "amount" in feature_types:
            confidence = min(0.95, base_confidence + 0.3)
            return {"class": "invoice", "confidence": confidence}

        if doc_type == "bank_statement" and "amount" in feature_types and "date" in feature_types:
            # Add extra confidence for bank statements with strong indicators
            extra_confidence = 0.3 if doc_type_feature.get("strength", 0) > 1 else 0.2
            confidence = min(0.95, base_confidence + extra_confidence)
            return {"class": "bank_statement", "confidence": confidence}

        if "amount" in feature_types and "date" in feature_types:
            # Might be a financial document
            confidence = min(0.7, base_confidence)
            if doc_type:
                return {"class": doc_type, "confidence": confidence}
            return {"class": "unknown", "confidence": confidence}

        # Default case - low confidence
        return {"class": "unknown", "confidence": max(0.1, base_confidence)}