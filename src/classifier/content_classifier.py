from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import pypdf
from PIL import Image
import magic
from openai import OpenAI
import os
from dotenv import load_dotenv
from dataclasses import dataclass
import cv2
import numpy as np
import pytesseract
import httpx
import logging
from .exceptions import (
    ClassificationError,
    TextExtractionError,
    FeatureExtractionError
)
from .config.config_manager import IndustryConfigManager
import re
from src.classifier.pattern_learning.cached_pattern_matcher import CachedPatternMatcher
from src.classifier.pattern_learning.pattern_store import PatternStore

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ClassificationResult:
    doc_type: str
    confidence: float
    features: Dict[str, Any]

class ContentClassifier:
    """
    A content-based document classifier that uses LLM to determine document type.
    """

    def __init__(self, default_industry: str = None):
        """
        Initialize the content classifier.

        Args:
            default_industry: Default industry to use for classification
        """
        self.mime_magic = magic.Magic(mime=True)
        self.default_industry = default_industry
        self.config_manager = IndustryConfigManager()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Initialize OpenAI client with only the required parameters
        http_client = httpx.Client(
            follow_redirects=True,
            timeout=60.0
        )
        self.client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )

        # Check if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract is not installed. Please install tesseract-ocr package.")

        self.pattern_store = PatternStore()
        self.pattern_matcher = CachedPatternMatcher()

    def _get_industry_prompt(self, industry: str) -> str:
        """Get industry-specific prompt additions."""
        try:
            industry_config = self.config_manager.load_industry_config(industry)

            if industry == "healthcare":
                features = industry_config["features"]
                required = features["validation_rules"]["required_features"]
                specific = features["specific"]

                return f"""Industry Context: healthcare
Focus on medical document features and classify as medical documents when appropriate:
- Patient information ({specific["patient_id"]["description"]})
- Medical codes ({specific["medical_code"]["description"]})
- Provider details (NPI: {specific["provider_npi"]["description"]})
- Required features: {', '.join(required)}"""

            elif industry == "financial":
                features = industry_config["features"]
                required = features["validation_rules"]["required_features"]
                specific = features["specific"]

                return f"""Industry Context: financial
Focus on financial document features and classify as financial documents when appropriate:
- Invoice numbers ({specific["invoice_number"]["description"]})
- Payment terms ({specific["payment_terms"]["description"]})
- Account numbers ({specific["account_number"]["description"]})
- Required features: {', '.join(required)}
Do not classify as medical documents unless absolutely certain."""

        except Exception as e:
            logger.warning(f"Failed to load industry config for prompt: {str(e)}")
            return ""

        return ""

    def _extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from an image using OCR.

        Args:
            image_path: Path to the image file

        Returns:
            str: Extracted text

        Raises:
            TextExtractionError: If text extraction fails
        """
        logger.debug(f"Starting text extraction from image: {image_path}")

        # First validate the image file
        if not isinstance(image_path, (str, Path)):
            logger.error(f"Invalid path type: {type(image_path)}")
            raise TextExtractionError("Invalid file path type")

        # Check if file exists and is not empty
        try:
            if not Path(image_path).exists():
                raise TextExtractionError("File does not exist")
            if Path(image_path).stat().st_size == 0:
                raise TextExtractionError("File is empty")
        except Exception as e:
            logger.error(f"File access error: {str(e)}")
            raise TextExtractionError(f"File access error: {str(e)}")

        # Try to validate and open the image
        try:
            # First try to open with PIL to validate format
            try:
                with open(image_path, 'rb') as f:
                    header = f.read(8)  # Read first 8 bytes
                    if not any(header.startswith(sig) for sig in [b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'GIF87a', b'GIF89a']):
                        logger.error("Invalid image file format")
                        raise TextExtractionError("Invalid image file format")
            except Exception as e:
                logger.error(f"Failed to read file header: {str(e)}")
                raise TextExtractionError(f"Invalid image file: {str(e)}")

            with Image.open(str(image_path)) as img:
                try:
                    img.verify()
                    logger.debug("Image verification successful")
                except Exception as e:
                    logger.error(f"Image verification failed: {str(e)}")
                    raise TextExtractionError(f"Invalid image file: {str(e)}")

            # Try to read with OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error("OpenCV failed to read the image")
                raise TextExtractionError("Failed to read image file: Not a valid image format")

            # Convert to grayscale
            try:
                logger.debug("Converting image to grayscale")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                logger.error(f"Failed to convert image to grayscale: {str(e)}")
                raise TextExtractionError(f"Failed to process image: {str(e)}")

            # Try different preprocessing techniques
            texts = []
            logger.debug("Starting image preprocessing and OCR")

            # Try all four orientations
            angles = [0, 90, 180, 270]
            for angle in angles:
                logger.debug(f"Processing image at {angle} degrees rotation")
                # Rotate image if needed
                if angle != 0:
                    height, width = gray.shape
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(gray, rotation_matrix, (width, height),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                else:
                    rotated = gray

                # 1. Basic thresholding
                logger.debug("Applying basic thresholding")
                _, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                texts.append(pytesseract.image_to_string(binary))

                # 2. Adaptive thresholding
                logger.debug("Applying adaptive thresholding")
                binary_adaptive = cv2.adaptiveThreshold(
                    rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                texts.append(pytesseract.image_to_string(binary_adaptive))

                # 3. Denoised version
                logger.debug("Applying denoising")
                denoised = cv2.fastNlMeansDenoising(rotated)
                texts.append(pytesseract.image_to_string(denoised))

                # Try different PSM modes
                psm_modes = ['--psm 6', '--psm 1', '--psm 3']
                for psm in psm_modes:
                    logger.debug(f"Trying OCR with PSM mode: {psm}")
                    texts.append(pytesseract.image_to_string(binary, config=psm))
                    texts.append(pytesseract.image_to_string(binary_adaptive, config=psm))
                    texts.append(pytesseract.image_to_string(denoised, config=psm))

            # Choose the best result
            logger.debug("Selecting best OCR result")
            def score_text(text):
                # Count alphanumeric characters
                alnum_count = sum(c.isalnum() for c in text)
                # Bonus points for numbers
                number_count = sum(c.isdigit() for c in text) * 2
                # Bonus points for common document keywords
                keyword_bonus = sum(10 for keyword in ['INVOICE', 'TOTAL', 'AMOUNT', 'DATE']
                                  if keyword in text.upper())
                return alnum_count + number_count + keyword_bonus

            best_text = max(texts, key=score_text)
            logger.debug(f"Text extraction completed. Found text: {best_text[:50]}...")
            return best_text.strip()

        except Exception as e:
            logger.error(f"Error during text extraction: {str(e)}", exc_info=True)
            if isinstance(e, TextExtractionError):
                raise  # Re-raise TextExtractionError as is
            elif isinstance(e, pytesseract.TesseractError):
                if "Too few characters" in str(e):
                    logger.debug("Tesseract found no text in image")
                    return ""  # Return empty string for images with no text
            raise TextExtractionError(f"Failed to extract text from image: {str(e)}")

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
                return self._extract_text_from_image(file_path)
            else:
                # For text files or unknown types, try reading as text with different encodings
                try:
                    return file_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        return file_path.read_text(encoding='latin-1')
                    except UnicodeDecodeError:
                        raise TextExtractionError("File has encoding issues")
        except NotImplementedError:
            raise  # Re-raise NotImplementedError as is
        except pypdf.errors.FileNotDecryptedError:
            raise TextExtractionError("File is encrypted and cannot be read")
        except Exception as e:
            raise TextExtractionError(f"Failed to extract text: {str(e)}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        text = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                if pdf_reader.is_encrypted:
                    raise TextExtractionError("File is encrypted and cannot be read")
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise TextExtractionError(f"Failed to extract PDF text: {str(e)}")

    def _normalize_doc_type(self, doc_type: str, keep_spaces: bool = False) -> str:
        """Normalize document type string for comparison."""
        if not doc_type:
            return ""

        # Convert to lowercase and remove special characters
        normalized = doc_type.lower()
        normalized = re.sub(r'[^\w\s-]', '', normalized)

        # Handle common variations
        replacements = {
            'medical claim': 'medical_claim',
            'medicalclaim': 'medical_claim',
            'med claim': 'medical_claim',
            'medical invoice': 'medical_claim',
            'healthcare claim': 'medical_claim',
            'prescription': 'medical_prescription',
            'medical prescription': 'medical_prescription',
            'med prescription': 'medical_prescription',
            'rx': 'medical_prescription',
            'invoice': 'invoice',
            'bill': 'invoice',
            'billing': 'invoice',
            'statement': 'invoice',
            'financial statement': 'invoice'
        }

        # Try exact matches first
        if normalized in replacements:
            return replacements[normalized]

        # Try partial matches
        for key, value in replacements.items():
            if key in normalized:
                return value

        # Remove spaces unless keep_spaces is True
        if not keep_spaces:
            normalized = normalized.replace(' ', '_')

        return normalized

    def _classify_with_llm(self, text: str, industry: Optional[str] = None) -> ClassificationResult:
        """Classify document using LLM."""
        try:
            # Get industry-specific prompt additions
            industry_prompt = self._get_industry_prompt(industry) if industry else ""
            logger.debug(f"Using industry context: {industry}")
            logger.debug(f"Industry prompt: {industry_prompt}")

            # Enhanced prompt for better feature extraction
            prompt = f"""Analyze this document and classify it. Extract all relevant features.
Pay special attention to:
1. Document type
2. Dates (any format)
3. Amounts and monetary values
4. Document numbers (invoice numbers, patient IDs, etc.)
5. Payment terms (look for terms like 'Net 30', 'Due in X days', 'Payment due', etc.)

For financial documents:
- Look for payment terms in different formats (Net X, Due in X days, etc.)
- Extract all monetary amounts
- Note any late payment penalties or discounts

For healthcare documents:
- Extract all CPT codes (5-digit codes)
- Look for ICD-10 diagnosis codes
- Identify patient IDs and provider NPIs
- Note service dates and amounts

{industry_prompt}

Document text:
{text}

Provide a structured response with:
1. Document type
2. Confidence score (0.0-1.0)
3. All extracted features with their values

Format features as "Feature Type: Value" for easy parsing."""

            try:
                logger.debug("Calling OpenAI API...")
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-4",  # Using GPT-4 for better accuracy
                    messages=[
                        {"role": "system", "content": "You are a document classification assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=1000
                )

                # Validate API response
                if not response.choices:
                    logger.error("Invalid API response: No choices returned")
                    raise ClassificationError("Invalid API response: No choices returned")

                logger.debug(f"API Response received: {response.choices[0].message.content[:200]}...")
            except Exception as api_error:
                logger.error(f"API call failed: {str(api_error)}")
                if "choices" in str(api_error):
                    raise ClassificationError("Invalid API response: Malformed response structure")
                raise ClassificationError(f"Classification failed: {str(api_error)}")

            # Parse response
            doc_type, confidence, features = self._parse_llm_response(response.choices[0].message.content)

            # Validate against industry config if available
            if industry and doc_type != "unknown":
                config = INDUSTRY_CONFIGS.get(industry)
                if config:
                    for doc_config in config.document_types:
                        if self._normalize_doc_type(doc_config.name) == self._normalize_doc_type(doc_type):
                            validation_results = doc_config.validate_features(features)

                            # Calculate confidence based on feature presence and validation
                            feature_confidence = validation_results["confidence_score"]

                            # Boost confidence if all required features are present
                            if not validation_results["missing_required"]:
                                feature_confidence = min(1.0, feature_confidence + 0.2)

                            # Boost confidence if document type matches industry
                            if industry == "healthcare" and "medical" in doc_type:
                                feature_confidence = min(1.0, feature_confidence + 0.1)
                            elif industry == "financial" and doc_type == "invoice":
                                feature_confidence = min(1.0, feature_confidence + 0.1)

                            # Use the higher confidence between LLM and feature validation
                            confidence = max(confidence, feature_confidence)

                            # Add validation warnings
                            if validation_results["missing_required"]:
                                features.append({
                                    "type": "validation_warning",
                                    "values": [f"Missing required fields: {', '.join(validation_results['missing_required'])}"],
                                    "present": True
                                })

                            if validation_results["invalid_format"]:
                                features.append({
                                    "type": "validation_warning",
                                    "values": [f"Invalid format for fields: {', '.join(validation_results['invalid_format'])}"],
                                    "present": True
                                })
                            break

            return ClassificationResult(
                doc_type=doc_type,
                confidence=confidence,
                features=features
            )

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Classification failed: {str(e)}")

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response into structured format."""
        try:
            logger.debug(f"Starting to parse LLM response: {response[:200]}...")
            lines = response.split('\n')

            doc_type = None
            confidence = None
            features = []
            current_section = None
            current_features = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                line_lower = line.lower()
                logger.debug(f"Processing line: {line}")

                # Extract document type - now handles numbered list format
                if 'document type:' in line_lower:
                    doc_type = line.split(':', 1)[1].strip().lower()
                    doc_type = self._normalize_doc_type(doc_type)
                    logger.debug(f"Found document type: {doc_type}")
                    continue

                # Extract confidence - now handles numbered list format
                if 'confidence' in line_lower and ':' in line:
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        logger.debug(f"Found confidence: {confidence}")
                    except ValueError:
                        confidence = 0.5  # Default confidence
                    continue

                # Handle feature sections
                if 'extracted features:' in line_lower:
                    current_section = 'features'
                    continue

                if current_section == 'features' and line.startswith('-'):
                    # Parse feature line
                    if ':' in line:
                        feature_type, value = line.split(':', 1)
                        feature_type = feature_type.strip('- ').lower()
                        value = value.strip()

                        # Handle different feature types
                        if 'invoice number' in feature_type:
                            features.append({
                                "type": "invoice_number",
                                "values": [value],
                                "present": True
                            })
                        elif 'date' in feature_type:
                            features.append({
                                "type": "date",
                                "values": [value],
                                "present": True
                            })
                        elif 'bill to' in feature_type:
                            features.append({
                                "type": "bill_to",
                                "values": [value],
                                "present": True
                            })
                        elif any(amount_term in feature_type.lower() for amount_term in ['total', 'subtotal', 'amount', 'tax']):
                            amount_match = re.search(r'\$?([\d,]+\.?\d*)', value)
                            if amount_match:
                                features.append({
                                    "type": "amount",
                                    "values": [amount_match.group(1)],
                                    "present": True
                                })
                        elif 'payment terms' in feature_type.lower():
                            features.append({
                                "type": "payment_terms",
                                "values": [value],
                                "present": True
                            })
                        elif 'due date' in feature_type.lower():
                            features.append({
                                "type": "due_date",
                                "values": [value],
                                "present": True
                            })
                        else:
                            # Add as generic feature
                            features.append({
                                "type": feature_type,
                                "values": [value],
                                "present": True
                            })

            # Set defaults if needed
            if doc_type is None:
                doc_type = "unknown"
                confidence = 0.3
            elif confidence is None:
                confidence = 0.5

            return doc_type, confidence, features

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise ClassificationError(f"Invalid API response: {str(e)}")

    def _add_parsed_features(self, features: list, current_features: list):
        """Helper method to add parsed features to the feature list."""
        for feature_type, value in current_features:
            features.append({
                "type": feature_type,
                "values": [value],
                "present": True
            })

    def _extract_feature_items(self, feature_line: str) -> List[str]:
        """Extract individual features from a feature line."""
        # Remove the feature type prefix and split on commas or similar separators
        items = feature_line.split(':')[1].split(',') if ':' in feature_line else []
        return [item.strip() for item in items if item.strip()]

    def classify_file(self, file_path: Union[str, Path], industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a document file and extract its features.

        Args:
            file_path: Path to the document file
            industry: Optional industry context for classification

        Returns:
            Dictionary containing classification results
        """
        try:
            # Validate industry if specified
            if industry and industry not in self.config_manager.list_available_industries():
                raise ValueError(f"Invalid industry: {industry}")

            # Use default industry if none specified
            target_industry = industry or self.default_industry

            # Extract text content
            text = self.extract_text(file_path)

            # --- Pattern-based feature extraction ---
            patterns = self.pattern_store.get_patterns_by_industry(target_industry)
            pattern_matches = self.pattern_matcher.find_matches(text, patterns)
            pattern_features = [
                {
                    "feature_type": m.pattern.feature_type,
                    "text": m.text,
                    "confidence": m.confidence.value,
                    "context": m.context
                }
                for m in pattern_matches
            ]

            # --- LLM classification ---
            result = self._classify_with_llm(text, target_industry)

            # TODO: Merge/resolve LLM and pattern features in a unified way
            return {
                "class": result.doc_type,
                "confidence": result.confidence,
                "features": result.features,
                "pattern_features": pattern_features
            }

        except TextExtractionError as e:
            raise ClassificationError(f"Text extraction failed: {str(e)}")
        except ValueError as e:
            raise e  # Re-raise ValueError for invalid industry
        except Exception as e:
            raise ClassificationError(f"Classification failed: {str(e)}")