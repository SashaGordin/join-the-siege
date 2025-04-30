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
from .industry_config import (
    INDUSTRY_CONFIGS,
    IndustryConfig,
    DocumentTypeConfig,
    FeatureDefinition,
    FeatureImportance
)
import re

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

    def _get_industry_prompt(self, industry: str) -> str:
        """Get industry-specific prompt additions."""
        if industry == "healthcare":
            return """Industry Context: healthcare
Focus on medical document features and classify as medical documents when appropriate:
- Patient information (required for medical documents)
- CPT and ICD codes (required for medical claims)
- Provider details
- Service dates
- Medical charges"""
        elif industry == "financial":
            return """Industry Context: financial
Focus on financial document features and classify as financial documents when appropriate:
- Invoice numbers (required for invoices)
- Payment terms
- Due dates
- Line items
- Total amounts
Do not classify as medical documents unless absolutely certain."""
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
            logger.debug(f"Starting to parse LLM response: {response[:200]}...")  # Log first 200 chars of response
            lines = response.split('\n')

            doc_type = None
            confidence = None
            features = []
            ambiguity_detected = False

            current_section = None
            current_features = []

            # Track feature counts for ambiguity detection
            feature_counts = {
                "healthcare": 0,
                "financial": 0
            }

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                line_lower = line.lower()
                logger.debug(f"Processing line: {line}")

                # Extract document type
                if line_lower.startswith('document type:'):
                    doc_type = line.split(':', 1)[1].strip().lower()
                    # Normalize document type
                    doc_type = self._normalize_doc_type(doc_type)
                    logger.debug(f"Found document type: {doc_type}")
                    continue

                # Extract confidence
                if line_lower.startswith('confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        logger.debug(f"Found confidence: {confidence}")
                    except ValueError:
                        confidence = 0.5  # Default confidence for unparseable values
                        logger.warning(f"Could not parse confidence value from: {line}, using default: {confidence}")
                    continue

                # Handle feature sections
                if line.startswith('Features:'):
                    current_section = 'features'
                    logger.debug("Entering features section")
                    continue

                if current_section == 'features':
                    # Check if this is a new feature section
                    if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                        logger.debug(f"Found new feature section: {line}")
                        # Add previous features if any
                        if current_features:
                            self._add_parsed_features(features, current_features)
                        current_features = []
                        continue

                    # Parse feature line
                    if ':' in line:
                        feature_type, value = line.split(':', 1)
                        feature_type = feature_type.strip('- ').lower()
                        value = value.strip()
                        logger.debug(f"Processing feature - type: {feature_type}, value: {value}")

                        # Track industry-specific features for ambiguity detection
                        healthcare_terms = ['patient', 'cpt', 'icd', 'diagnosis', 'provider', 'medical']
                        financial_terms = ['invoice', 'payment', 'bill', 'account', 'net 30']

                        # Check healthcare terms
                        if any(term in feature_type or term in value.lower() for term in healthcare_terms):
                            feature_counts['healthcare'] += 1
                            logger.debug(f"Detected healthcare feature, count now: {feature_counts['healthcare']}")

                        # Check financial terms
                        if any(term in feature_type or term in value.lower() for term in financial_terms):
                            feature_counts['financial'] += 1
                            logger.debug(f"Detected financial feature, count now: {feature_counts['financial']}")

                        # Handle validation warnings and missing fields
                        if any(warning in feature_type.lower() for warning in ['warning', 'missing', 'validation']):
                            logger.debug(f"Found validation warning: {value}")
                            features.append({
                                "type": "validation_warning",
                                "values": [value],
                                "present": True
                            })
                            continue

                        # Enhanced healthcare feature detection
                        if any(id_term in feature_type for id_term in ['patient id', 'patient', 'id']):
                            logger.debug(f"Found patient ID: {value}")
                            features.append({
                                "type": "patient_id",
                                "values": [value],
                                "present": True
                            })
                            continue

                        # CPT code detection
                        cpt_terms = ['cpt', 'procedure', 'service code']
                        if any(cpt_term in feature_type or cpt_term in value.lower() for cpt_term in cpt_terms):
                            logger.debug(f"Processing CPT code: {value}")
                            cpt_match = re.search(r'(\d{5})', value)
                            if cpt_match:
                                features.append({
                                    "type": "cpt_code",
                                    "values": [cpt_match.group(1)],
                                    "present": True
                                })
                            continue

                        # Enhanced invoice number detection
                        if any(inv_term in feature_type.lower() for inv_term in ['invoice number', 'invoice #', 'invoice no', 'invoice id']):
                            logger.debug(f"Processing invoice number: {value}")
                            # Extract invoice number
                            inv_match = re.search(r'([A-Z0-9][-A-Z0-9]*)', value)
                            if inv_match:
                                features.append({
                                    "type": "invoice_number",
                                    "values": [inv_match.group(1)],
                                    "present": True
                                })
                            else:
                                features.append({
                                    "type": "invoice_number",
                                    "values": [value.strip()],
                                    "present": True
                                })
                            continue

                        # ICD code detection
                        icd_terms = ['icd', 'diagnosis']
                        if any(icd_term in feature_type or icd_term in value.lower() for icd_term in icd_terms):
                            logger.debug(f"Processing ICD code: {value}")
                            icd_match = re.search(r'([A-Z]\d{2}(?:\.\d+)?)', value)
                            if icd_match:
                                features.append({
                                    "type": "diagnosis_code",
                                    "values": [icd_match.group(1)],
                                    "present": True
                                })
                            continue

                        # NPI detection
                        if any(npi_term in feature_type for npi_term in ['npi', 'provider number']):
                            logger.debug(f"Processing NPI: {value}")
                            npi_match = re.search(r'(\d{10})', value)
                            if npi_match:
                                features.append({
                                    "type": "provider_npi",
                                    "values": [npi_match.group(1)],
                                    "present": True
                                })
                            continue

                        # Date detection
                        date_terms = ['date', 'due', 'service', 'dos']
                        if any(date_term in feature_type.lower() for date_term in date_terms):
                            logger.debug(f"Processing date: {value}")
                            date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', value)
                            if date_match:
                                features.append({
                                    "type": "date",
                                    "values": [date_match.group(1)],
                                    "present": True
                                })
                                continue
                            alt_date_match = re.search(r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})', value)
                            if alt_date_match:
                                features.append({
                                    "type": "date",
                                    "values": [alt_date_match.group(1)],
                                    "present": True
                                })
                            continue

                        # Amount detection
                        amount_terms = ['amount', 'total', 'price', 'cost', '$']
                        logger.debug(f"Checking for amount terms in feature type: {feature_type}")
                        logger.debug(f"Checking for $ in value: {value}")
                        if any(amount_term in feature_type.lower() for amount_term in amount_terms) or '$' in value:
                            logger.debug(f"Processing amount: {value}")
                            amount_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', value)
                            if amount_match:
                                logger.debug(f"Found amount match: {amount_match.group(0)}")
                                features.append({
                                    "type": "amount",
                                    "values": [amount_match.group(0)],
                                    "present": True
                                })
                                logger.debug(f"Added amount feature: {features[-1]}")
                            else:
                                logger.debug(f"No amount pattern found in value: {value}")
                            continue

                        # Payment terms detection
                        payment_terms = ['payment', 'terms', 'due', 'net']
                        logger.debug(f"Checking for payment terms in: {feature_type} - {value}")
                        if any(term in feature_type.lower() or term in value.lower() for term in payment_terms):
                            logger.debug(f"Processing payment terms: {value}")
                            terms_match = re.search(r'(Net\s+\d+|Due\s+\d+(?:\s+days)?)', value, re.IGNORECASE)
                            if terms_match:
                                logger.debug(f"Found payment terms: {terms_match.group(1)}")
                                features.append({
                                    "type": "payment_terms",
                                    "values": [terms_match.group(1)],
                                    "present": True
                                })
                            else:
                                logger.debug(f"No payment terms pattern found in value: {value}")
                            continue

                        # Add as generic feature if no specific type matched
                        logger.debug(f"Adding generic feature - type: {feature_type}, value: {value}")
                        current_features.append((feature_type, value))

            # Add any remaining features
            if current_features:
                logger.debug(f"Adding remaining features: {current_features}")
                self._add_parsed_features(features, current_features)

            # Detect ambiguity
            if feature_counts['healthcare'] > 0 and feature_counts['financial'] > 0:
                logger.debug("Detected mixed industry signals")
                ambiguity_detected = True
                features.append({
                    "type": "validation_warning",
                    "values": ["Document contains mixed industry signals"],
                    "present": True
                })

            # Handle incomplete responses and infer document type
            if not doc_type or doc_type == "unknown":
                logger.debug("Document type missing or unknown, attempting to infer from features")

                # Check for strong healthcare indicators
                healthcare_indicators = any(f["type"] == "patient_id" for f in features) or \
                                     any(f["type"] == "cpt_code" for f in features) or \
                                     any(f["type"] == "diagnosis_code" for f in features) or \
                                     feature_counts['healthcare'] > feature_counts['financial'] or \
                                     any('medical' in str(f).lower() for f in features) or \
                                     'medical' in text.lower()
                logger.debug(f"Healthcare indicators present: {healthcare_indicators}")
                logger.debug(f"Healthcare feature count: {feature_counts['healthcare']}")
                logger.debug(f"Financial feature count: {feature_counts['financial']}")

                # Check for strong financial indicators
                financial_indicators = any(f["type"] == "payment_terms" for f in features) or \
                                    any(f["type"] == "invoice_number" for f in features) or \
                                    feature_counts['financial'] > feature_counts['healthcare'] or \
                                    any('invoice' in str(f).lower() for f in features) or \
                                    'invoice' in text.lower()
                logger.debug(f"Financial indicators present: {financial_indicators}")

                if healthcare_indicators and not financial_indicators:
                    doc_type = "medical_claim"
                    confidence = confidence or 0.6  # Lower confidence for inferred type
                    logger.debug("Inferred document type: medical_claim")
                elif financial_indicators and not healthcare_indicators:
                    doc_type = "invoice"
                    confidence = confidence or 0.6  # Lower confidence for inferred type
                    logger.debug("Inferred document type: invoice")
                elif healthcare_indicators and financial_indicators:
                    # Use industry context to break the tie
                    if industry == "healthcare":
                        doc_type = "medical_claim"
                        confidence = 0.6
                        logger.debug("Using healthcare industry context to resolve ambiguity")
                    elif industry == "financial":
                        doc_type = "invoice"
                        confidence = 0.6
                        logger.debug("Using financial industry context to resolve ambiguity")
                    else:
                        # Default to the type with more indicators
                        doc_type = "medical_claim" if feature_counts['healthcare'] >= feature_counts['financial'] else "invoice"
                        confidence = 0.5  # Low confidence due to ambiguity
                        logger.debug(f"Defaulting to {doc_type} based on feature counts")
                elif industry:  # If we have an industry context but no clear indicators
                    doc_type = "medical_claim" if industry == "healthcare" else "invoice"
                    confidence = 0.5  # Low confidence due to lack of indicators
                    logger.debug(f"Using industry context {industry} to set default type: {doc_type}")
                else:
                    doc_type = "unknown"
                    confidence = 0.5
                    logger.debug("No clear indicators or industry context, setting type to unknown")

            # Set defaults for incomplete responses
            if not doc_type:
                doc_type = "unknown"
                logger.debug("No document type could be inferred, setting to unknown")
            if confidence is None:
                confidence = 0.5
                logger.debug("No confidence score found, using default: 0.5")

            # Add validation warnings for missing required features
            if doc_type == "medical_claim":
                required_features = {"patient_id", "cpt_code", "date", "amount"}
                found_features = {f["type"] for f in features}
                missing = required_features - found_features
                if missing:
                    features.append({
                        "type": "validation_warning",
                        "values": [f"Missing required fields: {', '.join(missing)}"],
                        "present": True
                    })
                    # Reduce confidence for missing required fields
                    confidence *= 0.8
            elif doc_type == "invoice":
                required_features = {"invoice_number", "date", "amount"}
                found_features = {f["type"] for f in features}
                missing = required_features - found_features
                if missing:
                    features.append({
                        "type": "validation_warning",
                        "values": [f"Missing required fields: {', '.join(missing)}"],
                        "present": True
                    })
                    # Reduce confidence for missing required fields
                    confidence *= 0.8

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
            if industry and industry not in INDUSTRY_CONFIGS:
                raise ValueError(f"Invalid industry: {industry}")

            # Use default industry if none specified
            target_industry = industry or self.default_industry

            # Extract text content
            text = self.extract_text(file_path)

            # Classify using LLM with industry context
            result = self._classify_with_llm(text, target_industry)

            return {
                "class": result.doc_type,
                "confidence": result.confidence,
                "features": result.features
            }

        except TextExtractionError as e:
            raise ClassificationError(f"Text extraction failed: {str(e)}")
        except ValueError as e:
            raise e  # Re-raise ValueError for invalid industry
        except Exception as e:
            raise ClassificationError(f"Classification failed: {str(e)}")