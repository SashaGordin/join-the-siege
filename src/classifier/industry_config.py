"""Industry-specific configuration for document classification."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class FeatureImportance(Enum):
    """Importance levels for document features."""
    REQUIRED = "required"  # Must be present for valid classification
    RECOMMENDED = "recommended"  # Should be present for high confidence
    OPTIONAL = "optional"  # Nice to have but not necessary


@dataclass
class FeatureDefinition:
    """Definition of a document feature."""
    name: str
    importance: FeatureImportance
    validation_pattern: Optional[str] = None  # Regex pattern for validation
    example_values: List[str] = None  # Example valid values
    description: str = ""

    def __post_init__(self):
        if self.example_values is None:
            self.example_values = []


@dataclass
class DocumentTypeConfig:
    """Configuration for a specific document type."""
    name: str
    features: List[FeatureDefinition]
    confidence_threshold: float  # Minimum confidence for valid classification
    description: str = ""

    def validate_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extracted features against this configuration.

        Args:
            features: List of extracted features

        Returns:
            Dict containing validation results
        """
        results = {
            "valid": True,
            "missing_required": [],
            "invalid_format": [],
            "confidence_score": 0.0
        }

        # Track found features
        found_features = {f["type"]: f for f in features if f.get("present", False)}

        # Check required features
        total_weight = 0
        feature_score = 0

        for feature_def in self.features:
            feature = found_features.get(feature_def.name)

            # Calculate weight based on importance
            weight = {
                FeatureImportance.REQUIRED: 1.0,
                FeatureImportance.RECOMMENDED: 0.5,
                FeatureImportance.OPTIONAL: 0.2
            }[feature_def.importance]

            total_weight += weight

            if not feature:
                if feature_def.importance == FeatureImportance.REQUIRED:
                    results["missing_required"].append(feature_def.name)
                    results["valid"] = False
                continue

            # Validate format if pattern exists
            if feature_def.validation_pattern and feature.get("values"):
                pattern = re.compile(feature_def.validation_pattern)
                valid_values = [v for v in feature["values"] if pattern.match(str(v))]
                if not valid_values:
                    results["invalid_format"].append(feature_def.name)
                    if feature_def.importance == FeatureImportance.REQUIRED:
                        results["valid"] = False
                else:
                    feature_score += weight
            else:
                feature_score += weight

        # Calculate confidence score
        results["confidence_score"] = feature_score / total_weight if total_weight > 0 else 0.0

        return results


@dataclass
class IndustryConfig:
    """Configuration for an industry."""
    name: str
    document_types: List[DocumentTypeConfig]
    description: str = ""


# Healthcare industry configuration
HEALTHCARE_CONFIG = IndustryConfig(
    name="healthcare",
    description="Healthcare industry document configuration",
    document_types=[
        DocumentTypeConfig(
            name="medical_claim",
            description="Medical insurance claim document",
            confidence_threshold=0.7,
            features=[
                FeatureDefinition(
                    name="patient_id",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^(?:Patient|ID)[-:]?\s*[A-Z0-9-]+$",
                    example_values=["Patient: 12345", "ID: PAT-2024-001"],
                    description="Patient identification number"
                ),
                FeatureDefinition(
                    name="cpt_code",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^(?:CPT:?\s*)?(\d{5})(?:\s*-\s*.*)?$",
                    example_values=["CPT: 99213", "99213 - Office Visit"],
                    description="CPT procedure codes"
                ),
                FeatureDefinition(
                    name="diagnosis_code",
                    importance=FeatureImportance.RECOMMENDED,
                    validation_pattern=r"^(?:ICD-10:?\s*)?[A-Z]\d{2}(?:\.\d+)?$",
                    example_values=["ICD-10: J45.901"],
                    description="ICD-10 diagnosis codes"
                ),
                FeatureDefinition(
                    name="provider_npi",
                    importance=FeatureImportance.RECOMMENDED,
                    validation_pattern=r"^(?:NPI:?\s*)?(\d{10})$",
                    example_values=["NPI: 1234567890"],
                    description="Provider NPI number"
                ),
                FeatureDefinition(
                    name="date_of_service",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
                    example_values=["03/15/2024"],
                    description="Date when service was provided"
                ),
                FeatureDefinition(
                    name="amount",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\$?\d+(?:,\d{3})*(?:\.\d{2})?$",
                    example_values=["$150.00", "1,234.56"],
                    description="Amount charged for service"
                )
            ]
        ),
        DocumentTypeConfig(
            name="medical_prescription",
            description="Medical prescription document",
            confidence_threshold=0.8,
            features=[
                FeatureDefinition(
                    name="patient_name",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^[A-Za-z\s,\.]+$",
                    example_values=["John Doe"],
                    description="Patient's full name"
                ),
                FeatureDefinition(
                    name="medication",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^[A-Za-z0-9\s\-]+(?:\s+\d+(?:mg|ml|g))?$",
                    example_values=["Amoxicillin 500mg"],
                    description="Medication name and dosage"
                ),
                FeatureDefinition(
                    name="prescription_date",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
                    example_values=["03/15/2024"],
                    description="Date prescription was written"
                ),
                FeatureDefinition(
                    name="prescriber_npi",
                    importance=FeatureImportance.RECOMMENDED,
                    validation_pattern=r"^(?:NPI:?\s*)?(\d{10})$",
                    example_values=["NPI: 1234567890"],
                    description="Prescriber's NPI number"
                )
            ]
        )
    ]
)

# Financial industry configuration
FINANCIAL_CONFIG = IndustryConfig(
    name="financial",
    description="Financial industry document configuration",
    document_types=[
        DocumentTypeConfig(
            name="invoice",
            description="Invoice document",
            confidence_threshold=0.7,
            features=[
                FeatureDefinition(
                    name="invoice_number",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^(?:INV|INVOICE)[-:]?\s*[A-Z0-9-]+$",
                    example_values=["INV-2024-001", "INVOICE: 12345"],
                    description="Invoice identification number"
                ),
                FeatureDefinition(
                    name="amount",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\$?\d+(?:,\d{3})*(?:\.\d{2})?$",
                    example_values=["$1,234.56"],
                    description="Invoice amount"
                ),
                FeatureDefinition(
                    name="date",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
                    example_values=["03/15/2024"],
                    description="Invoice date"
                ),
                FeatureDefinition(
                    name="payment_terms",
                    importance=FeatureImportance.RECOMMENDED,
                    validation_pattern=r"^(?:Net\s+\d+|Due\s+(?:on|in)\s+\d+\s+days)$",
                    example_values=["Net 30", "Due in 15 days"],
                    description="Payment terms"
                )
            ]
        ),
        DocumentTypeConfig(
            name="bank_statement",
            description="Bank statement document",
            confidence_threshold=0.8,
            features=[
                FeatureDefinition(
                    name="account_number",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^(?:Account|Acct)[-:]?\s*[A-Z0-9-]+$",
                    example_values=["Account: 12345", "Acct-2024-001"],
                    description="Bank account number"
                ),
                FeatureDefinition(
                    name="statement_date",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
                    example_values=["03/15/2024"],
                    description="Statement date"
                ),
                FeatureDefinition(
                    name="balance",
                    importance=FeatureImportance.REQUIRED,
                    validation_pattern=r"^\$?\d+(?:,\d{3})*(?:\.\d{2})?$",
                    example_values=["$1,234.56"],
                    description="Account balance"
                ),
                FeatureDefinition(
                    name="transaction",
                    importance=FeatureImportance.RECOMMENDED,
                    validation_pattern=r"^.*\$?\d+(?:,\d{3})*(?:\.\d{2})?(?:\s+[A-Z]+)?$",
                    example_values=["DEPOSIT $500.00", "WITHDRAWAL $100.00"],
                    description="Transaction entries"
                )
            ]
        )
    ]
)

# Dictionary of all industry configurations
INDUSTRY_CONFIGS: Dict[str, IndustryConfig] = {
    "healthcare": HEALTHCARE_CONFIG,
    "financial": FINANCIAL_CONFIG
}