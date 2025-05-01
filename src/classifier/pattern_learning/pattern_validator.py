"""Pattern validation and generation module."""

import re
import uuid
import logging
from typing import Optional, Dict, Any

from .models import Pattern, PatternType

logger = logging.getLogger(__name__)

class PatternValidator:
    """Validates and generates patterns from candidate text."""

    def __init__(self):
        """Initialize the pattern validator."""
        # Common regex patterns for different feature types
        self.feature_patterns = {
            "amount": r"^\$?\d+(\.\d{2})?$",
            "date": (
                r"^\d{4}-\d{2}-\d{2}$|"
                r"^\d{2}/\d{2}/\d{4}$|"
                r"^\d{2}-\d{2}-\d{4}$"
            ),
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
            "url": r"^https?://[^\s/$.?#].[^\s]*$"
        }

    def generate_pattern(
        self,
        text: str,
        feature_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Pattern]:
        """
        Generate a pattern from candidate text.

        Args:
            text: The text to generate a pattern from
            feature_type: Type of feature this pattern represents
            metadata: Additional context metadata

        Returns:
            A validated Pattern object or None if invalid
        """
        try:
            # Check if we have a predefined pattern for this feature type
            if feature_type in self.feature_patterns:
                pattern_str = self.feature_patterns[feature_type]
                if not re.match(pattern_str, text):
                    # Text doesn't match predefined pattern
                    return None
                return self._create_pattern(
                    pattern_str,
                    feature_type,
                    PatternType.REGEX,
                    metadata
                )

            # For custom feature types, generate a pattern
            pattern_str = self._generate_regex_pattern(text)
            if not pattern_str:
                return None

            # Validate the generated pattern
            if self._validate_pattern(pattern_str, text):
                return self._create_pattern(
                    pattern_str,
                    feature_type,
                    PatternType.REGEX,
                    metadata
                )

            return None

        except Exception as e:
            logger.error(f"Error generating pattern: {str(e)}")
            return None

    def _generate_regex_pattern(self, text: str) -> Optional[str]:
        """
        Generate a regex pattern from text.

        Args:
            text: Text to generate pattern from

        Returns:
            Regex pattern string or None if invalid
        """
        if not text or len(text) < 3:
            return None

        try:
            # Escape special regex characters
            pattern = re.escape(text)

            # Define pattern components
            components = {
                r'\d+': r'\d+',  # Numbers
                r'[A-Za-z]+': r'[A-Za-z]+',  # Letters
                r'[@\.\-_]': r'[@\.\-_]',  # Common separators
                r'\d{4}': r'\d{4}',  # Year-like numbers
                r'\d{2,3}': r'\d{2,3}',  # Month/day numbers
            }

            # Replace specific sequences with more general patterns
            for sequence, replacement in components.items():
                if re.search(sequence, pattern):
                    pattern = pattern.replace(sequence, replacement)

            # Add word boundaries for more precise matching
            pattern = f"^{pattern}$"

            # Validate the generated pattern
            if not self._validate_pattern(pattern, text):
                return None

            return pattern

        except re.error:
            return None

    def _validate_pattern(self, pattern: str, test_text: str) -> bool:
        """
        Validate a generated pattern.

        Args:
            pattern: The pattern to validate
            test_text: Text the pattern should match

        Returns:
            True if pattern is valid, False otherwise
        """
        try:
            # Compile pattern to check syntax
            regex = re.compile(pattern)

            # Test if pattern matches original text
            if not regex.match(test_text):
                return False

            # For basic patterns (like \d+), we accept them as is
            basic_patterns = [r'\d+', r'[A-Za-z]+', r'[0-9]+']
            if pattern in basic_patterns:
                return True

            # Check pattern isn't too permissive
            too_permissive = [r'.*', r'.+', r'\w+']
            if pattern in too_permissive:
                return False

            # Check pattern length is reasonable
            if len(pattern) < 3 or len(pattern) > 200:
                return False

            # Test pattern against some invalid inputs
            test_cases = [
                "",  # Empty string
                " ",  # Single space
                "a",  # Single character
                "." * 300,  # Too long
                test_text + "extra",  # Original text with extra
                "invalid"  # Common invalid input
            ]

            # Pattern should not match any of these test cases
            for test in test_cases:
                if test != test_text and regex.match(test):
                    return False

            return True

        except re.error:
            return False
        except Exception as e:
            logger.error(f"Error validating pattern: {str(e)}")
            return False

    def _create_pattern(
        self,
        expression: str,
        feature_type: str,
        pattern_type: PatternType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Pattern:
        """Create a new Pattern instance."""
        return Pattern(
            id=str(uuid.uuid4()),
            type=pattern_type,
            expression=expression,
            feature_type=feature_type,
            industry=metadata.get("industry") if metadata else None
        )