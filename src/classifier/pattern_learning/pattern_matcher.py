"""Pattern matching system for feature detection."""

import re
from typing import List, Dict, Optional, Set, Tuple, Any
from difflib import SequenceMatcher
import logging
from Levenshtein import distance as levenshtein_distance
from jellyfish import soundex, metaphone
from .models import (
    Pattern,
    PatternMatch,
    PatternType,
    MatchType,
    ConfidenceScore
)

logger = logging.getLogger(__name__)

class PatternMatcher:
    """
    Matches patterns against text to identify features.

    Features:
    - Multiple pattern types (regex, fuzzy, context)
    - Confidence scoring
    - Context awareness
    - Match ranking
    - Named capture groups
    - Boundary awareness
    - Pattern validation
    """

    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize pattern matcher.

        Args:
            min_confidence: Minimum confidence threshold for matches
        """
        self.min_confidence = min_confidence
        self._regex_cache: Dict[str, Tuple[re.Pattern, Dict[str, str]]] = {}

    def _preprocess_regex_pattern(self, pattern: Pattern) -> Tuple[str, Dict[str, str]]:
        """
        Preprocess regex pattern to add boundary awareness and extract named groups.

        Args:
            pattern: Pattern to preprocess

        Returns:
            Tuple of (processed pattern string, group info dictionary)
        """
        expression = pattern.expression
        group_info = {}

        # Extract and validate named groups
        named_groups = re.findall(r'\(\?P<([^>]+)>', expression)
        for group in named_groups:
            group_info[group] = "named_group"

        # Add word boundaries if not present and pattern looks like a word
        if not any(c in expression for c in r'^\$\b\B'):
            if re.match(r'^[\\w\\s-]+$', expression):
                expression = r'\b' + expression + r'\b'

        # Add start/end anchors for full line matches if appropriate
        if '\n' not in expression and not any(c in expression for c in r'^\$'):
            if pattern.metadata.get('full_line_match', False):
                expression = '^' + expression + '$'

        return expression, group_info

    def _validate_regex_pattern(self, pattern_str: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate regex pattern for common issues.

        Args:
            pattern_str: Pattern string to validate
            metadata: Optional pattern metadata

        Returns:
            True if pattern is valid, False otherwise
        """
        try:
            # Try to compile the pattern first
            re.compile(pattern_str)

            # Check for common issues
            issues = []

            # Check for unbalanced brackets
            stack = []
            escaped = False
            in_char_class = False

            for i, c in enumerate(pattern_str):
                if c == '\\' and not escaped:
                    escaped = True
                    continue

                if escaped:
                    escaped = False
                    continue

                if c == '[' and not escaped:
                    if in_char_class:
                        issues.append(f"Nested character class at position {i}")
                    stack.append(i)
                    in_char_class = True
                elif c == ']' and not escaped:
                    if not stack:
                        issues.append(f"Unmatched closing bracket at position {i}")
                    elif in_char_class:
                        stack.pop()
                        in_char_class = False

            if in_char_class:
                issues.append(f"Unterminated character set at position {stack[-1]}")

            # Check for unbalanced parentheses
            stack = []
            escaped = False
            for i, c in enumerate(pattern_str):
                if c == '\\' and not escaped:
                    escaped = True
                    continue
                if escaped:
                    escaped = False
                    continue
                if c == '(' and not escaped:
                    stack.append(i)
                elif c == ')' and not escaped:
                    if not stack:
                        issues.append(f"Unmatched closing parenthesis at position {i}")
                    else:
                        stack.pop()
            if stack:
                issues.append(f"Unclosed group at position {stack[-1]}")

            # Check for inefficient patterns only if not explicitly allowed
            if not (metadata and metadata.get("allow_inefficient")):
                if '.*.*' in pattern_str:
                    issues.append("Multiple consecutive wildcards")
                if pattern_str.count('*') > 3:
                    issues.append("Too many wildcards")
                if pattern_str.startswith('.*') and pattern_str.endswith('.*'):
                    issues.append("Pattern starts and ends with wildcard")

            if issues:
                logger.warning(f"Pattern validation issues: {', '.join(issues)}")
                return len(issues) == 0 or (metadata and metadata.get("allow_inefficient"))

            return True

        except re.error as e:
            logger.error(f"Invalid regex pattern: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating pattern: {str(e)}")
            return False

    def _find_regex_matches(self, text: str, pattern: Pattern) -> List[PatternMatch]:
        """Find regex pattern matches."""
        if not text or not text.strip():
            return []

        if not self._validate_regex_pattern(pattern.expression, pattern.metadata):
            return []

        matches = []
        try:
            for match in re.finditer(pattern.expression, text):
                # Skip empty matches
                if not match.group().strip():
                    continue

                # Calculate named group confidence
                named_groups = {}
                group_confidence = 1.0
                for name, value in match.groupdict().items():
                    if value and value.strip():
                        named_groups[name] = value.strip()
                        group_confidence *= 0.9  # Slight penalty for each group

                # Calculate overall confidence
                confidence = self._calculate_regex_confidence(match, pattern, group_confidence)

                # Add inefficiency penalty if pattern is marked as inefficient
                if pattern.metadata and pattern.metadata.get("allow_inefficient"):
                    confidence.factors["inefficient_pattern"] = 0.85  # 15% penalty instead of 30%
                    confidence.value *= confidence.factors["inefficient_pattern"]

                # Skip low confidence matches
                if confidence.value < self.min_confidence:
                    continue

                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        match_type=MatchType.EXACT,
                        confidence=confidence,
                        context={"named_groups": named_groups}
                    )
                )
        except re.error:
            logger.error(f"Error matching pattern: {pattern.expression}")
            return []

        return matches

    def _calculate_group_confidence(
        self,
        named_groups: Dict[str, str],
        group_info: Dict[str, str]
    ) -> float:
        """
        Calculate confidence based on named group matches.

        Args:
            named_groups: Dictionary of named group matches
            group_info: Dictionary of group information

        Returns:
            Confidence score for group matches
        """
        if not group_info:
            return 1.0

        # Calculate how many expected groups were matched
        matched_groups = sum(1 for group in group_info if group in named_groups)
        return matched_groups / len(group_info)

    def _calculate_regex_confidence(
        self,
        match: re.Match,
        pattern: Pattern,
        group_confidence: float
    ) -> ConfidenceScore:
        """Calculate confidence score for regex match."""
        # Get matched text
        text = match.group()

        # Skip empty or whitespace-only matches
        if not text or not text.strip():
            return ConfidenceScore(value=0.0)

        # Calculate match quality factors
        factors = {
            "pattern_confidence": pattern.confidence.value,
            "match_length": min(1.0, len(text) / 10),  # Length factor
            "group_count": min(1.0, len(match.groups()) / 3),   # Capture group factor
            "named_groups": group_confidence,                    # Named group factor
            "boundary_match": 1.0 if (                          # Boundary factor
                match.start() == 0 or not match.string[match.start() - 1].isalnum()
            ) and (
                match.end() == len(match.string) or not match.string[match.end()].isalnum()
            ) else 0.8,
            "word_match": 1.0 if re.match(r'^\w+$', text) else 0.8  # Word match factor
        }

        # Calculate weighted average
        weights = {
            "pattern_confidence": 0.3,
            "match_length": 0.15,
            "group_count": 0.1,
            "named_groups": 0.2,
            "boundary_match": 0.15,
            "word_match": 0.1
        }

        # Base confidence
        confidence = sum(score * weights[factor] for factor, score in factors.items())

        # Boost confidence for exact word matches with boundaries
        if factors["word_match"] == 1.0 and factors["boundary_match"] == 1.0:
            confidence = min(1.0, confidence * 1.2)

        # Boost confidence for full matches
        if match.group() == match.string.strip():
            confidence = min(1.0, confidence * 1.1)

        # Penalize very short matches unless they're exact words
        if len(text) < 3 and not re.match(r'^\w+$', text):
            confidence *= 0.8

        # Penalize matches with non-word characters
        if re.search(r'[^\w\s]', text):
            confidence *= 0.9

        return ConfidenceScore(value=confidence, factors=factors)

    def find_matches(
        self,
        text: str,
        patterns: List[Pattern],
        context: Optional[Dict[str, str]] = None
    ) -> List[PatternMatch]:
        """
        Find all pattern matches in text.

        Args:
            text: Text to search in
            patterns: List of patterns to match
            context: Optional contextual information

        Returns:
            List of pattern matches found
        """
        matches: List[PatternMatch] = []

        for pattern in patterns:
            if pattern.type == PatternType.REGEX:
                matches.extend(self._find_regex_matches(text, pattern))
            elif pattern.type == PatternType.FUZZY:
                matches.extend(self._find_fuzzy_matches(text, pattern))
            elif pattern.type == PatternType.CONTEXT:
                if context:
                    matches.extend(self._find_context_matches(text, pattern, context))
            elif pattern.type == PatternType.LEARNED:
                matches.extend(self._find_learned_matches(text, pattern))

        # Filter by confidence and sort by score
        matches = [m for m in matches if m.confidence.value >= self.min_confidence]
        matches.sort(key=lambda m: m.confidence.value, reverse=True)

        return self._resolve_overlapping_matches(matches)

    def _find_fuzzy_matches(self, text: str, pattern: Pattern) -> List[PatternMatch]:
        """Find fuzzy matches in text using pattern examples."""
        matches = []
        if not pattern.examples:
            return matches

        # For name patterns, use specialized matching
        if pattern.metadata.get('is_name', False):
            # Split text into potential name chunks
            text_chunks = self._get_text_chunks(text)

            for chunk in text_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                # Compare with examples
                max_score = 0
                best_example = None
                best_factors = {}

                for example in pattern.examples:
                    # Calculate similarity scores
                    seq_ratio = SequenceMatcher(None, chunk.lower(), example.lower()).ratio()
                    lev_ratio = 1 - (levenshtein_distance(chunk, example) / max(len(chunk), len(example)))
                    ngram_ratio = self._calculate_ngram_similarity(chunk, example)
                    phonetic_ratio = self._calculate_phonetic_similarity(chunk, example)

                    # Calculate name-specific confidence
                    confidence = self._calculate_name_confidence(
                        seq_ratio,
                        lev_ratio,
                        ngram_ratio,
                        phonetic_ratio,
                        chunk,
                        example
                    )

                    if confidence > max_score:
                        max_score = confidence
                        best_example = example
                        best_factors = {
                            'sequence_match': seq_ratio,
                            'levenshtein': lev_ratio,
                            'ngram': ngram_ratio,
                            'phonetic': phonetic_ratio,
                            'matched_example': example
                        }

                # Only create a match if the score is high enough
                if max_score >= pattern.metadata.get('min_confidence', self.min_confidence):
                    # Find position in original text
                    start_pos = text.lower().find(chunk.lower())
                    if start_pos >= 0:
                        matches.append(
                            PatternMatch(
                                pattern=pattern,
                                text=chunk,
                                start=start_pos,
                                end=start_pos + len(chunk),
                                match_type=MatchType.PARTIAL,
                                confidence=ConfidenceScore(
                                    value=max_score,
                                    factors=best_factors
                                )
                            )
                        )

        # For address patterns, use specialized matching
        elif pattern.metadata.get('is_address', False):
            # Split text into lines to handle each address separately
            lines = text.split('\n')

            # Process each line
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Parse the input address
                text_parts = self._parse_address(line)
                if not text_parts:
                    continue

                text_num, text_name, text_type = text_parts

                # Compare with examples
                max_score = 0
                best_example = None
                best_factors = {}

                for example in pattern.examples:
                    # Parse example address
                    example_parts = self._parse_address(example)
                    if not example_parts:
                        continue

                    example_num, example_name, example_type = example_parts

                    # Calculate component similarities
                    number_sim = 1.0 if text_num == example_num else 0.0

                    # Street name similarity using sequence matcher
                    name_sim = SequenceMatcher(None, text_name.lower(), example_name.lower()).ratio()

                    # Street type similarity using normalization
                    type_map = {
                        'st': 'street', 'str': 'street', 'street': 'street',
                        'ave': 'avenue', 'av': 'avenue', 'avenue': 'avenue',
                        'rd': 'road', 'road': 'road',
                        'ln': 'lane', 'lane': 'lane',
                        'dr': 'drive', 'drive': 'drive',
                        'blvd': 'boulevard', 'boulevard': 'boulevard'
                    }
                    text_type_norm = type_map.get(text_type.lower(), text_type.lower())
                    example_type_norm = type_map.get(example_type.lower(), example_type.lower())
                    type_sim = 1.0 if text_type_norm == example_type_norm else 0.0

                    # Calculate weighted score with strict requirements
                    score = 0.0  # Start with zero score

                    # Only proceed if number matches exactly
                    if number_sim == 1.0:
                        # Street name must be very similar
                        if name_sim >= 0.8:
                            # Street type must match after normalization
                            if type_sim == 1.0:
                                score = (
                                    (number_sim * 0.4) +
                                    (name_sim * 0.4) +
                                    (type_sim * 0.2)
                                )

                    # Store best match
                    if score > max_score:
                        max_score = score
                        best_example = example
                        best_factors = {
                            'sequence_match': name_sim,
                            'number_match': number_sim,
                            'type_match': type_sim
                        }

                # Only create a match if the score is high enough
                if max_score >= 0.8:  # Increased threshold for stricter matching
                    start_pos = sum(len(l) + 1 for l in lines[:line_num])
                    matches.append(
                        PatternMatch(
                            pattern=pattern,
                            text=line,
                            start=start_pos,
                            end=start_pos + len(line),
                            match_type=MatchType.PARTIAL,
                            confidence=ConfidenceScore(
                                value=max_score,
                                factors=best_factors
                            )
                        )
                    )

        else:
            # Regular fuzzy matching for non-address patterns
            # Split text into words for word-by-word matching
            words = text.lower().split()

            for word_idx, word in enumerate(words):
                word_start = len(' '.join(words[:word_idx])) + (word_idx > 0)
                word_end = word_start + len(word)

                best_score = 0
                best_factors = {}
                best_example = None

                for example in pattern.examples:
                    # Calculate similarity scores
                    seq_ratio = SequenceMatcher(None, word.lower(), example.lower()).ratio()
                    lev_ratio = 1 - (levenshtein_distance(word, example) / max(len(word), len(example)))
                    ngram_ratio = self._calculate_ngram_similarity(word, example)

                    confidence = self._calculate_text_confidence(
                        seq_ratio,
                        lev_ratio,
                        ngram_ratio,
                        word,  # Pass the current word
                        example  # Pass the example
                    )

                    if confidence > best_score:
                        best_score = confidence
                        best_example = example
                        best_factors = {
                            'sequence_match': seq_ratio,
                            'levenshtein': lev_ratio,
                            'ngram': ngram_ratio,
                            'matched_example': example
                        }

                if best_score >= pattern.metadata.get('min_confidence', self.min_confidence):
                    matches.append(
                        PatternMatch(
                            pattern=pattern,
                            text=word,
                            start=word_start,
                            end=word_end,
                            match_type=MatchType.PARTIAL,
                            confidence=ConfidenceScore(
                                value=best_score,
                                factors=best_factors
                            )
                        )
                    )

        return matches

    def _parse_address(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Parse address into components."""
        parts = text.strip().lower().split()
        if len(parts) < 2:
            return None

        # Find number component
        number = next((p for p in parts if p.isdigit()), None)
        if not number:
            return None

        # Get remaining parts
        remaining = parts[:parts.index(number)] + parts[parts.index(number)+1:]

        # Find street type
        street_types = {
            'st', 'street', 'ave', 'avenue', 'rd', 'road',
            'ln', 'lane', 'dr', 'drive', 'blvd', 'boulevard'
        }
        street_type = next((p for p in remaining if p.lower() in street_types), None)
        if not street_type:
            return None

        # Get street name
        street_name = ' '.join(p for p in remaining if p != street_type)
        if not street_name:
            return None

        return number, street_name, street_type

    def _normalize_street_type(self, street_type: str) -> str:
        """Normalize street type to canonical form."""
        type_map = {
            'st': 'street', 'str': 'street', 'street': 'street',
            'ave': 'avenue', 'av': 'avenue', 'avenue': 'avenue',
            'rd': 'road', 'road': 'road',
            'ln': 'lane', 'lane': 'lane',
            'dr': 'drive', 'drive': 'drive',
            'blvd': 'boulevard', 'boulevard': 'boulevard'
        }
        return type_map.get(street_type.lower().rstrip('.'), street_type.lower())

    def _is_in_expected_section(self, position: int, pattern: Pattern, text: str) -> bool:
        """Check if position is in the expected section."""
        if not pattern.metadata.get("expected_section"):
            return True

        section_info = self._analyze_sections(text, {"section_markers": {}}, position)
        return pattern.metadata["expected_section"] in [
            section_info.get("section_type", ""),
            *section_info.get("hierarchy", [])
        ]

    def _calculate_ngram_similarity(self, text1: str, text2: str, ngram_size: int = 3) -> float:
        """Calculate n-gram similarity between two strings with address-specific handling."""
        if not text1 or not text2:
            return 0.0

        # Special handling for addresses
        def split_address(addr: str) -> tuple:
            parts = addr.lower().split()
            if len(parts) >= 2:  # Allow for more flexible address formats
                # Try to find the number component
                number = next((p for p in parts if p.isdigit()), None)
                if number:
                    number_idx = parts.index(number)
                    # Get remaining parts
                    remaining = parts[:number_idx] + parts[number_idx + 1:]
                    # Try to identify street type
                    street_type = next((p for p in remaining if p.lower() in {
                        'st', 'street', 'ave', 'avenue', 'rd', 'road', 'ln', 'lane',
                        'dr', 'drive', 'blvd', 'boulevard'
                    }), None)
                    if street_type:
                        type_idx = remaining.index(street_type)
                        street_name = ' '.join(remaining[:type_idx] + remaining[type_idx + 1:])
                    else:
                        street_name = ' '.join(remaining)
                    return number, street_name, street_type
            return None

        # If both look like addresses, do component-wise comparison
        addr1 = split_address(text1)
        addr2 = split_address(text2)

        if addr1 and addr2:
            number1, name1, type1 = addr1
            number2, name2, type2 = addr2

            # Calculate component similarities
            number_sim = 1.0 if number1 == number2 else 0.0

            # Get n-grams for street names
            name_ngrams1 = set(self._get_ngrams(name1, ngram_size))
            name_ngrams2 = set(self._get_ngrams(name2, ngram_size))
            name_sim = len(name_ngrams1.intersection(name_ngrams2)) / max(len(name_ngrams1), len(name_ngrams2)) if name_ngrams1 and name_ngrams2 else 0.0

            # Handle common street type abbreviations
            type_map = {
                'st': 'street', 'str': 'street', 'street': 'street',
                'ave': 'avenue', 'av': 'avenue', 'avenue': 'avenue',
                'rd': 'road', 'road': 'road',
                'ln': 'lane', 'lane': 'lane',
                'dr': 'drive', 'drive': 'drive',
                'blvd': 'boulevard', 'boulevard': 'boulevard'
            }

            # More strict type matching
            type_sim = 1.0 if (type1 and type2 and type_map.get(type1.lower(), type1.lower()) == type_map.get(type2.lower(), type2.lower())) else 0.0

            # More strict weighting
            final_sim = (
                (number_sim * 0.5) +  # Increased weight for number match
                (name_sim * 0.3) +    # Reduced weight for name similarity
                (type_sim * 0.2)      # Maintained weight for type match
            )

            # Apply threshold for stricter matching
            return final_sim if final_sim >= 0.6 else 0.0

        # Fall back to regular n-gram similarity for non-address strings
        ngrams1 = set(self._get_ngrams(text1.lower(), ngram_size))
        ngrams2 = set(self._get_ngrams(text2.lower(), ngram_size))

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1.intersection(ngrams2)
        similarity = len(intersection) / max(len(ngrams1), len(ngrams2))

        # Apply threshold for non-address strings too
        return similarity if similarity >= 0.7 else 0.0

    def _get_ngrams(self, text: str, n: int) -> list:
        """Get n-grams from text."""
        if not text or len(text) < n:
            return []
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def _calculate_phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity using Soundex and Metaphone."""
        # Split into words
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        # Calculate similarity for each word pair
        total_score = 0
        comparisons = 0

        for w1 in words1:
            best_word_score = 0
            for w2 in words2:
                # Get phonetic codes
                soundex1, soundex2 = soundex(w1), soundex(w2)
                metaphone1, metaphone2 = metaphone(w1), metaphone(w2)

                # Calculate similarity scores
                soundex_match = soundex1 == soundex2
                metaphone_match = metaphone1 == metaphone2

                # Add sequence similarity for partial matches
                seq_sim = SequenceMatcher(None, w1, w2).ratio()

                # Combine scores with weights
                score = (
                    (soundex_match * 0.35) +
                    (metaphone_match * 0.35) +
                    (seq_sim * 0.3)
                )

                # Boost score for exact matches
                if w1 == w2:
                    score = min(1.0, score * 1.2)

                best_word_score = max(best_word_score, score)

            total_score += best_word_score
            comparisons += 1

        return total_score / comparisons if comparisons > 0 else 0.0

    def _calculate_name_confidence(
        self,
        seq_ratio: float,
        lev_ratio: float,
        ngram_ratio: float,
        phonetic_ratio: float,
        text: str = None,
        pattern_text: str = None
    ) -> float:
        # Base confidence calculation with adjusted weights
        confidence = (
            seq_ratio * 0.25 +      # Reduced from 0.3
            lev_ratio * 0.15 +      # Reduced from 0.2
            ngram_ratio * 0.15 +    # Reduced from 0.2
            phonetic_ratio * 0.45    # Increased from 0.3
        )

        # Special handling for title + last name matches
        if text and pattern_text:
            # Get the last words (likely last names)
            text_parts = text.strip().split()
            pattern_parts = pattern_text.strip().split()

            if text_parts and pattern_parts:
                text_last = text_parts[-1]
                pattern_last = pattern_parts[-1]

                # If last names match exactly, give a significant boost
                if text_last == pattern_last:
                    confidence = confidence * 1.4  # 40% boost for exact last name match

                # If we have a title and only one other word (likely "Professor Smith" vs "Dr. John Smith")
                if len(text_parts) == 2 and any(title in text_parts[0] for title in ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Professor', 'Doctor']):
                    confidence = confidence * 1.3  # Additional 30% boost for title + surname format

        # Apply high metric boost if any metric is very high
        max_metric = max(seq_ratio, lev_ratio, ngram_ratio, phonetic_ratio)
        if max_metric > 0.9:
            confidence = confidence * 1.2

        # Apply penalty if all metrics are low
        if max(seq_ratio, lev_ratio, ngram_ratio, phonetic_ratio) < 0.3:
            confidence = confidence * 0.8

        return min(1.0, confidence)  # Cap at 1.0

    def _calculate_text_confidence(
        self,
        seq_ratio: float,
        lev_ratio: float,
        ngram_ratio: float,
        text: str = None,
        example: str = None
    ) -> float:
        """Calculate confidence for general text matching."""
        weights = {
            "sequence": 0.4,
            "levenshtein": 0.3,
            "ngram": 0.3
        }

        # Calculate base score
        score = (
            seq_ratio * weights["sequence"] +
            lev_ratio * weights["levenshtein"] +
            ngram_ratio * weights["ngram"]
        )

        # Check for prefix matches if text and example are provided
        if text and example:
            text = text.lower()
            example = example.lower()

            # Boost score for prefix matches
            if text.startswith(example) or example.startswith(text):
                score = min(1.0, score * 1.3)  # 30% boost for prefix matches

            # Additional boost for near-exact prefix
            if len(text) >= len(example) and text[:len(example)] == example:
                score = min(1.0, score * 1.2)  # Additional 20% boost

        # Boost score for very close matches
        if seq_ratio > 0.8 or lev_ratio > 0.8:
            score = min(1.0, score * 1.2)

        # Less severe penalty for partial matches
        if max(seq_ratio, lev_ratio, ngram_ratio) < 0.4:
            score *= 0.7

        return score

    def _find_context_matches(
        self,
        text: str,
        pattern: Pattern,
        context: Dict[str, Any]
    ) -> List[PatternMatch]:
        """Find matches using contextual information."""
        logger.info(f"\n=== Starting Context Matching ===")
        logger.info(f"Pattern: {pattern.id}")
        logger.info(f"Pattern metadata: {pattern.metadata}")

        matches = []

        # First find base matches using regex
        base_matches = self._find_regex_matches(text, pattern)
        logger.info(f"Found {len(base_matches)} base matches")

        for match in base_matches:
            logger.info(f"\nProcessing match: {match.text}")
            logger.info(f"Match position: {match.start} to {match.end}")

            # Get section information for the match position
            section_info = self._analyze_sections(text, context, match.start)
            logger.info(f"Section info for match: {section_info}")

            # Evaluate different context aspects
            section_score = self._evaluate_section_context(match, section_info)
            proximity_score = self._evaluate_proximity(match, context, text)
            semantic_score = self._evaluate_semantic_context(match, context)

            logger.info(f"Context scores - Section: {section_score}, Proximity: {proximity_score}, Semantic: {semantic_score}")

            # Calculate context-based confidence
            context_confidence = self._calculate_context_confidence(
                section_score,
                proximity_score,
                semantic_score,
                pattern
            )
            logger.info(f"Final context confidence: {context_confidence}")

            # Update match with context information
            match.match_type = MatchType.CONTEXTUAL
            match.confidence.factors.update({
                "section": section_score,
                "proximity": proximity_score,
                "semantic": semantic_score,
                "context": context_confidence
            })
            match.confidence.value = (
                match.confidence.value * 0.6 +  # Base confidence
                context_confidence * 0.4        # Context confidence
            )

            # Determine the most appropriate section type
            section_type = section_info.get("section_type", "")
            hierarchy = section_info.get("hierarchy", [])

            # If the expected section exists in the hierarchy, use it
            if pattern.metadata.get("expected_section") in hierarchy:
                section_type = pattern.metadata["expected_section"]

            # Add section context
            match.context.update({
                "section": section_info.get("current_section", ""),
                "section_type": section_type,
                "hierarchy_level": section_info.get("level", 0)
            })

            logger.info(f"Updated match confidence: {match.confidence.value}")
            logger.info(f"Updated match context: {match.context}")

            if match.confidence.value >= self.min_confidence:
                logger.info("Match accepted - above confidence threshold")
                matches.append(match)
            else:
                logger.info("Match rejected - below confidence threshold")

        return matches

    def _analyze_sections(self, text: str, context: Dict[str, Any], position: int = 0) -> Dict[str, Any]:
        """Analyze document sections and hierarchy for a given position."""
        section_info = {
            "current_section": "",
            "section_type": "",
            "level": 0,
            "hierarchy": []
        }

        if "section_markers" not in context:
            return section_info

        # Convert markers to list and sort by position
        markers = []
        for marker_pattern, info in context["section_markers"].items():
            # Convert raw string pattern to normal string
            marker = marker_pattern.replace('\\n', '\n')

            # Find all occurrences of the marker
            pos = text.find(marker)
            if pos >= 0:
                marker_info = {
                    "pos": pos,
                    "marker": marker,
                    "type": info["type"],
                    "level": info["level"],
                    "end": pos + len(marker)
                }
                markers.append(marker_info)

        # Sort markers by position and level (higher level takes precedence)
        markers.sort(key=lambda x: (x["pos"], -x["level"]))

        # Build section hierarchy
        active_sections = []  # Track active sections at each level
        for marker in markers:
            if marker["pos"] <= position:
                # Remove any sections at same or higher level
                active_sections = [s for s in active_sections if s["level"] < marker["level"]]
                active_sections.append(marker)

        if active_sections:
            # Get the most specific (highest level) section
            current = active_sections[-1]
            section_info.update({
                "current_section": current["marker"].strip(),
                "section_type": current["type"],
                "level": current["level"]
            })
            # Build hierarchy from general to specific
            section_info["hierarchy"] = [s["type"] for s in active_sections]

        return section_info

    def _evaluate_section_context(self, match: PatternMatch, section_info: Dict[str, Any]) -> float:
        """Evaluate how well the match fits in the current section context."""
        if not match.pattern.metadata.get("expected_section"):
            return 0.5  # Neutral score if no section preference

        expected_section = match.pattern.metadata["expected_section"]
        current_section = section_info.get("section_type", "")
        hierarchy = section_info.get("hierarchy", [])

        # Direct match with current section
        if current_section == expected_section:
            return 1.0

        # Check if expected section is in hierarchy
        if expected_section in hierarchy:
            depth = len(hierarchy) - hierarchy.index(expected_section)
            score = max(0.4, 1.0 - (depth * 0.2))
            return score

        # Check for parent-child relationship
        if any(expected_section.startswith(s + '_') or s.startswith(expected_section + '_') for s in [current_section, *hierarchy]):
            return 0.8

        # Partial match
        if expected_section in current_section or current_section in expected_section:
            return 0.6

        # No match but at least we're in a section
        if current_section:
            return 0.3

        return 0.1

    def _evaluate_proximity(
        self,
        match: PatternMatch,
        context: Dict[str, Any],
        text: str
    ) -> float:
        """Evaluate proximity to relevant features."""
        if "nearby_features" not in context:
            return 0.5

        total_score = 0
        total_weight = 0

        for feature in context["nearby_features"]:
            feature_text = feature.get("text", "")
            importance = feature.get("importance", 1.0)

            # Find closest occurrence
            feature_pos = text.find(feature_text)
            if feature_pos >= 0:
                # Calculate distance score
                distance = min(
                    abs(feature_pos - match.start),
                    abs(feature_pos - match.end)
                )
                proximity_score = max(0.0, 1.0 - (distance / 1000))  # Decay over 1000 chars

                total_score += proximity_score * importance
                total_weight += importance

        return total_score / total_weight if total_weight > 0 else 0.5

    def _evaluate_semantic_context(
        self,
        match: PatternMatch,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate semantic context relevance."""
        if "semantic_info" not in context:
            return 0.5

        expected_concepts = match.pattern.metadata.get("semantic_concepts", [])
        if not expected_concepts:
            return 0.5

        context_concepts = context["semantic_info"].get("concepts", [])

        # Calculate concept overlap
        matches = sum(1 for concept in expected_concepts if concept in context_concepts)
        if not matches:
            return 0.3

        return 0.3 + (0.7 * (matches / len(expected_concepts)))

    def _calculate_context_confidence(
        self,
        section_score: float,
        proximity_score: float,
        semantic_score: float,
        pattern: Pattern
    ) -> float:
        """Calculate overall context-based confidence."""
        # Get weights from pattern metadata or use defaults
        weights = pattern.metadata.get("context_weights", {
            "section": 0.4,
            "proximity": 0.3,
            "semantic": 0.3
        })

        # Adjust section weight based on score and section importance
        if section_score > 0.8:
            weights["section"] *= 1.5  # Significant boost for strong section matches
        elif section_score < 0.4:
            weights["section"] *= 0.6  # Stronger penalty for wrong sections

        # Boost proximity score for matches near important features
        if proximity_score > 0.8:
            proximity_score = min(1.0, proximity_score * 1.2)

        # Adjust semantic score based on concept overlap
        if semantic_score > 0.7:
            semantic_score = min(1.0, semantic_score * 1.15)

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        # Calculate weighted score
        confidence = (
            section_score * weights["section"] +
            proximity_score * weights["proximity"] +
            semantic_score * weights["semantic"]
        )

        # Apply final adjustments
        if section_score > 0.9 and proximity_score > 0.8:
            confidence = min(1.0, confidence * 1.2)  # Boost for high-quality matches
        elif section_score < 0.3:
            confidence *= 0.7  # Stronger penalty for very wrong sections

        return confidence

    def _find_learned_matches(self, text: str, pattern: Pattern) -> List[PatternMatch]:
        """Find matches using learned patterns."""
        # This will be implemented as part of the learning system
        # For now, treat as regex patterns
        return self._find_regex_matches(text, pattern)

    def _get_text_chunks(self, text: str) -> Set[str]:
        """Split text into potential chunks for matching."""
        chunks = set()

        # Handle empty or whitespace text
        if not text or not text.strip():
            return chunks

        # Split by common separators
        for separator in ['\n', '.', ',', ';', ':', '-', ' ']:
            parts = [p.strip() for p in text.split(separator)]
            chunks.update(p for p in parts if p)

        # Add word combinations (up to 3 words)
        words = text.split()
        for i in range(len(words)):
            for j in range(1, 4):
                if i + j <= len(words):
                    chunk = ' '.join(words[i:i+j])
                    chunks.add(chunk)

        # Add full text if it's a single word or short phrase
        if len(text.split()) <= 3:
            chunks.add(text.strip())

        return chunks

    def _resolve_overlapping_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Resolve overlapping matches by keeping the highest confidence match."""
        if not matches:
            return matches

        # Sort by start position and confidence
        matches.sort(key=lambda m: (m.start, -m.confidence.value))

        resolved = [matches[0]]
        for match in matches[1:]:
            prev_match = resolved[-1]

            # Check for overlap
            if match.start >= prev_match.end:
                resolved.append(match)
            elif match.confidence.value > prev_match.confidence.value:
                # Replace previous match if current has higher confidence
                resolved[-1] = match

        return resolved

    def _find_exact_matches(self, text: str, pattern: Pattern) -> List[PatternMatch]:
        """Find exact matches in text using regex patterns."""
        matches = []
        if not pattern.regex_pattern:
            return matches

        try:
            # Add word boundary checks if not already present
            regex_pattern = pattern.regex_pattern
            if not regex_pattern.startswith(r'\b'):
                regex_pattern = r'\b' + regex_pattern
            if not regex_pattern.endswith(r'\b'):
                regex_pattern = regex_pattern + r'\b'

            # Handle special characters in text
            processed_text = text.strip()
            if not processed_text:
                return matches

            # Find all matches with their positions
            for match in re.finditer(regex_pattern, processed_text, re.IGNORECASE):
                start, end = match.span()
                matched_text = match.group()

                # Skip matches that are just whitespace or too short
                if not matched_text.strip() or len(matched_text) < pattern.metadata.get("min_length", 2):
                    continue

                # Calculate confidence based on match quality
                confidence = self._calculate_exact_match_confidence(
                    matched_text,
                    pattern,
                    start,
                    end,
                    processed_text
                )

                # Create match object
                pattern_match = PatternMatch(
                    pattern_id=pattern.id,
                    text=matched_text,
                    start_pos=start,
                    end_pos=end,
                    match_type="exact",
                    confidence=confidence
                )
                matches.append(pattern_match)

        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern.regex_pattern}': {str(e)}")
            return matches

        return matches

    def _calculate_exact_match_confidence(
        self,
        matched_text: str,
        pattern: Pattern,
        start: int,
        end: int,
        full_text: str
    ) -> float:
        """Calculate confidence for exact matches."""
        base_confidence = 0.9  # Start with high confidence for exact matches

        # Adjust based on match length
        min_length = pattern.metadata.get("min_length", 2)
        if len(matched_text) < min_length * 1.5:
            base_confidence *= 0.9
        elif len(matched_text) > min_length * 3:
            base_confidence = min(1.0, base_confidence * 1.1)

        # Check for word boundaries
        if start > 0 and full_text[start-1].isalnum():
            base_confidence *= 0.8
        if end < len(full_text) and full_text[end].isalnum():
            base_confidence *= 0.8

        # Boost confidence for matches that exactly match expected format
        if pattern.metadata.get("expected_format"):
            if re.match(pattern.metadata["expected_format"], matched_text):
                base_confidence = min(1.0, base_confidence * 1.2)

        return base_confidence