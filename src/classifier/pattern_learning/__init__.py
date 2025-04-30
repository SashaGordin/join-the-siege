"""
Pattern Learning Module for intelligent document classification.

This module provides pattern learning, recognition, and management capabilities:
- Pattern storage and versioning
- Pattern matching and scoring
- Pattern learning from examples
- Pattern optimization and validation
"""

from .pattern_store import PatternStore
from .pattern_matcher import PatternMatcher
from .models import Pattern, PatternMatch, PatternType, MatchType, ConfidenceScore
from .db_models import Pattern as DBPattern

__all__ = [
    'PatternStore',
    'PatternMatcher',
    'Pattern',
    'PatternMatch',
    'PatternType',
    'MatchType',
    'ConfidenceScore',
    'DBPattern'
]