import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class IndustryConfigManager:
    """Manages industry configurations and feature definitions."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files. If None, uses default.
        """
        self.config_dir = config_dir or Path(__file__).parent
        self.industries_dir = self.config_dir / "industries"
        self.schemas_dir = self.config_dir / "schemas"

        # Load schemas
        self.industry_schema = self._load_json(self.schemas_dir / "industry_schema.json")
        self.feature_schema = self._load_json(self.schemas_dir / "feature_schema.json")

        # Load base features
        self.base_features = self._load_json(self.config_dir / "base_features.json")

        # Cache for loaded industry configs
        self._industry_cache: Dict[str, Dict[str, Any]] = {}

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise ValueError(f"Failed to load {file_path}: {str(e)}")

    def validate_config(self, config: Dict[str, Any], schema_type: str = "industry") -> bool:
        """
        Validate a configuration against its schema.

        Args:
            config: Configuration dictionary to validate
            schema_type: Type of schema to validate against ("industry" or "feature")

        Returns:
            bool: True if valid

        Raises:
            jsonschema.exceptions.ValidationError: If validation fails
        """
        schema = self.industry_schema if schema_type == "industry" else self.feature_schema
        try:
            jsonschema.validate(instance=config, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def _filter_and_merge_shared_features(self, shared_feature_ids: list) -> Dict[str, Any]:
        """
        Filter out invalid shared features and merge with base features.

        Args:
            shared_feature_ids: List of shared feature IDs to process

        Returns:
            Dict of valid shared features
        """
        shared_features = {}
        for feature_id in shared_feature_ids:
            if feature_id in self.base_features['features']:
                shared_features[feature_id] = self.base_features['features'][feature_id]
            else:
                logger.warning(f"Removing invalid shared feature reference: {feature_id}")
        return shared_features

    def load_industry_config(self, industry_name: str) -> Dict[str, Any]:
        """
        Load and validate an industry configuration.

        Args:
            industry_name: Name of the industry to load

        Returns:
            Dict containing the industry configuration

        Raises:
            ValueError: If industry config is invalid or not found
        """
        # Check cache first
        if industry_name in self._industry_cache:
            return self._industry_cache[industry_name]

        config_path = self.industries_dir / f"{industry_name}.json"

        if not config_path.exists():
            raise ValueError(f"Industry configuration not found: {industry_name}")

        try:
            config = self._load_json(config_path)
            self.validate_config(config, "industry")

            # Filter and merge shared features
            config['features']['shared'] = self._filter_and_merge_shared_features(
                config['features']['shared']
            )

            # Cache the config
            self._industry_cache[industry_name] = config
            return config

        except Exception as e:
            raise ValueError(f"Failed to load industry config: {str(e)}")

    def get_industry_features(self, industry_name: str) -> Dict[str, Any]:
        """Get combined features (shared + specific) for an industry."""
        config = self.load_industry_config(industry_name)
        return {
            'shared': config['features']['shared'],
            'specific': config['features']['specific']
        }

    def list_available_industries(self) -> list:
        """List all available industry configurations."""
        return [p.stem for p in self.industries_dir.glob("*.json")]

    def update_industry_config(self, industry_name: str, new_config: Dict[str, Any]) -> None:
        """
        Update an industry configuration.

        Args:
            industry_name: Name of the industry to update
            new_config: New configuration dictionary

        Raises:
            ValueError: If new configuration is invalid
        """
        # Create a copy for validation and saving
        config_to_save = new_config.copy()

        # Ensure shared features is a list of IDs before validation
        if isinstance(config_to_save['features']['shared'], dict):
            config_to_save['features']['shared'] = list(config_to_save['features']['shared'].keys())

        # Validate new config
        self.validate_config(config_to_save, "industry")

        # Save to file
        config_path = self.industries_dir / f"{industry_name}.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)

            # Update cache with the expanded shared features
            loaded_config = config_to_save.copy()
            loaded_config['features']['shared'] = self._filter_and_merge_shared_features(
                config_to_save['features']['shared']
            )
            self._industry_cache[industry_name] = loaded_config
        except Exception as e:
            raise ValueError(f"Failed to update industry config: {str(e)}")

    def _merge_shared_features(self, feature_ids: list) -> Dict[str, Any]:
        """
        Merge requested shared features from base features.
        Only includes features that exist in base_features.

        Args:
            feature_ids: List of feature IDs to merge

        Returns:
            Dict of valid shared features
        """
        return self._filter_and_merge_shared_features(feature_ids)