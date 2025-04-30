import pytest
from pathlib import Path
import json
import jsonschema
from src.classifier.config.config_manager import IndustryConfigManager

@pytest.fixture
def config_manager(tmp_path):
    """Create a config manager with temporary directory."""
    # Create necessary subdirectories
    industries_dir = tmp_path / "industries"
    schemas_dir = tmp_path / "schemas"
    industries_dir.mkdir()
    schemas_dir.mkdir()

    # Copy schema files
    src_dir = Path(__file__).parent.parent / "src" / "classifier" / "config"
    with open(src_dir / "schemas" / "industry_schema.json") as f:
        industry_schema = json.load(f)
    with open(src_dir / "schemas" / "feature_schema.json") as f:
        feature_schema = json.load(f)
    with open(src_dir / "base_features.json") as f:
        base_features = json.load(f)

    # Save to temp directory
    with open(schemas_dir / "industry_schema.json", "w") as f:
        json.dump(industry_schema, f)
    with open(schemas_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f)
    with open(tmp_path / "base_features.json", "w") as f:
        json.dump(base_features, f)

    return IndustryConfigManager(config_dir=tmp_path)

@pytest.fixture
def sample_industry_config():
    """Sample valid industry configuration."""
    return {
        "industry_name": "test_industry",
        "version": "1.0.0",
        "description": "Test industry configuration",
        "features": {
            "shared": ["date", "amount"],
            "specific": {
                "test_feature": {
                    "type": "test",
                    "importance": "required",
                    "patterns": ["test\\s+pattern"],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^test.*$"
                            }
                        ]
                    },
                    "description": "Test feature"
                }
            },
            "validation_rules": {
                "required_features": ["test_feature"]
            },
            "confidence_thresholds": {
                "minimum": 0.6,
                "high": 0.8
            }
        }
    }

def test_load_schemas(config_manager):
    """Test that schemas are loaded correctly."""
    assert config_manager.industry_schema is not None
    assert config_manager.feature_schema is not None
    assert isinstance(config_manager.industry_schema, dict)
    assert isinstance(config_manager.feature_schema, dict)

def test_validate_industry_config(config_manager, sample_industry_config):
    """Test industry configuration validation."""
    # Test valid config
    assert config_manager.validate_config(sample_industry_config, "industry") is True

    # Test invalid config
    invalid_config = sample_industry_config.copy()
    del invalid_config["version"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        config_manager.validate_config(invalid_config, "industry")

def test_load_industry_config(config_manager, sample_industry_config):
    """Test loading industry configuration."""
    # Save test config
    config_path = config_manager.industries_dir / "test_industry.json"
    with open(config_path, "w") as f:
        json.dump(sample_industry_config, f)

    # Load and verify
    loaded_config = config_manager.load_industry_config("test_industry")
    assert loaded_config["industry_name"] == "test_industry"
    assert loaded_config["version"] == "1.0.0"
    assert "date" in loaded_config["features"]["shared"]
    assert "amount" in loaded_config["features"]["shared"]

def test_merge_shared_features(config_manager, sample_industry_config):
    """Test merging of shared features."""
    # Save test config
    config_path = config_manager.industries_dir / "test_industry.json"
    with open(config_path, "w") as f:
        json.dump(sample_industry_config, f)

    # Load and verify shared features
    config = config_manager.load_industry_config("test_industry")
    shared_features = config["features"]["shared"]

    assert "date" in shared_features
    assert "amount" in shared_features
    assert isinstance(shared_features["date"], dict)
    assert isinstance(shared_features["amount"], dict)
    assert "patterns" in shared_features["date"]
    assert "patterns" in shared_features["amount"]

def test_nonexistent_industry(config_manager):
    """Test handling of non-existent industry."""
    with pytest.raises(ValueError, match="Industry configuration not found"):
        config_manager.load_industry_config("nonexistent")

def test_update_industry_config(config_manager, sample_industry_config):
    """Test updating industry configuration."""
    # Initial save
    config_manager.update_industry_config("test_industry", sample_industry_config)

    # Modify and update
    updated_config = sample_industry_config.copy()
    updated_config["version"] = "1.0.1"
    config_manager.update_industry_config("test_industry", updated_config)

    # Verify update
    loaded_config = config_manager.load_industry_config("test_industry")
    assert loaded_config["version"] == "1.0.1"

def test_list_available_industries(config_manager, sample_industry_config):
    """Test listing available industries."""
    # Save test configs
    config_manager.update_industry_config("test_industry1", sample_industry_config)
    config_manager.update_industry_config("test_industry2", sample_industry_config)

    # List and verify
    industries = config_manager.list_available_industries()
    assert "test_industry1" in industries
    assert "test_industry2" in industries

def test_invalid_shared_feature_reference(config_manager):
    """Test handling of invalid shared feature references."""
    invalid_config = {
        "industry_name": "test_industry",
        "version": "1.0.0",
        "description": "Test industry configuration",
        "features": {
            "shared": ["date", "nonexistent_feature"],  # nonexistent_feature doesn't exist
            "specific": {},
            "validation_rules": {},
            "confidence_thresholds": {
                "minimum": 0.6,
                "high": 0.8
            }
        }
    }

    # Should still load but log a warning about missing feature
    config_manager.update_industry_config("test_industry", invalid_config)
    loaded_config = config_manager.load_industry_config("test_industry")
    assert "date" in loaded_config["features"]["shared"]
    assert "nonexistent_feature" not in loaded_config["features"]["shared"]

def test_corrupted_json_handling(config_manager):
    """Test handling of corrupted JSON files."""
    # Create a corrupted JSON file
    config_path = config_manager.industries_dir / "corrupted.json"
    with open(config_path, "w") as f:
        f.write("{invalid json content")

    with pytest.raises(ValueError, match="Failed to load industry config"):
        config_manager.load_industry_config("corrupted")

def test_concurrent_config_updates(config_manager, sample_industry_config):
    """Test handling multiple updates to the same industry config."""
    # Initial save
    config_manager.update_industry_config("test_industry", sample_industry_config)

    # Update version
    updated_config1 = sample_industry_config.copy()
    updated_config1["version"] = "1.0.1"
    config_manager.update_industry_config("test_industry", updated_config1)

    # Update description
    updated_config2 = sample_industry_config.copy()
    updated_config2["version"] = "1.0.2"
    updated_config2["description"] = "Updated description"
    config_manager.update_industry_config("test_industry", updated_config2)

    # Verify final state
    final_config = config_manager.load_industry_config("test_industry")
    assert final_config["version"] == "1.0.2"
    assert final_config["description"] == "Updated description"

def test_feature_validation_rules(config_manager):
    """Test validation rules in feature configurations."""
    config = {
        "industry_name": "test_industry",
        "version": "1.0.0",
        "description": "Test industry configuration",
        "features": {
            "shared": ["date"],
            "specific": {
                "custom_date": {
                    "type": "date",
                    "importance": "required",
                    "patterns": ["\\d{4}-\\d{2}-\\d{2}"],
                    "validation": {
                        "format": "date",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^\\d{4}-\\d{2}-\\d{2}$"
                            }
                        ]
                    },
                    "description": "Custom date format"
                }
            },
            "validation_rules": {
                "required_features": ["custom_date"]
            },
            "confidence_thresholds": {
                "minimum": 0.6,
                "high": 0.8
            }
        }
    }

    # Should validate successfully
    config_manager.validate_config(config, "industry")

    # Test invalid validation rule
    invalid_config = config.copy()
    invalid_config["features"]["specific"]["custom_date"]["validation"]["rules"][0]["type"] = "invalid_type"
    with pytest.raises(jsonschema.exceptions.ValidationError):
        config_manager.validate_config(invalid_config, "industry")

def test_empty_industry_dir(config_manager):
    """Test behavior when industries directory is empty."""
    # Should return empty list
    assert len(config_manager.list_available_industries()) == 0

def test_cache_invalidation(config_manager, sample_industry_config):
    """Test that cache is properly invalidated on updates."""
    # Initial save
    config_manager.update_industry_config("test_industry", sample_industry_config)

    # Load to cache
    first_load = config_manager.load_industry_config("test_industry")

    # Update config
    updated_config = sample_industry_config.copy()
    updated_config["version"] = "1.0.1"
    config_manager.update_industry_config("test_industry", updated_config)

    # Load again, should get updated version
    second_load = config_manager.load_industry_config("test_industry")
    assert second_load["version"] == "1.0.1"

def test_shared_feature_inheritance(config_manager, sample_industry_config):
    """Test that shared features properly inherit base feature properties."""
    config_manager.update_industry_config("test_industry", sample_industry_config)
    loaded_config = config_manager.load_industry_config("test_industry")

    # Check that date feature inherits all properties from base features
    date_feature = loaded_config["features"]["shared"]["date"]
    assert "patterns" in date_feature
    assert "validation" in date_feature
    assert "description" in date_feature

def test_adding_new_industry_workflow(config_manager):
    """Test the complete workflow of adding a new industry (legal documents)."""
    # Define a new legal documents industry configuration
    legal_config = {
        "industry_name": "legal",
        "version": "1.0.0",
        "description": "Legal documents configuration",
        "features": {
            "shared": ["date", "amount", "signature"],  # Reusing common features
            "specific": {
                "case_number": {
                    "type": "identifier",
                    "importance": "required",
                    "patterns": ["Case\\s+No\\.?\\s*\\d{2}-\\d{4}", "\\d{2}-[A-Z]{2}-\\d{4}"],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^\\d{2}[-A-Z]*\\d{4}$"
                            }
                        ]
                    },
                    "description": "Legal case identifier"
                },
                "court_name": {
                    "type": "entity",
                    "importance": "required",
                    "patterns": ["IN THE .* COURT", "COURT OF .*"],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^(IN THE|COURT OF).*COURT$"
                            }
                        ]
                    },
                    "description": "Name of the court"
                },
                "party_names": {
                    "type": "entity",
                    "importance": "required",
                    "patterns": ["Plaintiff:", "Defendant:", "vs\\."],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^[A-Za-z\\s,\\.]+$"
                            }
                        ]
                    },
                    "description": "Names of involved parties"
                }
            },
            "validation_rules": {
                "required_features": ["case_number", "court_name", "party_names", "date"],
                "optional_features": ["amount", "signature"]
            },
            "confidence_thresholds": {
                "minimum": 0.7,
                "high": 0.9
            }
        }
    }

    # 1. Test adding the new industry
    config_manager.update_industry_config("legal", legal_config)

    # 2. Verify the industry was added
    available_industries = config_manager.list_available_industries()
    assert "legal" in available_industries

    # 3. Load and verify the configuration
    loaded_config = config_manager.load_industry_config("legal")
    assert loaded_config["industry_name"] == "legal"
    assert loaded_config["version"] == "1.0.0"

    # 4. Verify shared features were properly merged
    shared_features = loaded_config["features"]["shared"]
    assert isinstance(shared_features, dict)  # Should be expanded with actual feature definitions
    assert "date" in shared_features
    assert "amount" in shared_features
    assert "signature" in shared_features

    # 5. Verify specific features
    specific_features = loaded_config["features"]["specific"]
    assert "case_number" in specific_features
    assert "court_name" in specific_features
    assert "party_names" in specific_features

    # 6. Verify validation rules
    validation_rules = loaded_config["features"]["validation_rules"]
    assert "case_number" in validation_rules["required_features"]
    assert "signature" in validation_rules["optional_features"]

    # 7. Test updating an existing feature
    updated_config = loaded_config.copy()
    updated_config["features"]["specific"]["case_number"]["patterns"].append("\\d{4}-[A-Z]{4}")
    config_manager.update_industry_config("legal", updated_config)

    # 8. Verify update was successful
    final_config = config_manager.load_industry_config("legal")
    assert "\\d{4}-[A-Z]{4}" in final_config["features"]["specific"]["case_number"]["patterns"]

def test_multiple_industry_interactions(config_manager):
    """Test interactions between multiple industry configurations."""
    # Define two industries that share some features
    healthcare_config = {
        "industry_name": "healthcare",
        "version": "1.0.0",
        "description": "Healthcare documents configuration",
        "features": {
            "shared": ["date", "identifier"],
            "specific": {
                "patient_id": {
                    "type": "identifier",
                    "importance": "required",
                    "patterns": ["Patient ID:.*", "MRN:.*"],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^[A-Z0-9-]+$"
                            }
                        ]
                    },
                    "description": "Patient identifier"
                }
            },
            "validation_rules": {
                "required_features": ["patient_id", "date"],
                "optional_features": ["identifier"]
            },
            "confidence_thresholds": {
                "minimum": 0.8,
                "high": 0.95
            }
        }
    }

    legal_config = {
        "industry_name": "legal",
        "version": "1.0.0",
        "description": "Legal documents configuration",
        "features": {
            "shared": ["date", "identifier", "signature"],
            "specific": {
                "case_number": {
                    "type": "identifier",
                    "importance": "required",
                    "patterns": ["Case No:.*"],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^[A-Z0-9-]+$"
                            }
                        ]
                    },
                    "description": "Legal case identifier"
                }
            },
            "validation_rules": {
                "required_features": ["case_number", "signature"],
                "optional_features": ["date"]
            },
            "confidence_thresholds": {
                "minimum": 0.7,
                "high": 0.9
            }
        }
    }

    # Add both industries
    config_manager.update_industry_config("healthcare", healthcare_config)
    config_manager.update_industry_config("legal", legal_config)

    # Verify both were added
    industries = config_manager.list_available_industries()
    assert "healthcare" in industries
    assert "legal" in industries

    # Verify they maintain their separate configurations
    healthcare = config_manager.load_industry_config("healthcare")
    legal = config_manager.load_industry_config("legal")

    assert healthcare["features"]["validation_rules"]["required_features"] == ["patient_id", "date"]
    assert legal["features"]["validation_rules"]["required_features"] == ["case_number", "signature"]

    # Verify shared features are properly expanded in both
    assert isinstance(healthcare["features"]["shared"]["date"], dict)
    assert isinstance(legal["features"]["shared"]["date"], dict)

def test_version_management(config_manager):
    """Test version management and updates."""
    initial_config = {
        "industry_name": "test",
        "version": "1.0.0",
        "description": "Test config",
        "features": {
            "shared": ["date"],
            "specific": {},
            "validation_rules": {
                "required_features": ["date"],
                "optional_features": []
            },
            "confidence_thresholds": {
                "minimum": 0.7,
                "high": 0.9
            }
        }
    }

    # Add initial version
    config_manager.update_industry_config("test", initial_config)

    # Update with new version
    updated_config = initial_config.copy()
    updated_config["version"] = "1.1.0"
    updated_config["features"]["shared"] = ["date", "amount"]  # Replace instead of append
    config_manager.update_industry_config("test", updated_config)

    # Load and verify it's the new version
    loaded_config = config_manager.load_industry_config("test")
    assert loaded_config["version"] == "1.1.0"
    assert "amount" in loaded_config["features"]["shared"]

def test_error_recovery(config_manager):
    """Test system's ability to handle and recover from errors."""
    # Test with missing required fields
    invalid_config = {
        "industry_name": "test",
        # Missing version
        "features": {
            "shared": ["date"],
            "specific": {},
            "validation_rules": {},
            "confidence_thresholds": {}
        }
    }

    # Should raise validation error
    with pytest.raises(jsonschema.exceptions.ValidationError):
        config_manager.update_industry_config("test", invalid_config)

    # System should still be in valid state
    assert "test" not in config_manager.list_available_industries()

    # Test with invalid shared feature reference
    invalid_feature_config = {
        "industry_name": "test",
        "version": "1.0.0",
        "description": "Test config",
        "features": {
            "shared": ["nonexistent_feature"],
            "specific": {},
            "validation_rules": {},
            "confidence_thresholds": {
                "minimum": 0.7,
                "high": 0.9
            }
        }
    }

    # Should accept config but remove invalid feature
    config_manager.update_industry_config("test", invalid_feature_config)
    loaded_config = config_manager.load_industry_config("test")
    assert "nonexistent_feature" not in loaded_config["features"]["shared"]

def test_feature_inheritance_edge_cases(config_manager):
    """Test edge cases in feature inheritance and sharing."""
    config = {
        "industry_name": "test",
        "version": "1.0.0",
        "description": "Test config",
        "features": {
            "shared": ["date", "identifier"],
            "specific": {
                # Override a shared feature pattern
                "custom_date": {
                    "type": "date",
                    "importance": "required",
                    "patterns": ["Custom date pattern"],
                    "validation": {
                        "format": "text",
                        "rules": [
                            {
                                "type": "regex",
                                "value": "^custom$"
                            }
                        ]
                    },
                    "description": "Custom date format"
                }
            },
            "validation_rules": {
                "required_features": ["custom_date"],
                "optional_features": ["date", "identifier"]
            },
            "confidence_thresholds": {
                "minimum": 0.7,
                "high": 0.9
            }
        }
    }

    # Add config
    config_manager.update_industry_config("test", config)

    # Load and verify
    loaded_config = config_manager.load_industry_config("test")

    # Verify shared features exist
    assert "date" in loaded_config["features"]["shared"]
    assert "identifier" in loaded_config["features"]["shared"]

    # Verify specific feature overrides
    assert loaded_config["features"]["specific"]["custom_date"]["patterns"] == ["Custom date pattern"]

    # Verify original shared feature is unchanged
    assert "\\d{4}-\\d{2}-\\d{2}" in loaded_config["features"]["shared"]["date"]["patterns"]