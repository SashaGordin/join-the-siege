import pytest
import time
from pathlib import Path
from src.app import app
from src.classifier.config.config_manager import IndustryConfigManager
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def test_end_to_end_classification(tmp_path):
    client = app.test_client()
    # 1. Upload a PDF and classify
    pdf_path = tmp_path / "test.pdf"
    # Generate a real PDF with unique content using reportlab
    unique_text = f"Test Invoice {time.time()}"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, unique_text)
    c.drawString(100, 735, "Amount: $100.00")
    c.save()
    with open(pdf_path, "rb") as f:
        response = client.post("/classify_file", data={"file": (f, "test.pdf")})
    assert response.status_code == 200
    data = response.get_json()
    assert "classification" in data
    assert "document_type" in data["classification"]
    assert "confidence" in data["classification"]

    # 2. Test async status endpoint if available
    if "task_id" in data:
        task_id = data["task_id"]
        # Poll for completion
        for _ in range(10):
            status_resp = client.get(f"/status/{task_id}")
            status = status_resp.get_json().get("status")
            if status == "completed":
                break
            time.sleep(1)
        assert status == "completed"
        # Optionally, check result structure
        result = status_resp.get_json().get("result")
        assert result is not None
        assert "classification" in result
        assert "features" in result["classification"]

def test_add_new_industry_and_classify(tmp_path):
    # 1. Add a new industry config
    config_manager = IndustryConfigManager()
    new_config = {
        "industry_name": "test_industry",
        "version": "1.0.0",
        "description": "Test industry",
        "features": {
            "shared": ["date"],
            "specific": {
                "test_feature": {
                    "type": "test",
                    "importance": "required",
                    "patterns": ["Test\\s+Feature"],
                    "validation": {
                        "format": "text",
                        "rules": [{"type": "regex", "value": "^Test Feature$"}]
                    },
                    "description": "A test feature"
                }
            },
            "validation_rules": {"required_features": ["test_feature"]},
            "confidence_thresholds": {"minimum": 0.5, "high": 0.8}
        }
    }
    config_manager.update_industry_config("test_industry", new_config)
    # 2. Classify a document for the new industry
    client = app.test_client()
    response = client.post("/process", json={
        "content": "Test Feature",
        "metadata": {"industry": "test_industry"}
    })
    assert response.status_code == 200
    # Clean up: remove the test industry config if needed

def test_health_and_industries_endpoints():
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    industries = client.get("/industries")
    assert industries.status_code == 200
    assert isinstance(industries.get_json(), dict)