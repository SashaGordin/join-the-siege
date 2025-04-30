from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import uuid

from src.classifier import ContentClassifier
from src.utils.file_validator import FileValidator
from src.utils.exceptions import (
    FileValidationError,
    InvalidFileTypeError,
    FileTooLargeError,
    FileAccessError
)
from src.classifier.exceptions import (
    TextExtractionError,
    ClassificationError
)
from src.classifier.config.config_manager import IndustryConfigManager
from src.classifier.tasks import process_document
from src.classifier.services.cache_service import CacheService

app = Flask(__name__)

# Initialize our file validator, classifier, and config manager
file_validator = FileValidator()
content_classifier = ContentClassifier()
config_manager = IndustryConfigManager()
cache_service = CacheService()

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location while preserving its extension.

    Args:
        uploaded_file: The uploaded file from request.files

    Returns:
        Path: Path to the saved temporary file

    Raises:
        Exception: If file cannot be saved
    """
    # Get original filename and extension
    original_filename = secure_filename(uploaded_file.filename)
    _, ext = os.path.splitext(original_filename)

    # Create temporary file with the correct extension
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f"upload{ext}"

    # Save the file
    uploaded_file.save(temp_path)
    return temp_path

@app.route('/classify_file', methods=['POST'])
def classify_file_route():
    """
    Endpoint to classify a file based on its content.
    Expects a file in the request.files with key 'file'.
    Optionally accepts an 'industry' parameter to specify the industry context.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get industry parameter if provided
    industry = request.form.get('industry')
    if industry:
        try:
            # Verify industry exists
            config_manager.load_industry_config(industry)
        except ValueError as e:
            available_industries = config_manager.list_available_industries()
            return jsonify({
                "error": f"Invalid industry: {industry}. Valid options are: {', '.join(available_industries)}"
            }), 400

    temp_path = None
    try:
        # Save uploaded file with correct extension
        temp_path = save_uploaded_file(file)

        # Validate the file
        try:
            file_validator.is_valid_file_type(temp_path)
        except InvalidFileTypeError as e:
            return jsonify({"error": str(e)}), 400
        except FileTooLargeError as e:
            return jsonify({"error": str(e)}), 413
        except FileAccessError as e:
            return jsonify({"error": str(e)}), 500
        except FileValidationError as e:
            return jsonify({"error": str(e)}), 400

        # Get file type info
        mime_type = file_validator.get_file_type(temp_path)

        # For images, we want to pass through validation but fail at classification
        if mime_type.startswith('image/'):
            return jsonify({"error": "OCR not implemented yet"}), 501

        # Classify the file
        try:
            classification_result = content_classifier.classify_file(temp_path, industry=industry)

            # Generate preview if possible
            try:
                preview_data = file_validator.get_file_preview(temp_path)
            except Exception:
                preview_data = {"preview_available": False}

            # Add industry-specific information if applicable
            response_data = {
                "classification": {
                    "document_type": classification_result["class"],
                    "confidence": classification_result["confidence"],
                    "features": classification_result["features"],
                    "pattern_features": classification_result.get("pattern_features", [])
                },
                "file_info": {
                    "mime_type": mime_type,
                    "filename": file.filename
                },
                "preview": preview_data
            }

            if industry:
                # Add industry-specific validation info
                validation_warnings = [
                    f for f in classification_result["features"]
                    if f["type"] == "validation_warning"
                ]
                if validation_warnings:
                    response_data["validation"] = {
                        "warnings": [w["values"] for w in validation_warnings],
                        "industry": industry
                    }

            return jsonify(response_data), 200

        except TextExtractionError as e:
            return jsonify({"error": f"Text extraction failed: {str(e)}"}), 422
        except ClassificationError as e:
            return jsonify({"error": f"Classification failed: {str(e)}"}), 422
        except NotImplementedError as e:
            return jsonify({"error": str(e)}), 501

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    finally:
        # Clean up temporary file and directory
        if temp_path:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_path.parent)
            except Exception:
                pass  # Best effort cleanup

@app.route('/industries', methods=['GET'])
def list_industries():
    """List available industry configurations."""
    try:
        industries = {}
        available_industries = config_manager.list_available_industries()

        for industry_id in available_industries:
            config = config_manager.load_industry_config(industry_id)
            industries[industry_id] = {
                "name": config["industry_name"],
                "description": config["description"],
                "features": {
                    "shared": list(config["features"]["shared"].keys()),
                    "specific": {
                        name: feature["description"]
                        for name, feature in config["features"]["specific"].items()
                    }
                },
                "validation_rules": config["features"]["validation_rules"],
                "confidence_thresholds": config["features"]["confidence_thresholds"]
            }
        return jsonify(industries), 200
    except Exception as e:
        return jsonify({"error": f"Failed to list industries: {str(e)}"}), 500

@app.route('/preview_file', methods=['POST'])
def preview_file_route():
    """
    Endpoint to get a preview of a file.
    Expects a file in the request.files with key 'file'.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = None
    try:
        # Save uploaded file with correct extension
        temp_path = save_uploaded_file(file)

        # Validate the file
        try:
            file_validator.is_valid_file_type(temp_path)
        except FileValidationError as e:
            return jsonify({"error": str(e)}), 400

        # Generate preview
        try:
            preview_data = file_validator.get_file_preview(temp_path)
            return jsonify(preview_data), 200
        except Exception as e:
            return jsonify({"error": f"Could not generate preview: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    finally:
        # Clean up temporary file and directory
        if temp_path:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_path.parent)
            except Exception:
                pass  # Best effort cleanup

@app.route('/process', methods=['POST'])
def process():
    """Process document asynchronously."""
    try:
        # Handle JSON parsing errors
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        try:
            data = request.get_json()
        except Exception:
            return jsonify({'error': 'Invalid JSON format'}), 400

        if not data:
            return jsonify({'error': 'Empty request body'}), 400

        if 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400

        if not data['content']:
            return jsonify({'error': 'Content cannot be empty'}), 400

        document_id = data.get('document_id', str(uuid.uuid4()))
        content = data['content']
        metadata = data.get('metadata', {})

        # Check cache first
        try:
            cache_key = f"document_processing:{document_id}"
            cached_result = cache_service.get(cache_key)
            if cached_result:
                return jsonify({
                    'status': 'completed',
                    'result': cached_result,
                    'cached': True
                })
        except ConnectionError:
            # Log the error but continue processing
            app.logger.warning("Redis connection failed, continuing without cache")
        except Exception as e:
            # Log other cache errors but continue
            app.logger.warning(f"Cache error: {str(e)}, continuing without cache")

        # Submit async task
        task = process_document.delay(document_id, content, metadata)

        return jsonify({
            'status': 'processing',
            'task_id': task.id,
            'document_id': document_id
        })

    except Exception as e:
        app.logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get task status."""
    try:
        task = process_document.AsyncResult(task_id)

        if task.ready():
            if task.successful():
                return jsonify({
                    'status': 'completed',
                    'result': task.get()
                })
            elif task.failed():
                return jsonify({
                    'status': 'failed',
                    'error': str(task.result)
                }), 500
            else:
                # Task result expired
                return jsonify({
                    'status': 'expired',
                    'error': 'Task result has expired'
                }), 404
        else:
            return jsonify({
                'status': 'processing',
                'task_id': task_id
            })

    except Exception as e:
        app.logger.error(f"Error checking task status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
