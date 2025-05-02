from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import uuid
import hashlib
import logging

from src.classifier.hybrid_classifier import HybridClassifier
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize our file validator, classifier, and config manager
file_validator = FileValidator()
hybrid_classifier = HybridClassifier()
config_manager = IndustryConfigManager()
cache_service = CacheService()

# Compatibility alias for legacy code/tests
content_classifier = hybrid_classifier

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
    Classify an uploaded file.

    Returns:
        JSON response with classification results or error
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    logger.info(f"Received file: {file.filename}")
    logger.info(f"Content type: {file.content_type}")

    # Early return for image files (not implemented)
    # if file.content_type.startswith("image/"):
    #     return jsonify({"error": "Image classification/OCR not implemented"}), 501

    try:
        # Save uploaded file with correct extension
        temp_path = save_uploaded_file(file)
        logger.info(f"Saved uploaded file to: {temp_path}")

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

        # Read file content for caching
        file_content = Path(temp_path).read_bytes()
        content_hash = hashlib.sha256(file_content).hexdigest()
        logger.info(f"File content hash: {content_hash}")

        # Log Redis connection info
        redis_info = cache_service.redis_client.info()
        logger.info(f"Redis connection info:")
        logger.info(f"- Host: {cache_service.redis_client.connection_pool.connection_kwargs['host']}")
        logger.info(f"- Port: {cache_service.redis_client.connection_pool.connection_kwargs['port']}")
        logger.info(f"- DB: {cache_service.redis_client.connection_pool.connection_kwargs['db']}")
        logger.info(f"- Connected clients: {redis_info.get('connected_clients')}")
        logger.info(f"- Redis version: {redis_info.get('redis_version')}")

        # Check cache first
        logger.info("Checking cache for classification")
        cached_result = cache_service.get_classification(file_content)
        if cached_result:
            logger.info("Cache hit - returning cached result")
            return jsonify(cached_result), 200

        # Generate preview if possible
        try:
            preview_data = file_validator.get_file_preview(temp_path)
        except Exception:
            preview_data = {"preview_available": False}

        # Classify the file if not in cache
        try:
            logger.info("Cache miss - classifying file")
            logger.info("Cache miss - performing classification")
            # Extract text from file
            text = hybrid_classifier.content_classifier.extract_text(temp_path)
            # Use hybrid classifier
            raw_result = hybrid_classifier.classify_document(text)

            # Transform the result to our standard format
            classification_result = {
                "document_type": raw_result.doc_type,
                "confidence": raw_result.confidence,
                "features": raw_result.features,
                "pattern_matches": [
                    {
                        "type": m.pattern.feature_type,
                        "text": m.text,
                        "confidence": m.confidence.value,
                        "context": m.context
                    } for m in raw_result.pattern_matches
                ],
                "metadata": raw_result.metadata
            }

            # Prepare the complete response data
            response_data = {
                "classification": classification_result,
                "file_info": {
                    "mime_type": mime_type,
                    "filename": file.filename
                },
                "preview": preview_data
            }

            # Before caching
            logger.info("Preparing to cache classification result")
            logger.info(f"Current Redis keys: {cache_service.redis_client.keys('*')}")

            # Cache the complete response data
            success = cache_service.set_classification(file_content, response_data)
            if not success:
                logger.error("Failed to cache classification result")
            else:
                logger.info("Successfully cached classification result")
                # Verify the cache
                verification = cache_service.get_classification(file_content)
                if verification:
                    logger.info("Cache verification successful")
                    logger.info(f"Redis keys after caching: {cache_service.redis_client.keys('*')}")
                else:
                    logger.error("Cache verification failed")
                    logger.error(f"Redis keys after failed verification: {cache_service.redis_client.keys('*')}")

            return jsonify(response_data), 200

        except NotImplementedError as e:
            return jsonify({"error": str(e)}), 501
        except TextExtractionError as e:
            return jsonify({"error": f"Text extraction failed: {str(e)}"}), 422
        except ClassificationError as e:
            return jsonify({"error": f"Classification failed: {str(e)}"}), 422

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
            logger.error(f"Could not generate preview: {str(e)}")
            return jsonify({"error": f"Could not generate preview: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
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

        # Validate industry if provided
        if 'industry' in metadata:
            industry = metadata['industry']
            if industry not in config_manager.list_available_industries():
                return jsonify({
                    'error': f'Invalid industry: {industry}. Available industries: '
                            f'{", ".join(config_manager.list_available_industries())}'
                }), 400

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
                result = task.get()
                return jsonify({
                    'status': 'completed',
                    'result': {
                        'document_id': result['document_id'],
                        'doc_type': result['doc_type'],
                        'confidence': result['confidence'],
                        'features': result['features'],
                        'pattern_matches': result['pattern_matches'],
                        'metadata': result['metadata']
                    }
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Kubernetes probes."""
    try:
        # Check if we can connect to Redis
        cache_service = CacheService()
        cache_service.get("health_check")

        # Check if we can initialize the classifier
        classifier = HybridClassifier()

        return jsonify({
            'status': 'healthy',
            'redis': 'connected',
            'classifier': 'initialized'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
