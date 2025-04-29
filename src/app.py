from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path

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

app = Flask(__name__)

# Initialize our file validator and classifier
file_validator = FileValidator()
content_classifier = ContentClassifier()

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
            classification_result = content_classifier.classify_file(temp_path)

            # Generate preview if possible
            try:
                preview_data = file_validator.get_file_preview(temp_path)
            except Exception:
                preview_data = {"preview_available": False}

            return jsonify({
                "classification": {
                    "document_type": classification_result["class"],
                    "confidence": classification_result["confidence"],
                    "features": classification_result["features"]
                },
                "file_info": {
                    "mime_type": mime_type,
                    "filename": file.filename
                },
                "preview": preview_data
            }), 200

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

if __name__ == '__main__':
    app.run(debug=True)
