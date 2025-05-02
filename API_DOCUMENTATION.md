# API Documentation

This API powers a document classification service that uses AI to categorize and extract features from uploaded files (PDF, text, etc.). It supports synchronous and asynchronous processing, industry-specific configurations, and integrates with Redis and Celery for caching and background tasks.

---

## Classifier Architecture: ContentClassifier & HybridClassifier

### ContentClassifier
- The `ContentClassifier` is responsible for analyzing the content of documents (PDF, text, images, etc.) and determining their type, confidence, and key features.
- It uses a Large Language Model (LLM, e.g., OpenAI GPT) to:
  - Extract text from files (including OCR for images)
  - Generate a prompt with industry-specific context and required features
  - Parse the LLM's response to identify document type, confidence, and features (amounts, dates, IDs, etc.)
- It can also use pattern matching (regex, fuzzy, etc.) to extract features, leveraging a `PatternStore` for industry-specific patterns.
- The classifier is extensible: new industries and features can be added via configuration files and pattern definitions.

### HybridClassifier
- The `HybridClassifier` combines the strengths of LLM-based classification and pattern-based extraction.
- It works by:
  1. Using the `ContentClassifier` to get an initial classification and feature set from the LLM.
  2. Using the `PatternMatcher` and `PatternStore` to find additional features and matches based on industry-specific patterns.
  3. Merging the results, cross-validating features, and computing a final confidence score (weighted between LLM and pattern results).
- The hybrid approach ensures:
  - Robustness to both structured and unstructured documents
  - High accuracy for both common and industry-specific features
  - Extensibility for new industries and document types

**In summary:**
- The `ContentClassifier` provides flexible, LLM-driven document understanding with optional pattern support.
- The `HybridClassifier` fuses LLM and pattern results for maximum accuracy and reliability, especially in production and industry-specific scenarios.

---

## Project Overview

This project is a document classification API that uses AI to categorize and extract features from uploaded files. It supports both synchronous and asynchronous processing, industry-specific configurations, and integrates with Redis and Celery for caching and background tasks.

---

## Setup & Deployment

### Local Development

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd heron_classifier
   ```

2. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```bash
   python -m src.app
   ```

4. **Run tests:**
   ```bash
   pytest
   ```

---

### Docker Compose (Recommended for Local & Railway Deployment)

The project includes a `docker-compose.yml` for running the full stack (Flask app, Redis, Celery worker):

```yaml
services:
  web:
    build: .
    ports:
      - "5001:5000"
    env_file:
      - .env
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_DB=0
    depends_on:
      - redis
      - celery-worker
    networks:
      - app-network

  celery-worker:
    build: .
    command: celery -A src.classifier.config.celery_config worker --loglevel=info
    env_file:
      - .env
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_DB=0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/2
    depends_on:
      - redis
    networks:
      - app-network

  redis:
    image: redis:5.0.1
    ports:
      - "6379:6379"
    networks:
      - app-network
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

volumes:
  redis-data:

networks:
  app-network:
    driver: bridge
```

**To start all services:**
```bash
docker-compose up --build
```

---

### Environment Variables

Set these in your `.env` file (used by Docker Compose):

- `REDIS_HOST=redis`
- `REDIS_PORT=6379`
- `REDIS_DB=0`
- `OPENAI_API_KEY`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## Redis

- Used for caching classification results and as a broker for Celery tasks.
- Runs as a service in Docker Compose.
- Flask and Celery connect to Redis using the service name `redis` as the host.

---

## Celery

- Handles background/asynchronous processing of documents.
- The worker is started as a separate service in Docker Compose.
- Uses Redis as both the broker and result backend.

---

## CI/CD Pipeline

Automated with GitHub Actions:

- **On every push or pull request to `main`:**
  - Runs tests with coverage.
  - Builds and pushes Docker images to Docker Hub.
  - Triggers deployment to Render (or your chosen platform) via a deploy hook.

**Workflow summary:**
```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    # ...runs tests, coverage, etc...

  build-and-push:
    needs: test
    # ...builds and pushes Docker image...

  deploy:
    needs: build-and-push
    # ...triggers deploy via deploy hook...
```

---

## Example Usage

### Classify a File
```bash
curl -X POST -F 'file=@path/to/file.pdf' http://localhost:5000/classify_file
```

### Get a File Preview
```bash
curl -X POST -F 'file=@path/to/file.pdf' http://localhost:5000/preview_file
```

### Submit for Asynchronous Processing
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"content": "file content", "metadata": {"industry": "finance"}}' \
  http://localhost:5000/process
```

### Check Task Status
```bash
curl http://localhost:5000/status/<task_id>
```

---

## Error Handling

- All error responses are JSON with an `"error"` field.
- Common error codes: `400`, `413`, `500`.

---

## Endpoints

### 1. POST `/classify_file`
**Description:**
Classifies an uploaded file (PDF, text, etc.) and returns the predicted document type, confidence, and features.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Form Data:**
  - `file`: The file to classify (required)

**Responses:**
- **200 OK**
  ```json
  {
    "classification": {
      "document_type": "invoice",
      "confidence": 0.95,
      "features": { ... }
    },
    "file_info": {
      "mime_type": "application/pdf",
      ...
    }
  }
  ```
- **400 Bad Request**
  - No file provided, no file selected, or invalid file type.
  ```json
  { "error": "No file provided" }
  ```
- **413 Request Entity Too Large**
  - File exceeds size limit.
  ```json
  { "error": "File too large" }
  ```
- **500 Internal Server Error**
  - Unexpected error.

---

### 2. POST `/preview_file`
**Description:**
Returns a preview (e.g., first page or summary) of an uploaded file.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Form Data:**
  - `file`: The file to preview (required)

**Responses:**
- **200 OK**
  - JSON with preview data (structure depends on file type).
- **400 Bad Request**
  - No file provided, no file selected, or invalid file.
- **500 Internal Server Error**
  - Error generating preview.

---

### 3. GET `/industries`
**Description:**
Lists available industry configurations and their features.

**Request:**
- No parameters.

**Response:**
- **200 OK**
  ```json
  {
    "finance": {
      "name": "Finance",
      "description": "...",
      "features": {
        "shared": ["feature1", "feature2"],
        "specific": { "featureA": "desc", ... }
      },
      "validation_rules": { ... },
      "confidence_thresholds": { ... }
    },
    ...
  }
  ```
- **500 Internal Server Error**
  - Error loading industries.

---

### 4. POST `/process`
**Description:**
Processes a document asynchronously (e.g., for long-running classification).

**Request:**
- **Content-Type:** `application/json`
- **Body:**
  ```json
  {
    "content": "file content or data",
    "document_id": "optional-id",
    "metadata": { "industry": "finance", ... }
  }
  ```

**Responses:**
- **200 OK**
  - If cached:
    ```json
    {
      "status": "completed",
      "result": { ... },
      "cached": true
    }
    ```
  - If processing:
    ```json
    {
      "status": "processing",
      "task_id": "celery-task-id",
      "document_id": "..."
    }
    ```
- **400 Bad Request**
  - Missing/invalid content, invalid industry, or bad JSON.
- **500 Internal Server Error**
  - Unexpected error.

---

### 5. GET `/status/<task_id>`
**Description:**
Checks the status of an asynchronous document processing task.

**Request:**
- **Path Parameter:** `task_id` (string)

**Responses:**
- **200 OK**
  - If completed:
    ```json
    {
      "status": "completed",
      "result": {
        "document_id": "...",
        "doc_type": "...",
        "confidence": ...,
        "features": { ... },
        "pattern_matches": { ... },
        "metadata": { ... }
      }
    }
    ```
  - If processing:
    ```json
    { "status": "processing", "task_id": "..." }
    ```
  - If failed:
    ```json
    { "status": "failed", "error": "..." }
    ```
  - If expired:
    ```json
    { "status": "expired", "error": "Task result has expired" }
    ```
- **500 Internal Server Error**
  - Unexpected error.

---

### 6. GET `/health`
**Description:**
Health check endpoint (checks Redis and classifier).

**Request:**
- No parameters.

**Responses:**
- **200 OK**
  ```json
  {
    "status": "healthy",
    "redis": "connected",
    "classifier": "initialized"
  }
  ```
- **500 Internal Server Error**
  ```json
  {
    "status": "unhealthy",
    "error": "..."
  }
  ```

---

## Extending to New Industries

The API supports industry-specific document classification and feature extraction. You can add support for new industries by creating a new industry configuration file.

### Steps to Add a New Industry

1. **Create a New Industry Config File**
   - Go to `src/classifier/config/industries/`.
   - Copy an existing industry config (e.g., `financial.json`) or create a new file (e.g., `legal.json`).

2. **Define Shared and Specific Features**
   - In your config, list shared features (by ID) that are common across industries (see `base_features.json`).
   - Add any industry-specific features under the `specific` section, including patterns, validation rules, and descriptions.

3. **Set Validation Rules and Confidence Thresholds**
   - Specify which features are required for a valid classification.
   - Set confidence thresholds for your industry if needed.

4. **Validate Your Config**
   - The config will be validated against the JSON schema in `src/classifier/config/schemas/industry_schema.json`.
   - You can use the `IndustryConfigManager` class or run the test suite to check for errors.

5. **(Optional) Add Patterns**
   - If you want to add new regex or fuzzy patterns for feature extraction, update the relevant sections or use the pattern management utilities.

6. **Test Your Industry**
   - Add or update tests (see `tests/test_config_manager.py`) to ensure your new industry is recognized and features are extracted as expected.
   - Restart the service to load the new config.

### Example: Minimal Legal Industry Config
```json
{
  "industry_name": "legal",
  "version": "1.0.0",
  "description": "Legal documents configuration",
  "features": {
    "shared": ["date", "amount", "signature"],
    "specific": {
      "case_number": {
        "type": "identifier",
        "importance": "required",
        "patterns": ["Case\\s+No\\.?\\s*\\d{2}-\\d{4}", "\\d{2}-[A-Z]{2}-\\d{4}"],
        "validation": {
          "format": "text",
          "rules": [{ "type": "regex", "value": "^\\d{2}[-A-Z]*\\d{4}$" }]
        },
        "description": "Legal case identifier"
      }
    },
    "validation_rules": {
      "required_features": ["case_number", "date"]
    },
    "confidence_thresholds": {
      "minimum": 0.7,
      "high": 0.9
    }
  }
}
```

For more details, see the code in `src/classifier/config/config_manager.py` and the test examples in `tests/test_config_manager.py`.

---

## Approach and Extension Guide

### Approach

- **Hybrid Classification:** Combines LLM-based and pattern-based extraction for robust, extensible document classification.
- **Industry Configurability:** Industry-specific features, patterns, and validation rules are defined in JSON configs, making it easy to add new domains.
- **Scalability:** Asynchronous processing (Celery), caching (Redis), and containerization (Docker) support high throughput and easy scaling.

### How to Extend

- **Add a New Industry:**
  - See the "Extending to New Industries" section above for step-by-step instructions.
- **Add New Features or Patterns:**
  - Update `base_features.json` or the `specific` section in your industry config. Add new patterns in the pattern store if needed.
- **Add New Endpoints:**
  - Implement new Flask routes in `src/app.py` and document them in this file.
- **Improve Extraction:**
  - Enhance `ContentClassifier`, `PatternMatcher`, or add new extraction modules as needed.

### Key Files

- `src/classifier/config/industries/` — Industry configs
- `src/classifier/config/base_features.json` — Shared features
- `src/classifier/content_classifier.py` — LLM-based extraction
- `src/classifier/pattern_learning/` — Pattern-based extraction
- `src/app.py` — API endpoints

---

## Detailed API Response Reference

### 1. POST `/classify_file`

**Success Response (`200 OK`):**
```json
{
  "classification": {
    "document_type": "invoice",
    "confidence": 0.95,
    "features": [
      { "type": "amount", "value": "$1500", "present": true },
      { "type": "date", "value": "2024-03-15", "present": true },
      { "type": "invoice_number", "value": "INV-2024-001", "present": true }
    ]
  },
  "file_info": {
    "mime_type": "application/pdf",
    "filename": "invoice.pdf",
    "size": 123456
  }
}
```
**Field meanings:**
- `classification.document_type`: e.g., `"invoice"`, `"bank_statement"`, `"unknown"`
- `classification.confidence`: float, 0–1
- `classification.features`: array of objects, each with:
  - `type`: e.g., `"amount"`, `"date"`, `"invoice_number"`
  - `value`: extracted value as string
  - `present`: boolean
- `file_info.mime_type`: e.g., `"application/pdf"`
- `file_info.filename`: original filename
- `file_info.size`: file size in bytes

**Error Responses:**
- `400 Bad Request`:
  - `{ "error": "No file provided" }`
  - `{ "error": "No file selected" }`
  - `{ "error": "Invalid file type" }`
- `413 Request Entity Too Large`:
  - `{ "error": "File too large" }`
- `500 Internal Server Error`:
  - `{ "error": "Unexpected error" }`

---

### 2. POST `/preview_file`

**Success Response (`200 OK`):**
```json
{
  "preview_available": true,
  "preview": "INVOICE\nInvoice Number: INV-2024-001\nDate: March 15, 2024\nTotal Amount: $1500",
  "file_info": {
    "mime_type": "application/pdf",
    "filename": "invoice.pdf",
    "size": 123456
  }
}
```
**Field meanings:**
- `preview_available`: boolean
- `preview`: string (text preview or summary)
- `file_info`: as above

**Error Responses:**
- `400 Bad Request`:
  - `{ "error": "No file provided" }`
  - `{ "error": "Invalid file" }`
- `500 Internal Server Error`:
  - `{ "error": "Error generating preview" }`

---

### 3. GET `/industries`

**Success Response (`200 OK`):**
```json
{
  "finance": {
    "name": "Finance",
    "description": "Financial documents (invoices, statements, etc.)",
    "features": {
      "shared": ["amount", "date", "account_number"],
      "specific": {
        "invoice_number": "Unique invoice identifier",
        "payment_terms": "Terms of payment"
      }
    },
    "validation_rules": {
      "required_features": ["amount", "date"]
    },
    "confidence_thresholds": {
      "minimum": 0.7,
      "high": 0.9
    }
  },
  "healthcare": {
    "name": "Healthcare",
    "description": "Medical and healthcare documents",
    "features": {
      "shared": ["amount", "date", "patient_id"],
      "specific": {
        "cpt_code": "Medical procedure code",
        "provider_npi": "Provider NPI number"
      }
    },
    "validation_rules": {
      "required_features": ["patient_id", "date"]
    },
    "confidence_thresholds": {
      "minimum": 0.6,
      "high": 0.85
    }
  }
}
```
**Field meanings:**
- Each top-level key is an industry.
- Each industry object includes:
  - `name`: string
  - `description`: string
  - `features.shared`: array of strings
  - `features.specific`: object mapping feature names to descriptions
  - `validation_rules.required_features`: array of strings
  - `confidence_thresholds.minimum`: float
  - `confidence_thresholds.high`: float

**Error Response:**
- `500 Internal Server Error`:
  - `{ "error": "Error loading industries" }`

---

### 4. POST `/process`

**Request Body Example:**
```json
{
  "content": "INVOICE\nInvoice Number: INV-2024-001\nDate: March 15, 2024\nTotal Amount: $1500",
  "document_id": "doc-123",
  "metadata": { "industry": "finance" }
}
```

**Success Response if cached (`200 OK`):**
```json
{
  "status": "completed",
  "result": {
    "document_id": "doc-123",
    "doc_type": "invoice",
    "confidence": 0.95,
    "features": [
      { "type": "amount", "value": "$1500", "present": true },
      { "type": "date", "value": "2024-03-15", "present": true },
      { "type": "invoice_number", "value": "INV-2024-001", "present": true }
    ],
    "pattern_matches": [],
    "metadata": {
      "industry": "finance",
      "llm_confidence": 0.95,
      "pattern_confidence": null,
      "pattern_count": 0
    }
  },
  "cached": true
}
```

**Success Response if processing started (`200 OK`):**
```json
{
  "status": "processing",
  "task_id": "celery-task-id-abc123",
  "document_id": "doc-123"
}
```

**Error Responses:**
- `400 Bad Request`:
  - `{ "error": "Missing content" }`
  - `{ "error": "Invalid industry" }`
- `500 Internal Server Error`:
  - `{ "error": "Unexpected error" }`

---

### 5. GET `/status/<task_id>`

**Success Response if completed (`200 OK`):**
```json
{
  "status": "completed",
  "result": {
    "document_id": "doc-123",
    "doc_type": "invoice",
    "confidence": 0.95,
    "features": [
      { "type": "amount", "value": "$1500", "present": true },
      { "type": "date", "value": "2024-03-15", "present": true },
      { "type": "invoice_number", "value": "INV-2024-001", "present": true }
    ],
    "pattern_matches": [],
    "metadata": {
      "industry": "finance",
      "llm_confidence": 0.95,
      "pattern_confidence": null,
      "pattern_count": 0
    }
  }
}
```

**Success Response if still processing:**
```json
{ "status": "processing", "task_id": "celery-task-id-abc123" }
```

**Success Response if failed:**
```json
{ "status": "failed", "error": "Classification failed: <reason>" }
```

**Success Response if expired:**
```json
{ "status": "expired", "error": "Task result has expired" }
```

**Error Response:**
- `500 Internal Server Error`:
  - `{ "error": "Unexpected error" }`

---

### 6. GET `/health`

**Success Response (`200 OK`):**
```json
{
  "status": "healthy",
  "redis": "connected",
  "classifier": "initialized"
}
```

**Error Response:**
- `500 Internal Server Error`:
  ```json
  {
    "status": "unhealthy",
    "error": "Redis connection failed"
  }
  ```
