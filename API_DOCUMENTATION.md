# API Documentation

This API powers a document classification service that uses AI to categorize and extract features from uploaded files (PDF, text, etc.). It supports synchronous and asynchronous processing, industry-specific configurations, and integrates with Redis and Celery for caching and background tasks.

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

## General Notes
- All endpoints return JSON.
- Error responses always include an `"error"` field.
- File upload endpoints expect the file under the key `file`.
- Asynchronous processing uses Celery and Redis for task management and caching.