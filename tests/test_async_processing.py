"""Tests for async document processing API."""

import pytest
import json
import time
import concurrent.futures
from unittest.mock import patch, MagicMock
from redis.exceptions import ConnectionError
from flask import Flask
from src.app import app as flask_app
from src.classifier.services.cache_service import CacheService
from src.classifier.hybrid_classifier import HybridClassificationResult
from src.classifier.pattern_learning.models import Pattern, PatternMatch, PatternType, MatchType, ConfidenceScore

@pytest.fixture
def mock_hybrid_result():
    """Create a mock hybrid classification result."""
    pattern = Pattern(
        id="test-pattern",
        type=PatternType.REGEX,
        expression=r"\$\d+\.\d{2}",
        feature_type="amount",
        industry="financial"
    )

    match = PatternMatch(
        pattern=pattern,
        text="$100.00",
        start=0,
        end=7,
        match_type=MatchType.EXACT,
        confidence=ConfidenceScore(0.9)
    )

    return HybridClassificationResult(
        doc_type="invoice",
        confidence=0.9,
        features=[
            {
                "type": "amount",
                "values": ["$100.00"],
                "present": True
            }
        ],
        pattern_matches=[match],
        llm_result=None,
        metadata={"industry": "financial"}
    )

@pytest.fixture
def mock_async_result(mock_hybrid_result):
    """Create a mock async result."""
    mock_result = MagicMock()
    mock_result.id = 'test-task-id'
    mock_result.ready.return_value = True
    mock_result.successful.return_value = True

    # Convert HybridClassificationResult to serializable format
    serialized_result = {
        'document_id': 'test-doc-1',
        'doc_type': mock_hybrid_result.doc_type,
        'confidence': mock_hybrid_result.confidence,
        'features': mock_hybrid_result.features,
        'pattern_matches': [
            {
                'pattern_id': match.pattern.id,
                'text': match.text,
                'start': match.start,
                'end': match.end,
                'match_type': match.match_type.value,
                'confidence': match.confidence.value,
                'context': match.context
            }
            for match in mock_hybrid_result.pattern_matches
        ],
        'metadata': mock_hybrid_result.metadata
    }

    mock_result.get.return_value = serialized_result
    return mock_result

@pytest.fixture
def mock_celery(mock_async_result):
    """Mock Celery task results."""
    with patch('src.classifier.tasks.process_document.delay') as mock_delay, \
         patch('src.classifier.tasks.process_document.AsyncResult') as mock_async:
        mock_delay.return_value = mock_async_result
        mock_async.return_value = mock_async_result
        yield mock_delay

@pytest.fixture
def app_with_mocks(mock_celery):
    """Configure Flask app with mocked dependencies."""
    with patch('src.app.cache_service') as mock_cache:
        mock_cache.get.return_value = None  # Default to cache miss
        mock_cache.set.return_value = True
        flask_app.config['TESTING'] = True
        return flask_app

@pytest.fixture
def client(app_with_mocks):
    """Flask test client with mocked dependencies."""
    with app_with_mocks.test_client() as client:
        yield client

def test_process_document_async(client, mock_celery, mock_async_result):
    """Test async document processing flow."""
    # Submit a document for processing
    response = client.post('/process', json={
        'document_id': 'test-doc-1',
        'content': 'This is a test document for async processing.',
        'metadata': {'test': True}
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'processing'
    assert data['task_id'] == 'test-task-id'

    # Check status endpoint
    status_resp = client.get('/status/test-task-id')
    assert status_resp.status_code == 200
    status_data = status_resp.get_json()
    assert status_data['status'] == 'completed'
    assert 'result' in status_data

def test_process_document_cache_hit(client, app_with_mocks):
    """Test cache hit on repeated document submission."""
    doc = {
        'document_id': 'test-doc-2',
        'content': 'This is a cache hit test.',
        'metadata': {'industry': 'financial'}
    }

    # Set up cache hit for second request
    cached_result = {
        'document_id': 'test-doc-2',
        'doc_type': 'invoice',
        'confidence': 0.9,
        'features': [
            {
                'type': 'amount',
                'values': ['$100.00'],
                'present': True
            }
        ],
        'pattern_matches': [
            {
                'pattern_id': 'test-pattern',
                'text': '$100.00',
                'start': 0,
                'end': 7,
                'match_type': 'EXACT',
                'confidence': 0.9,
                'context': {}
            }
        ],
        'metadata': {'industry': 'financial'}
    }

    with patch('src.app.cache_service') as mock_cache:
        # First request: cache miss
        mock_cache.get.return_value = None
        resp1 = client.post('/process', json=doc)
        assert resp1.status_code == 200
        data1 = resp1.get_json()
        assert data1['status'] == 'processing'

        # Second request: cache hit
        mock_cache.get.return_value = cached_result
        resp2 = client.post('/process', json=doc)
        assert resp2.status_code == 200
        data2 = resp2.get_json()
        assert data2['status'] == 'completed'
        assert data2.get('cached') is True
        assert data2['result'] == cached_result

def test_process_document_invalid_input(client):
    """Test error handling for invalid input."""
    # Missing content
    resp = client.post('/process', json={'document_id': 'bad-doc'})
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data
    assert 'No content provided' in data['error']

    # No JSON
    resp2 = client.post('/process',
                       data='not json',
                       content_type='application/json')
    assert resp2.status_code == 400
    data2 = resp2.get_json()
    assert 'error' in data2

    # Empty content
    resp3 = client.post('/process', json={
        'document_id': 'empty-doc',
        'content': ''
    })
    assert resp3.status_code == 400
    data3 = resp3.get_json()
    assert 'error' in data3

def test_process_document_task_failure(client, app_with_mocks):
    """Test handling of failed tasks."""
    with patch('src.classifier.tasks.process_document.AsyncResult') as mock_async:
        # Configure mock to simulate a failed task
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.successful.return_value = False
        mock_result.result = Exception("Task processing failed")
        mock_async.return_value = mock_result

        # Check status of failed task
        status_resp = client.get('/status/failed-task-id')
        assert status_resp.status_code == 500
        status_data = status_resp.get_json()
        assert status_data['status'] == 'failed'
        assert 'error' in status_data

def test_redis_connection_error(client, app_with_mocks):
    """Test graceful handling of Redis connection errors."""
    with patch('src.app.cache_service') as mock_cache:
        # Simulate Redis connection error
        mock_cache.get.side_effect = ConnectionError("Redis connection failed")

        response = client.post('/process', json={
            'document_id': 'test-doc',
            'content': 'Test content'
        })

        # Should still accept the task even if cache fails
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'processing'
        assert 'task_id' in data

def test_celery_integration(client, app_with_mocks):
    """Test actual Celery task execution (requires running Celery worker)."""
    from src.classifier.tasks import process_document

    # Submit a real task
    task = process_document.delay(
        'test-doc',
        'Test content',
        {'test': True}
    )

    # Should get a real task ID
    assert task.id is not None

    # Check task status (this will actually contact Celery)
    status_resp = client.get(f'/status/{task.id}')
    assert status_resp.status_code == 200
    status_data = status_resp.get_json()
    assert status_data['status'] in ['processing', 'completed', 'failed']

def test_concurrent_requests(client, app_with_mocks):
    """Test handling multiple concurrent requests."""
    num_requests = 5

    def make_request(i):
        # Create a new test client for each request to avoid context issues
        with app_with_mocks.test_client() as test_client:
            return test_client.post('/process', json={
                'document_id': f'test-doc-{i}',
                'content': f'Test content {i}',
                'metadata': {'test': True}
            })

    # Make concurrent requests
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                pytest.fail(f"Request failed: {str(e)}")

    # Check all responses
    for response in responses:
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'processing'
        assert 'task_id' in data

def test_task_retry(client, app_with_mocks):
    """Test task retry mechanism."""
    with patch('src.classifier.tasks.process_document.retry') as mock_retry:
        # Configure the task to fail and retry
        mock_async_result = MagicMock()
        mock_async_result.ready.return_value = False
        mock_async_result.failed.return_value = True
        mock_async_result.retries = 1

        with patch('src.classifier.tasks.process_document.AsyncResult') as mock_async:
            mock_async.return_value = mock_async_result

            # Check status of a retrying task
            status_resp = client.get('/status/retry-task-id')
            assert status_resp.status_code == 200
            status_data = status_resp.get_json()
            assert status_data['status'] == 'processing'

def test_task_result_expiry(client, app_with_mocks):
    """Test handling of expired task results."""
    # Mock an expired task result
    mock_async_result = MagicMock()
    mock_async_result.ready.return_value = True
    mock_async_result.successful.return_value = False
    mock_async_result.failed.return_value = False  # Neither successful nor failed indicates expired

    with patch('src.classifier.tasks.process_document.AsyncResult') as mock_async:
        mock_async.return_value = mock_async_result

        # Check status of expired task
        status_resp = client.get('/status/expired-task-id')
        assert status_resp.status_code == 404  # or whatever status code you want to use for expired tasks
        status_data = status_resp.get_json()
        assert 'error' in status_data
        assert 'expired' in status_data['error'].lower()

def test_load(client, app_with_mocks, mock_celery, mock_async_result):
    """Load test the API with concurrent requests."""
    num_requests = 50
    num_threads = 10
    success_threshold = 0.9  # 90% success rate required
    max_avg_time = 1.0  # Maximum average time per request in seconds

    # Set up logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Configure mock outside the request function to ensure consistent state
    mock_celery.return_value = mock_async_result
    logger.info("Initial mock configuration:")
    logger.info(f"Mock celery task ID: {mock_async_result.id}")
    logger.info(f"Mock celery ready: {mock_async_result.ready()}")
    logger.info(f"Mock celery successful: {mock_async_result.successful()}")

    def make_request():
        start_time = time.time()
        request_id = f'load-test-{time.time()}'
        logger.info(f"\n=== Starting request {request_id} ===")

        try:
            # Create a new test client for each request
            with app_with_mocks.test_client() as test_client:
                payload = {
                    'document_id': request_id,
                    'content': 'Load test content',
                    'metadata': {'test': True}
                }
                logger.info(f"Sending request with payload: {payload}")

                # Make the request within the test client context
                response = test_client.post('/process', json=payload)
                duration = time.time() - start_time

                logger.info(f"Response status: {response.status_code}")
                try:
                    response_data = response.get_json()
                    logger.info(f"Response data: {response_data}")
                except Exception as e:
                    logger.error(f"Failed to get JSON from response: {e}")
                    logger.error(f"Response content: {response.data}")
                    response_data = None

                result = {
                    'success': response.status_code == 200,
                    'duration': duration,
                    'status_code': response.status_code,
                    'response_data': response_data
                }
                logger.info(f"Request result: {result}")
                return result

        except Exception as e:
            duration = time.time() - start_time
            error_result = {
                'success': False,
                'duration': duration,
                'error': str(e)
            }
            logger.error(f"Request failed with error: {error_result}")
            return error_result

    print("\nStarting load test...")
    results = []

    # Create a thread pool and submit all requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_requests):
            logger.info(f"Submitting request {i+1}/{num_requests}")
            futures.append(executor.submit(make_request))
            time.sleep(0.1)  # Small delay between requests

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed request with result: {result}")
            except Exception as e:
                logger.error(f"Future failed with error: {e}")
                results.append({'success': False, 'error': str(e)})

    # Calculate statistics
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / num_requests
    avg_time = sum(r['duration'] for r in results) / num_requests
    max_time = max(r['duration'] for r in results)

    # Log detailed results
    logger.info("\n=== Detailed Results ===")
    logger.info(f"Total requests: {num_requests}")
    logger.info(f"Successful requests: {successes}")
    logger.info(f"Failed requests: {num_requests - successes}")
    logger.info(f"Success rate: {success_rate:.2%}")
    logger.info(f"Average time: {avg_time:.3f}s")
    logger.info(f"Max time: {max_time:.3f}s")

    # Log all failed requests
    failed_requests = [r for r in results if not r['success']]
    if failed_requests:
        logger.error("\n=== Failed Requests ===")
        for i, fail in enumerate(failed_requests, 1):
            logger.error(f"Failure {i}:")
            logger.error(f"Status code: {fail.get('status_code')}")
            logger.error(f"Response data: {fail.get('response_data')}")
            if fail.get('response_data') and 'error' in fail['response_data']:
                logger.error(f"Response error: {fail['response_data']['error']}")
            logger.error(f"Error: {fail.get('error')}")

    print(f"\nLoad Test Results:")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrent Threads: {num_threads}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Time: {avg_time:.3f}s")
    print(f"Max Time: {max_time:.3f}s")

    # Assert test requirements
    assert success_rate >= success_threshold, f"Success rate {success_rate:.2%} below threshold {success_threshold:.2%}"
    assert avg_time <= max_avg_time, f"Average time {avg_time:.3f}s exceeds limit {max_avg_time}s"

def test_load_with_cache_hits(client, app_with_mocks, mock_celery, mock_async_result):
    """Load test with frequent cache hits."""
    num_requests = 30
    num_threads = 5
    cached_result = {
        'status': 'completed',
        'cached': True,
        'result': {
            'features': {'cached': True},
            'matches': [{'text': 'cached match', 'confidence': 0.95}]
        }
    }

    # Configure cache to return hits for even-numbered requests
    def get_from_cache(key):
        request_id = key.split(':')[-1]  # Extract doc ID from cache key
        if 'doc-even' in request_id:
            return cached_result
        return None

    with patch('src.app.cache_service') as mock_cache:
        mock_cache.get.side_effect = get_from_cache
        mock_cache.set.return_value = True

        def make_request(i):
            with app_with_mocks.test_client() as test_client:
                doc_id = f'doc-{"even" if i % 2 == 0 else "odd"}-{i}'
                start_time = time.time()
                response = test_client.post('/process', json={
                    'document_id': doc_id,
                    'content': 'Test content',
                    'metadata': {'test': True}
                })
                duration = time.time() - start_time

                # Get response data and properly check for cache hit
                response_data = response.get_json()
                is_cached = (
                    response_data.get('cached', False) or
                    response_data.get('status') == 'completed' and
                    'cached' in str(response_data)
                )

                return {
                    'success': response.status_code == 200,
                    'duration': duration,
                    'cached': is_cached,
                    'response': response_data
                }

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # Calculate statistics
        successes = sum(1 for r in results if r['success'])
        cache_hits = sum(1 for r in results if r.get('cached', False))
        avg_time = sum(r['duration'] for r in results) / num_requests

        # Log results for debugging
        print("\nCache Test Results:")
        print(f"Total requests: {num_requests}")
        print(f"Successful requests: {successes}")
        print(f"Cache hits: {cache_hits}")
        print(f"Average time: {avg_time:.3f}s")

        # Log some sample responses
        print("\nSample responses:")
        for i, result in enumerate(results[:5]):
            print(f"\nRequest {i}:")
            print(f"Success: {result['success']}")
            print(f"Cached: {result['cached']}")
            print(f"Response: {result['response']}")

        assert successes == num_requests, "All requests should succeed"
        assert cache_hits >= num_requests // 2, "Should have cache hits for even-numbered requests"
        assert avg_time < 0.1, "Average response time should be fast"

def test_load_with_redis_failures(client, app_with_mocks, mock_celery, mock_async_result):
    """Load test with intermittent Redis failures."""
    num_requests = 20
    num_threads = 4

    # Configure cache to fail intermittently
    fail_counter = 0
    def failing_cache_get(key):
        nonlocal fail_counter
        fail_counter += 1
        if fail_counter % 3 == 0:  # Fail every third request
            raise ConnectionError("Simulated Redis failure")
        return None

    with patch('src.app.cache_service') as mock_cache:
        mock_cache.get.side_effect = failing_cache_get
        mock_cache.set.side_effect = failing_cache_get

        def make_request(i):
            with app_with_mocks.test_client() as test_client:
                start_time = time.time()
                response = test_client.post('/process', json={
                    'document_id': f'test-doc-{i}',
                    'content': 'Test content',
                    'metadata': {'test': True}
                })
                duration = time.time() - start_time
                return {
                    'success': response.status_code == 200,
                    'duration': duration
                }

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # All requests should succeed despite cache failures
        successes = sum(1 for r in results if r['success'])
        assert successes == num_requests, "All requests should succeed even with Redis failures"

def test_load_with_long_tasks(client, app_with_mocks):
    """Load test with long-running tasks."""
    num_requests = 15
    num_threads = 3

    # Track task states
    task_states = {}
    task_counter = 0

    def create_mock_result(task_id):
        mock = MagicMock()
        mock.id = task_id
        # Task will complete after 3 status checks
        mock.ready.side_effect = lambda: task_states[task_id]['checks'] >= 3
        mock.successful.return_value = True
        mock.get.return_value = {
            'document_id': f'long-task-doc-{task_id.split("-")[-1]}',
            'doc_type': 'test_type',
            'confidence': 0.9,
            'features': {'test_feature': True},
            'pattern_matches': [{'text': 'test match', 'confidence': 0.9}],
            'metadata': {'test': True},
            'status': 'completed',
            'error': None
        }
        return mock

    def get_mock_result(task_id):
        if task_id not in task_states:
            task_states[task_id] = {'checks': 0}
        task_states[task_id]['checks'] += 1
        return task_states[task_id]['mock']

    with patch('src.classifier.tasks.process_document.delay') as mock_delay, \
         patch('src.classifier.tasks.process_document.AsyncResult') as mock_async:

        def mock_delay_func(*args, **kwargs):
            nonlocal task_counter
            task_counter += 1
            task_id = f'long-task-{task_counter}'
            mock_result = create_mock_result(task_id)
            task_states[task_id] = {'checks': 0, 'mock': mock_result}
            return mock_result

        mock_delay.side_effect = mock_delay_func
        mock_async.side_effect = get_mock_result

        def make_request(i):
            with app_with_mocks.test_client() as test_client:
                start_time = time.time()

                # Submit task
                response = test_client.post('/process', json={
                    'document_id': f'long-task-doc-{i}',
                    'content': 'Test content',
                    'metadata': {'test': True}
                })

                if response.status_code != 200:
                    print(f"Request {i} failed with status {response.status_code}")
                    return {
                        'success': False,
                        'duration': time.time() - start_time,
                        'status_checks': 0,
                        'error': f"Initial request failed: {response.status_code}"
                    }

                data = response.get_json()
                task_id = data['task_id']

                # Check status until complete
                status_checks = 0
                max_checks = 5
                while status_checks < max_checks:
                    status_resp = test_client.get(f'/status/{task_id}')
                    if status_resp.status_code != 200:
                        print(f"Status check failed for task {task_id}: {status_resp.status_code}")
                        break

                    status_data = status_resp.get_json()
                    print(f"Task {task_id} status: {status_data.get('status')} (check {status_checks + 1})")

                    if status_data.get('status') == 'completed':
                        break

                    status_checks += 1
                    time.sleep(0.1)

                duration = time.time() - start_time
                final_status = status_data.get('status') if 'status_data' in locals() else 'unknown'

                return {
                    'success': final_status == 'completed',
                    'duration': duration,
                    'status_checks': status_checks,
                    'task_id': task_id,
                    'final_status': final_status
                }

        # Run requests
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Request failed with error: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e)
                    })

        # Print detailed results
        print("\nLong Tasks Test Results:")
        print(f"Total requests: {num_requests}")
        successes = sum(1 for r in results if r['success'])
        print(f"Successful requests: {successes}")
        print(f"Task states recorded: {len(task_states)}")

        print("\nDetailed results:")
        for i, result in enumerate(results):
            print(f"\nRequest {i}:")
            for k, v in result.items():
                print(f"{k}: {v}")
            if result['task_id'] in task_states:
                print(f"Task checks: {task_states[result['task_id']]['checks']}")

        # Verify results
        assert successes == num_requests, f"All long-running tasks should complete (got {successes}/{num_requests})"

        avg_checks = sum(r['status_checks'] for r in results) / num_requests
        assert avg_checks > 1, "Tasks should require multiple status checks"

        # Verify all tasks went through proper state transitions
        for task_id, state in task_states.items():
            assert state['checks'] >= 3, f"Task {task_id} didn't go through enough status checks"