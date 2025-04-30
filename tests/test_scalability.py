import pytest
import time
import json
import logging
import threading
import concurrent.futures
from unittest.mock import patch, MagicMock
from typing import Dict, List
from flask import Flask, current_app
from src.app import app as flask_app

# Configure thread-safe logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-local storage for logging
thread_local = threading.local()

@pytest.fixture
def app():
    """Create a Flask test app."""
    flask_app.config.update({
        'TESTING': True,
        'SECRET_KEY': 'test_secret_key',
        'SESSION_TYPE': 'filesystem',
        'PRESERVE_CONTEXT_ON_EXCEPTION': False,  # Important for test client
        'SERVER_NAME': 'localhost'  # Required for url_for() to work
    })
    logger.info("Created test Flask app with TESTING=True")

    # Push an application context
    with flask_app.app_context():
        yield flask_app

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    logger.info("Created mock Redis client")
    return mock

@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    logger.info("Created mock cache service")
    return mock

@pytest.fixture
def pattern_matcher():
    """Mock pattern matcher."""
    mock = MagicMock()
    mock.find_matches.return_value = []
    logger.info("Created mock pattern matcher")
    return mock

@pytest.fixture
def app_with_mocks(app, mock_redis, mock_cache_service, pattern_matcher):
    """Create app with all required mocks."""
    logger.info("Setting up app with mocks")

    # Store completed tasks
    completed_tasks = {}

    def mock_delay(document_id, content, metadata=None):
        # Create a mock task
        mock_task = MagicMock()
        mock_task.id = f'task_{document_id}'
        mock_task.ready.return_value = True
        mock_task.successful.return_value = True

        # Generate result
        result = {
            'document_id': document_id,
            'features': {'test': True},
            'matches': [],
            'status': 'completed'
        }
        mock_task.get.return_value = result

        # Store in completed tasks
        completed_tasks[document_id] = result

        # Store in cache
        cache_key = f'document_processing:{document_id}'
        mock_cache_service.set(cache_key, result)

        return mock_task

    with patch('src.app.cache_service', mock_cache_service), \
         patch('src.app.content_classifier', MagicMock()), \
         patch('src.app.process_document.delay', side_effect=mock_delay):

        yield app

def make_request_with_context(app, doc: Dict) -> Dict:
    """Make a request ensuring proper Flask context."""
    start_time = time.time()
    logger.info(f"Making request for document {doc.get('document_id', 'unknown')}")
    try:
        # Create a new test client for each request
        with app.test_client() as test_client:
            # Make the request within the test client's context
            response = test_client.post('/process', json=doc)
            duration = time.time() - start_time

            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response data: {response.get_data(as_text=True)}")

            result = {
                'success': response.status_code == 200,
                'duration': duration,
                'doc_id': doc['document_id'],
                'cached': 'cached' in str(response.get_json() or {}),
                'status_code': response.status_code,
                'response_data': response.get_data(as_text=True)
            }
            logger.info(f"Request completed: {json.dumps(result)}")
            return result
    except Exception as e:
        duration = time.time() - start_time
        error_result = {
            'success': False,
            'duration': duration,
            'doc_id': doc.get('document_id', 'unknown'),
            'error': str(e),
            'error_type': type(e).__name__
        }
        logger.error(f"Request failed: {json.dumps(error_result)}", exc_info=True)
        return error_result

def test_burst_load(app_with_mocks):
    """Test processing a burst of documents quickly (simulating peak load)."""
    logger.info("Starting burst load test")
    num_requests = 500  # Simulate burst of 500 requests
    num_threads = 10    # Use 10 threads for concurrent processing

    logger.info(f"Preparing {num_requests} requests with {num_threads} threads")

    # Generate document pool
    docs = [
        {
            'document_id': f'doc_{i}',
            'content': f'Test document {i}',
            'metadata': {'test': True}
        }
        for i in range(num_requests)
    ]

    # Track metrics
    start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a partial function with the app
        request_func = lambda doc: make_request_with_context(app_with_mocks, doc)

        # Submit all requests
        futures = [executor.submit(request_func, doc) for doc in docs]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if not result['success']:
                    logger.error(f"Request failed: {json.dumps(result)}")
            except Exception as e:
                logger.error(f"Thread execution failed", exc_info=True)
                results.append({
                    'success': False,
                    'duration': time.time() - start_time,
                    'error': str(e),
                    'error_type': type(e).__name__
                })

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    successful_requests = sum(1 for r in results if r['success'])
    failed_requests = len(results) - successful_requests
    success_rate = successful_requests / num_requests
    actual_throughput = num_requests / total_time
    avg_latency = sum(r['duration'] for r in results) / len(results)

    # Log detailed metrics
    logger.info("\nBurst Load Test Results:")
    logger.info(f"Total requests: {num_requests}")
    logger.info(f"Successful requests: {successful_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    logger.info(f"Success rate: {success_rate:.2%}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Throughput: {actual_throughput:.2f} docs/second")
    logger.info(f"Average latency: {avg_latency:.3f}s")

    # Log failure details if any
    if failed_requests > 0:
        logger.error("\nFailure Details:")
        failure_types = {}
        for result in results:
            if not result['success']:
                error_type = result.get('error_type', 'Unknown')
                error_msg = result.get('error', 'No error message')
                status_code = result.get('status_code', 'No status code')
                response_data = result.get('response_data', 'No response data')

                if error_type not in failure_types:
                    failure_types[error_type] = 0
                failure_types[error_type] += 1

                logger.error(f"Document {result.get('doc_id')}: {error_type}")
                logger.error(f"Error message: {error_msg}")
                logger.error(f"Status code: {status_code}")
                logger.error(f"Response data: {response_data}")

        logger.error("\nFailure Summary:")
        for error_type, count in failure_types.items():
            logger.error(f"{error_type}: {count} occurrences")

    # Verify performance meets requirements
    assert success_rate >= 0.98, f"Success rate {success_rate:.2%} below 98% threshold"
    assert avg_latency <= 0.1, f"Average latency {avg_latency:.3f}s exceeds 0.1s threshold"
    assert actual_throughput >= 50, f"Throughput {actual_throughput:.2f} below 50 docs/second"

def test_sustained_throughput(app_with_mocks):
    """Test if we can maintain the throughput needed for 100k docs/day."""
    target_daily_docs = 100000
    target_throughput = target_daily_docs / (24 * 60 * 60)  # docs per second
    duration = 60  # Test for 1 minute
    threads = 5  # Use 5 threads for sustained load

    with app_with_mocks.test_client() as test_client:
        def worker(thread_id: int, end_time: float) -> List[Dict]:
            results = []
            request_count = 0
            while time.time() < end_time:
                doc = {
                    'document_id': f'doc_{thread_id}_{request_count}',
                    'content': f'Test document {time.time()}',
                    'metadata': {'test': True}
                }
                result = make_request_with_context(app_with_mocks, doc)
                results.append(result)
                request_count += 1

                # Calculate current throughput and adjust delay if needed
                if len(results) > 1:
                    current_throughput = 1 / (results[-1]['duration'] if results[-1].get('duration', 0) > 0 else 0.1)
                    if current_throughput > target_throughput * 1.2:
                        time.sleep(0.01)  # Slow down if exceeding target by 20%

            return results

        start_time = time.time()
        end_time = start_time + duration

        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker, i, end_time) for i in range(threads)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    thread_results = future.result()
                    all_results.extend(thread_results)
                except Exception as e:
                    print(f"Thread failed: {str(e)}")

        # Calculate overall metrics
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r.get('success', False))
        actual_throughput = total_requests / duration
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        print(f"\nSustained Throughput Test Results:")
        print(f"Duration: {duration} seconds")
        print(f"Total requests: {total_requests}")
        print(f"Successful requests: {successful_requests}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average throughput: {actual_throughput:.2f} docs/second")
        print(f"Target throughput: {target_throughput:.2f} docs/second")
        print(f"Throughput ratio: {(actual_throughput/target_throughput):.2f}x target")

        # Verify we can maintain required throughput
        assert success_rate >= 0.98, f"Success rate {success_rate:.2%} below 98% threshold"
        assert actual_throughput >= target_throughput, (
            f"Average throughput {actual_throughput:.2f} below target {target_throughput:.2f}"
        )

def test_cache_effectiveness(app_with_mocks, mock_cache_service):
    """Test if caching helps maintain throughput with duplicate documents."""
    # Configure mock cache for predictable behavior
    cache_data = {}
    def mock_get(key):
        result = cache_data.get(key)
        if result:
            logger.info(f"Cache hit for key: {key}")
            return result
        logger.info(f"Cache miss for key: {key}")
        return None
    def mock_set(key, value):
        logger.info(f"Caching result for key: {key}")
        cache_data[key] = value
        return True
    mock_cache_service.get.side_effect = mock_get
    mock_cache_service.set.side_effect = mock_set

    num_requests = 200  # Total requests
    num_unique_docs = 50  # Number of unique documents
    num_threads = 5

    # Generate a pool of unique documents
    doc_pool = [
        {
            'document_id': f'doc_{i}',
            'content': f'Test document {i}',
            'metadata': {'test': True}
        }
        for i in range(num_unique_docs)
    ]

    def make_request(i: int) -> Dict:
        # Create a new test client for each request
        test_client = app_with_mocks.test_client()

        # Reuse documents to test caching
        doc = doc_pool[i % num_unique_docs]

        try:
            # Push an application context for this request
            with app_with_mocks.app_context():
                # Make the request within both app and request contexts
                response = test_client.post('/process', json=doc)
                response_data = response.get_json()

                # Check for cache hit in response data
                is_cache_hit = (
                    response_data.get('status') == 'completed' and
                    response_data.get('cached') is True
                )

                # Log response details for debugging
                logger.debug(f"Response for doc {doc['document_id']}: {response_data}")
                if is_cache_hit:
                    logger.info(f"Cache hit for document {doc['document_id']}")
                else:
                    logger.info(f"Cache miss for document {doc['document_id']}")

                return {
                    'success': response.status_code == 200,
                    'duration': 0,  # We'll calculate this later if needed
                    'doc_id': doc['document_id'],
                    'cached': is_cache_hit,
                    'response_data': response_data
                }
        except Exception as e:
            logger.error(f"Request failed for doc {doc['document_id']}: {str(e)}")
            return {
                'success': False,
                'duration': 0,
                'doc_id': doc['document_id'],
                'error': str(e)
            }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Future failed: {str(e)}")
                results.append({
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })

    # Calculate statistics
    successes = sum(1 for r in results if r['success'])
    cache_hits = sum(1 for r in results if r.get('cached', False))

    # Log results
    logger.info(f"\nCache Test Results:")
    logger.info(f"Total requests: {num_requests}")
    logger.info(f"Successful requests: {successes}")
    logger.info(f"Cache hits: {cache_hits}")
    logger.info(f"Cache hit rate: {cache_hits/num_requests:.2%}")

    # Sample some responses for debugging
    logger.info("\nSample responses:")
    for i, result in enumerate(results[:5]):
        logger.info(f"\nRequest {i}:")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Cached: {result['cached']}")
        logger.info(f"Doc ID: {result.get('doc_id', 'unknown')}")
        logger.info(f"Response data: {result.get('response_data', {})}")

    # Assertions
    assert successes == num_requests, "All requests should succeed"
    assert cache_hits > 0, "Should have some cache hits"
    # Since we have 200 requests and 50 unique docs, we expect at least 150 cache hits
    expected_min_cache_hits = num_requests - num_unique_docs
    assert cache_hits >= expected_min_cache_hits, f"Expected at least {expected_min_cache_hits} cache hits, got {cache_hits}"