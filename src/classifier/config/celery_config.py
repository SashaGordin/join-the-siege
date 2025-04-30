"""Celery configuration for task queues."""

from celery import Celery
from kombu import Exchange, Queue

# Create Celery app
celery_app = Celery('classifier',
                   broker='redis://localhost:6379/1',  # Redis DB 1 for message broker
                   backend='redis://localhost:6379/2')  # Redis DB 2 for results backend

# Configure Celery
celery_app.conf.update(
    # Task queues configuration
    task_queues=(
        Queue('document_processing', Exchange('document_processing'), routing_key='document_processing'),
        Queue('pattern_matching', Exchange('pattern_matching'), routing_key='pattern_matching'),
        Queue('feature_extraction', Exchange('feature_extraction'), routing_key='feature_extraction'),
    ),

    # Route tasks to specific queues
    task_routes={
        'classifier.tasks.process_document': {'queue': 'document_processing'},
        'classifier.tasks.match_patterns': {'queue': 'pattern_matching'},
        'classifier.tasks.extract_features': {'queue': 'feature_extraction'},
    },

    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task result settings
    task_ignore_result=False,  # We want to store task results
    result_expires=3600,  # Results expire after 1 hour

    # Performance settings
    worker_prefetch_multiplier=1,  # Prevent worker from prefetching too many tasks
    task_acks_late=True,  # Only acknowledge task after it's completed

    # Retry settings
    task_retry_delay=300,  # 5 minutes between retries
    task_max_retries=3,  # Maximum 3 retries per task

    # Monitoring settings
    worker_send_task_events=True,  # Enable task events for monitoring
    task_send_sent_event=True,  # Track when tasks are sent

    # Rate limiting
    task_default_rate_limit='100/m',  # Default to 100 tasks per minute

    # Error handling
    task_reject_on_worker_lost=True,  # Reject tasks if worker disconnects
    task_soft_time_limit=600,  # 10 minute soft timeout
    task_time_limit=900,  # 15 minute hard timeout
)

# Optional: Configure logging
celery_app.conf.update(
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'
)