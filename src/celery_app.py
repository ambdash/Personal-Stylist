from celery import Celery
from kombu import Queue, Exchange
import os

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis_password")

# Create Celery app with authentication
broker_url = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
result_backend = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

celery_app = Celery(
    'personal_stylist',
    broker=broker_url,
    backend=result_backend,
    broker_connection_retry_on_startup=True
)

# Configure queues
celery_app.conf.task_queues = (
    Queue('inference', Exchange('inference'), routing_key='inference'),
    Queue('metrics', Exchange('metrics'), routing_key='metrics'),
)

# Task routes
celery_app.conf.task_routes = {
    'generate_text': {'queue': 'inference'},
    'generate_recommendation': {'queue': 'inference'},
    'add_style_to_neo4j': {'queue': 'metrics'},
    'add_item_to_neo4j': {'queue': 'metrics'},
}

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    result_expires=3600,  # 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,
    broker_connection_retry=True,
    broker_connection_max_retries=None,  # Retry forever
    broker_pool_limit=None,  # Disable connection pooling
    redis_max_connections=None,  # Disable Redis connection limit
    broker_transport_options={
        'visibility_timeout': 43200,  # 12 hours
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
    }
)

# Include tasks from different modules
celery_app.autodiscover_tasks(['src.api']) 