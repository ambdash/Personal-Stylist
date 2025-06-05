from celery import Celery
from kombu import Exchange, Queue
import os
from dotenv import load_dotenv

load_dotenv()

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis_password")

# Create Celery app with proper Redis URL format
redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

# Define exchanges
db_exchange = Exchange('db', type='direct')
llm_exchange = Exchange('llm', type='direct')

# Define queues
task_queues = [
    Queue('db', db_exchange, routing_key='db'),
    Queue('llm', llm_exchange, routing_key='llm'),
]

# Initialize Celery app
celery_app = Celery(
    "personal_stylist",
    broker=redis_url,
    backend=redis_url,
    include=[
        'src.api.tasks.db_tasks',    # Database tasks
        'src.api.tasks.llm_tasks'    # LLM tasks
    ]
)

# Configure Celery
celery_app.conf.update(
    task_queues=task_queues,
    task_default_queue='db',
    task_default_exchange='db',
    task_default_routing_key='db',
    task_routes={
        # Database tasks
        'src.api.tasks.db_tasks.*': {'queue': 'db'},
        'create_node': {'queue': 'db'},
        'update_node': {'queue': 'db'},
        'delete_node': {'queue': 'db'},
        'create_relation': {'queue': 'db'},
        'search_nodes': {'queue': 'db'},
        
        # LLM tasks
        'src.api.tasks.llm_tasks.*': {'queue': 'llm'},
        'inference': {'queue': 'llm'},
        'rag_inference': {'queue': 'llm'},
    },
    task_annotations={
        'src.api.tasks.db_tasks.*': {'rate_limit': '100/m'},
        'src.api.tasks.llm_tasks.*': {'rate_limit': '30/m'},
    },
    worker_prefetch_multiplier=1,  # Prevent worker from prefetching too many tasks
    task_acks_late=True,  # Only acknowledge task after it's completed
    task_reject_on_worker_lost=True,  # Reject task if worker dies
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_max_tasks_per_child=200,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None  # Keep retrying indefinitely
)

# Export the app
__all__ = ['celery_app'] 