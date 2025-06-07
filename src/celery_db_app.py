from celery import Celery
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Redis configuration from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery app for database worker only
app = Celery(
    'personal_stylist_db',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'src.workers.db_worker_safe'  # Use safe version to avoid SIGSEGV crashes
    ]
)

# Celery configuration
app.conf.update(
    # Task routing
    task_routes={
        'src.workers.db_worker.*': {'queue': 'database'},
    },
    
    # Result backend settings
    result_backend=CELERY_RESULT_BACKEND,
    result_expires=3600,  # 1 hour
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Queue settings
    task_default_queue='database',
    task_create_missing_queues=True,
    
    # Connection settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
)

# Import database worker tasks - using safe version to avoid SIGSEGV crashes
app.autodiscover_tasks(['src.workers.db_worker_safe'])

if __name__ == '__main__':
    app.start() 