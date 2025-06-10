from celery import Celery
import os
from pathlib import Path
import sys
import multiprocessing

# Set multiprocessing start method to spawn to avoid CUDA fork issues
# Only set if not already set to avoid conflicts
try:
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, continue
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure Celery for inference workers
app = Celery('personal_stylist_inference')

# Configure broker and backend
app.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Use solo pool for GPU workers to avoid multiprocessing issues
    worker_pool='solo',
    
    task_routes={
        'inference_worker.*': {'queue': 'inference'},
    },
    task_default_queue='inference',
    task_default_exchange='inference',
    task_default_exchange_type='direct',
    task_default_routing_key='inference',
)

# Import inference worker tasks
app.autodiscover_tasks(['src.workers.inference_worker'])

if __name__ == '__main__':
    app.start() 