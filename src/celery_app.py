from celery import Celery
from kombu import Queue, Exchange
import os

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

# Celery configuration
app = Celery(
    "fashion_stylist",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
)

# Configure queues
app.conf.task_queues = (
    Queue("inference", Exchange("inference"), routing_key="inference"),
    Queue("metrics", Exchange("metrics"), routing_key="metrics"),
)

# Task routes
app.conf.task_routes = {
    "src.ml.tasks.generate_recommendation": {"queue": "inference"},
    "src.monitoring.tasks.track_metric": {"queue": "metrics"},
}

# Other Celery settings
app.conf.update(
    result_expires=3600,  # Results expire in 1 hour
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Moscow",
    enable_utc=True,
) 