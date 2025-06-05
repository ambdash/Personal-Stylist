from celery import Celery
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis_password")

celery_app = Celery(
    'inference',
    broker=f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0',
    backend=f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'
)

@celery_app.task
def generate_recommendation(prompt: str) -> str:
    # TODO: Implement actual recommendation generation
    return f"Рекомендация для запроса: {prompt}" 