from fastapi import APIRouter
from prometheus_client import Counter, Histogram, generate_latest
from typing import Dict
import time

router = APIRouter(prefix="/v1/metrics", tags=["monitoring"])

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total request count')
LATENCY = Histogram('request_latency_seconds', 'Request latency')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')

def check_model_health() -> str:
    try:
        # Add actual model health check logic here
        return "healthy"
    except Exception:
        return "unhealthy"

def check_db_health() -> str:
    try:
        # Add actual database health check logic here
        return "healthy"
    except Exception:
        return "unhealthy"

def get_system_latency() -> float:
    return LATENCY.observe(time.time())

@router.get("/prometheus")
async def prometheus_metrics():
    """Endpoint for Prometheus metrics"""
    return generate_latest()

@router.get("/health/detailed")
async def detailed_health() -> Dict:
    """Detailed health check including model and database status"""
    health_status = {
        "status": "healthy",
        "components": {
            "model": check_model_health(),
            "database": check_db_health(),
            "api": "healthy"
        },
        "latency": get_system_latency()
    }
    return health_status 