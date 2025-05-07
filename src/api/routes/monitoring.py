from fastapi import APIRouter
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from typing import Dict
import time
from src.ml.inference.engine import InferenceEngine
from src.api.schemas.request_models import HealthResponse, MetricsResponse
import psutil
import os
from datetime import datetime

router = APIRouter(prefix="/v1/metrics", tags=["monitoring"])

START_TIME = datetime.now()

engine = InferenceEngine()

REQUEST_COUNT = Counter('request_count', 'Total request count')
LATENCY = Histogram('request_latency_seconds', 'Request latency')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')

def get_system_metrics() -> Dict:
    """Get system metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_memory_used": None,  # Add GPU metrics if needed
    }

def calculate_uptime() -> float:
    """Calculate API uptime in seconds"""
    return (datetime.now() - START_TIME).total_seconds()

def check_model_health() -> str:
    try:
        return "healthy"
    except Exception:
        return "unhealthy"

def check_db_health() -> str:
    try:
        return "healthy"
    except Exception:
        return "unhealthy"

def get_system_latency() -> float:
    return LATENCY.observe(time.time())

@router.get("/prometheus")
async def prometheus_metrics():
    """Endpoint for Prometheus metrics"""
    return generate_latest(REGISTRY)

@router.get("/health/detailed", response_model=HealthResponse)
async def detailed_health() -> HealthResponse:
    """Detailed health check including model and system status"""
    # Get model health status
    model_health = engine.health_check()
    
    # Get model statistics
    model_stats = engine.get_model_stats()
    
    # Prepare model status list
    models = []
    for model_name, stats in model_stats.items():
        models.append({
            "name": model_name,
            "status": model_health["models"].get(model_name, {}).get("status", "unknown"),
            "loaded": stats["loaded"],
            "last_used": stats["last_used"],
            "avg_inference_time": stats["total_time"] / stats["total_requests"] if stats["total_requests"] > 0 else None
        })
    
    total_requests = sum(stats["total_requests"] for stats in model_stats.values())
    total_time = sum(stats["total_time"] for stats in model_stats.values())
    avg_latency = total_time / total_requests if total_requests > 0 else 0
    
    return HealthResponse(
        status=model_health["status"],
        models=models,
        api_version="1.0",
        uptime=calculate_uptime(),
        total_requests=total_requests,
        avg_latency=avg_latency
    )

@router.get("/metrics/detailed", response_model=MetricsResponse)
async def detailed_metrics() -> MetricsResponse:
    """Get detailed metrics about model performance and system status"""
    model_stats = engine.get_model_stats()
    
    # Calculate metrics
    total_requests = sum(stats["total_requests"] for stats in model_stats.values())
    total_errors = sum(stats["errors"] for stats in model_stats.values())
    error_rate = total_errors / total_requests if total_requests > 0 else 0
    
    # Get requests per model
    requests_per_model = {
        model: stats["total_requests"]
        for model, stats in model_stats.items()
    }
    
    # Get model load times
    model_load_times = {
        model: stats.get("load_time", None)
        for model, stats in model_stats.items()
        if stats.get("loaded", False)
    }
    
    # Calculate average latency
    total_time = sum(stats["total_time"] for stats in model_stats.values())
    average_latency = total_time / total_requests if total_requests > 0 else 0
    
    return MetricsResponse(
        total_requests=total_requests,
        requests_per_model=requests_per_model,
        average_latency=average_latency,
        error_rate=error_rate,
        model_load_times=model_load_times
    ) 