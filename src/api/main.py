from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from src.api.db.neo4j.config import neo4j
from .routes.db_operations import router as db_router
from .routes.inference import router as inference_router
from .routes.unified_inference import router as unified_inference_router
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from celery.result import AsyncResult
import redis
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", "redis_password"),
    decode_responses=True
)

# Initialize FastAPI app
app = FastAPI(
    title="Personal Stylist API",
    description="API for personal stylist bot with ML inference and database operations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(db_router)
app.include_router(inference_router)
app.include_router(unified_inference_router)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "Personal Stylist API is working"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Neo4j connection
        neo4j.execute_query("RETURN 1")
        
        # Check Redis connection
        redis_client.ping()
        
        return {
            "status": "healthy",
            "services": {
                "neo4j": "connected",
                "redis": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a Celery task"""
    try:
        task_result = AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None
        }
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

