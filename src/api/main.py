from fastapi import FastAPI
from src.api.routes import (
    inference, 
    monitoring, 
    database, 
    cache, 
    analytics,
    telegram
)
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from fastapi.responses import JSONResponse
from .db.neo4j_init.init_db import init_database
from src.api.services.kafka_consumer import KafkaMessageConsumer
from src.api.services.kafka_producer import KafkaMessageProducer
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Stylist API",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware with explicit hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "API is working"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# Include routes
app.include_router(inference.router)
app.include_router(monitoring.router)
app.include_router(database.router)
app.include_router(cache.router)
app.include_router(analytics.router)
app.include_router(telegram.router)

# Initialize Kafka consumer and handlers
kafka_consumer = KafkaMessageConsumer()

def handle_user_messages(message: Dict[str, Any]):
    """Handle user messages from Kafka"""
    logger.info(f"Processing user message: {message}")
    # Add analytics, store in database, etc.

def handle_bot_responses(message: Dict[str, Any]):
    """Handle bot responses from Kafka"""
    logger.info(f"Processing bot response: {message}")
    # Add analytics, store in database, etc.

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize Neo4j
        init_database()
        logger.info("Successfully initialized Neo4j database")

        # Start Kafka consumers
        kafka_consumer.start_consumer("user_messages", handle_user_messages)
        kafka_consumer.start_consumer("bot_responses", handle_bot_responses)
        logger.info("Successfully started Kafka consumers")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    kafka_consumer.stop_all()

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000, 
        log_level="debug",
        access_log=True,
        proxy_headers=True
    )

