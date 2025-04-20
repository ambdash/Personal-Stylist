from fastapi import FastAPI
from src.api.routes import (
    inference, 
    monitoring, 
    database, 
    cache, 
    analytics
)
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from fastapi.responses import JSONResponse

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

if __name__ == "__main__":
    # Explicitly bind to all interfaces
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Bind to all interfaces
        port=8000, 
        log_level="debug",
        access_log=True,
        proxy_headers=True
    )

