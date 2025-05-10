from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from .db.neo4j_config import neo4j_connection
from .routes.triple_routes import router as triple_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Stylist API",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")
async def root():
    return {"message": "API is working"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Neo4j connection
        with neo4j_connection.get_session() as session:
            session.run("RETURN 1")
        return {
            "status": "healthy",
            "neo4j": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

# Include routes
app.include_router(triple_router)

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

