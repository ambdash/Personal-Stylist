import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.celery_db_app import app

logger = logging.getLogger(__name__)

@app.task(name='db_worker_minimal.health_check')
def health_check() -> Dict[str, Any]:
    """Minimal health check without Neo4j"""
    try:
        return {
            "success": True,
            "message": "Minimal worker is healthy",
            "worker": "db_worker_minimal"
        }
    except Exception as e:
        logger.error(f"Minimal health check failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.task(name='db_worker_minimal.test_neo4j')
def test_neo4j() -> Dict[str, Any]:
    """Test Neo4j import and connection"""
    try:
        # Try importing Neo4j
        from neo4j import GraphDatabase
        
        # Try connecting
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
        
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            
        driver.close()
        
        return {
            "success": True,
            "message": "Neo4j connection successful",
            "test_value": test_value
        }
        
    except Exception as e:
        logger.error(f"Neo4j test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 