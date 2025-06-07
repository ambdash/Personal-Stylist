from src.api.db.neo4j.service import Neo4jService
from src.api.db.neo4j.config import Neo4jConnection
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_neo4j_connection():
    """Test Neo4j connection and data access"""
    try:
        # Initialize service
        neo4j_service = Neo4jService(Neo4jConnection())
        
        # Test node type retrieval
        logger.info("Testing node type retrieval...")
        for node_type in ["ОДЕЖДА", "ОБУВЬ", "АКСЕССУАРЫ", "СТИЛЬ", "ЦВЕТ"]:
            nodes = await neo4j_service.get_nodes_by_type(node_type)
            logger.info(f"Found {len(nodes)} nodes of type {node_type}")
            if nodes:
                logger.info(f"Sample nodes: {nodes[:3]}")
        
        # Test word search
        logger.info("\nTesting word search...")
        test_words = ["джинсы", "кроссовки", "шарф"]
        for word in test_words:
            nodes = await neo4j_service.search_nodes_by_word(word)
            logger.info(f"Search for '{word}' found {len(nodes)} nodes")
            if nodes:
                logger.info(f"Results: {nodes}")
        
        logger.info("\nNeo4j connection and data access test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing Neo4j connection: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_neo4j_connection()) 