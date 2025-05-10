import logging
from .config import Neo4jConnection
from .queries import Neo4jQueries

logger = logging.getLogger(__name__)

def init_database():
    """Initialize Neo4j database with sample data"""
    connection = Neo4jConnection()
    queries = Neo4jQueries()
    
    try:
        with connection.get_session() as session:
            # Create sample styles
            session.run("""
                CREATE (s1:Style {name: 'Y2K', description: '2000s fashion style'})
                CREATE (s2:Style {name: 'Grunge', description: '90s grunge style'})
                CREATE (s3:Style {name: 'Minimalist', description: 'Minimalist fashion'})
            """)
            
            # Create sample items
            session.run("""
                CREATE (i1:Item {name: 'Cargo Pants', type: 'bottoms'})
                CREATE (i2:Item {name: 'Crop Top', type: 'tops'})
                CREATE (i3:Item {name: 'Platform Shoes', type: 'shoes'})
            """)
            
            # Create relationships
            session.run("""
                MATCH (s:Style {name: 'Y2K'}), (i:Item {name: 'Cargo Pants'})
                CREATE (s)-[:INCLUDES {season: 'all'}]->(i)
            """)
            
            logger.info("Successfully initialized Neo4j database with sample data")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    finally:
        connection.close() 