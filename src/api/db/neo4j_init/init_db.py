## STUB EXAMPLE OF USING NEO4J
from src.api.db.neo4j import db
import logging

logger = logging.getLogger(__name__)

def init_database():
    # Clear existing data
    clear_db = """
    MATCH (n)
    DETACH DELETE n
    """
    
    # Create constraints
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:FashionItem) REQUIRE i.item_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Style) REQUIRE s.name IS UNIQUE"
    ]
    
    # Load items
    load_items = """
    LOAD CSV WITH HEADERS FROM 'file:///items.csv' AS row
    CREATE (i:FashionItem {
        item_id: toInteger(row.item_id),
        name: row.name,
        category: row.category,
        style: row.style,
        color: row.color,
        season: split(row.season, ',')
    })
    """
    
    # Load style rules
    load_styles = """
    LOAD CSV WITH HEADERS FROM 'file:///style_rules.csv' AS row
    CREATE (s:Style {
        name: row.style,
        description: row.description,
        key_pieces: split(row.key_pieces, ',')
    })
    """
    
    create_compatibility = """
    LOAD CSV WITH HEADERS FROM 'file:///compatibility.csv' AS row
    MATCH (i1:FashionItem {item_id: toInteger(row.item1_id)})
    MATCH (i2:FashionItem {item_id: toInteger(row.item2_id)})
    CREATE (i1)-[r:COMPATIBLE_WITH {score: toFloat(row.compatibility_score)}]->(i2)
    """
    
    # Connect items to styles
    connect_styles = """
    MATCH (i:FashionItem)
    MATCH (s:Style {name: i.style})
    CREATE (i)-[:HAS_STYLE]->(s)
    """
    
    queries = [clear_db] + constraints + [load_items, load_styles, create_compatibility, connect_styles]
    
    for query in queries:
        try:
            db.execute_query(query)
            logger.info(f"Successfully executed query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            raise 