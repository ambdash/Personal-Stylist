from neo4j import GraphDatabase
import os
import logging
from pathlib import Path
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local Neo4j connection using environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def run_cypher_statements(session, cypher_file):
    """Run multiple Cypher statements from a file"""
    with open(cypher_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split the content into individual statements
    statements = []
    current_statement = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):  # Skip empty lines and comments
            continue
        
        current_statement.append(line)
        if line.endswith(';'):  # Statement end
            statements.append(' '.join(current_statement))
            current_statement = []
    
    # Add any remaining statement
    if current_statement:
        statements.append(' '.join(current_statement))
    
    # Run each statement
    for statement in statements:
        if statement.strip():  # Skip empty statements
            try:
                session.run(statement)
                logger.info(f"Executed statement: {statement[:100]}...")  # Log first 100 chars
            except Exception as e:
                logger.error(f"Error executing statement: {statement[:100]}... Error: {e}")

def create_node_with_label(session, label, node_id, name):
    """Create a node with the given label"""
    query = f"""
    CREATE (n:{label} {{id: $id, name: $name}})
    RETURN n
    """
    session.run(query, {"id": node_id, "name": name})

def create_relationship(session, start_id, end_id, rel_type):
    """Create a relationship between nodes"""
    query = f"""
    MATCH (a {{id: $start_id}})
    MATCH (b {{id: $end_id}})
    CREATE (a)-[r:{rel_type}]->(b)
    RETURN r
    """
    session.run(query, {"start_id": start_id, "end_id": end_id})

def import_csv_data(driver, data_dir):
    """Import CSV data into Neo4j"""
    try:
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing data")
            
            # Get paths - use new_nodes.csv and new_edges.csv
            nodes_path = data_dir / "new_nodes.csv"
            edges_path = data_dir / "new_edges.csv"
            clean_cypher_path = data_dir / "clean_and_transform.cypher"
            
            if not nodes_path.exists():
                raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
            if not edges_path.exists():
                raise FileNotFoundError(f"Edges file not found: {edges_path}")
            
            logger.info(f"Found data files:")
            logger.info(f"  - Nodes: {nodes_path}")
            logger.info(f"  - Edges: {edges_path}")
            
            # Import nodes
            node_count = 0
            with open(nodes_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    create_node_with_label(session, row['label'], row['id'], row['name'])
                    node_count += 1
                    if node_count % 1000 == 0:  # Progress indicator every 1000 nodes
                        logger.info(f"Imported {node_count} nodes...")
            logger.info(f"Imported {node_count} nodes from new_nodes.csv")
            
            # Import relationships
            edge_count = 0
            with open(edges_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    create_relationship(session, row['start'], row['end'], row['relation'])
                    edge_count += 1
                    if edge_count % 1000 == 0:  # Progress indicator every 1000 relationships
                        logger.info(f"Imported {edge_count} relationships...")
            logger.info(f"Imported {edge_count} relationships from new_edges.csv")
            
            # Run cleaning and transformation
            if clean_cypher_path.exists():
                logger.info("Running clean_and_transform.cypher script...")
                run_cypher_statements(session, clean_cypher_path)
                logger.info("Executed cleaning and transformation commands")
            else:
                logger.warning(f"Clean cypher file not found: {clean_cypher_path}")
                # Create basic indexes anyway
                basic_indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (n:Одежда) ON (n.id);",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Аксессуар) ON (n.id);",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Обувь) ON (n.id);",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Цвет) ON (n.id);",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Материал) ON (n.id);",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Концепт) ON (n.id);",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Эстетика) ON (n.id);"
                ]
                for index_query in basic_indexes:
                    session.run(index_query)
                logger.info("Created basic indexes")
            
            logger.info(f"Successfully imported and cleaned all data in Neo4j: {node_count} nodes, {edge_count} relationships")
    except Exception as e:
        logger.error(f"Error importing CSV data: {e}")
        raise

def init_neo4j():
    """Initialize Neo4j database with data"""
    try:
        logger.info(f"Connecting to Neo4j at {NEO4J_URI} with user {NEO4J_USER}")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Test connection first
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("Neo4j connection successful")
        
        # Use the Neo4j directory that's known to work
        data_dir = Path(__file__).resolve().parent.parent / "src" / "api" / "db" / "neo4j"
        
        logger.info(f"Using Neo4j directory: {data_dir}")
        
        # Verify directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Neo4j directory not found: {data_dir}")
        
        # Check for required files
        nodes_file = data_dir / "new_nodes.csv"
        edges_file = data_dir / "new_edges.csv"
        
        if not nodes_file.exists():
            raise FileNotFoundError(f"Required file not found: {nodes_file}")
        if not edges_file.exists():
            raise FileNotFoundError(f"Required file not found: {edges_file}")
        
        # Import data into Neo4j
        import_csv_data(driver, data_dir)
        
        logger.info("Neo4j database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing Neo4j: {e}")
        raise
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    init_neo4j() 