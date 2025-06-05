from neo4j import GraphDatabase
import os
import logging
from pathlib import Path
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"  # Make sure this matches your local Neo4j password

def run_cypher_statements(session, cypher_file):
    """Run multiple Cypher statements from a file"""
    with open(cypher_file, 'r') as f:
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
    
    # Run each statement
    for statement in statements:
        if statement.strip():  # Skip empty statements
            session.run(statement)
            logger.info(f"Executed statement: {statement[:100]}...")  # Log first 100 chars

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
            
            # Get paths
            nodes_path = data_dir / "new_nodes.csv"
            edges_path = data_dir / "new_edges.csv"
            clean_cypher_path = data_dir / "clean_and_transform.cypher"
            
            if not nodes_path.exists():
                raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
            if not edges_path.exists():
                raise FileNotFoundError(f"Edges file not found: {edges_path}")
            
            # Import nodes
            with open(nodes_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    create_node_with_label(session, row['label'], row['id'], row['name'])
            logger.info("Imported nodes from new_nodes.csv")
            
            # Import relationships
            with open(edges_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    create_relationship(session, row['start'], row['end'], row['relation'])
            logger.info("Imported relationships from new_edges.csv")
            
            # Run cleaning and transformation
            if clean_cypher_path.exists():
                run_cypher_statements(session, clean_cypher_path)
                logger.info("Executed cleaning and transformation commands")
            else:
                logger.error(f"Clean cypher file not found: {clean_cypher_path}")
            
            logger.info("Successfully imported and cleaned all data in Neo4j")
    except Exception as e:
        logger.error(f"Error importing CSV data: {e}")
        raise

def init_neo4j():
    """Initialize Neo4j database with data"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Use the Neo4j directory that's known to work
        data_dir = Path(__file__).resolve().parent.parent / "src" / "api" / "db" / "neo4j"
        
        logger.info(f"Using Neo4j directory: {data_dir}")
        
        # Verify directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Neo4j directory not found: {data_dir}")
        
        # Import data into Neo4j
        import_csv_data(driver, data_dir)
        
        logger.info("Neo4j database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Neo4j: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    init_neo4j() 