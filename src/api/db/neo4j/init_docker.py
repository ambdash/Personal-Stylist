try:
    from neo4j import GraphDatabase
except ImportError:
    from neo4j.v1 import GraphDatabase

import os
import logging
from pathlib import Path
import csv
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            try:
                session.run(statement)
                logger.info(f"Executed statement: {statement[:100]}...")
            except Exception as e:
                logger.error(f"Error executing statement: {statement[:100]}... Error: {str(e)}")
                raise

def create_node_with_label(session, label, node_id, name):
    """Create a node with the given label"""
    query = f"""
    CREATE (n:{label} {{id: $id, name: $name}})
    RETURN n
    """
    try:
        session.run(query, {"id": node_id, "name": name})
    except Exception as e:
        logger.error(f"Error creating node {name} with label {label}: {str(e)}")
        raise

def create_relationship(session, start_id, end_id, rel_type):
    """Create a relationship between nodes"""
    query = f"""
    MATCH (a {{id: $start_id}})
    MATCH (b {{id: $end_id}})
    CREATE (a)-[r:{rel_type}]->(b)
    RETURN r
    """
    try:
        session.run(query, {"start_id": start_id, "end_id": end_id})
    except Exception as e:
        logger.error(f"Error creating relationship between {start_id} and {end_id}: {str(e)}")
        raise

def import_csv_data(driver, data_dir):
    """Import CSV data into Neo4j"""
    try:
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing data")
            
            # Get paths
            nodes_path = Path(data_dir) / "new_nodes.csv"
            edges_path = Path(data_dir) / "new_edges.csv"
            clean_cypher_path = Path(data_dir) / "clean_and_transform.cypher"
            
            logger.info(f"Using paths: nodes={nodes_path}, edges={edges_path}, cypher={clean_cypher_path}")
            
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
                logger.warning(f"Clean cypher file not found: {clean_cypher_path}")
            
            logger.info("Successfully imported and cleaned all data in Neo4j")
    except Exception as e:
        logger.error(f"Error importing CSV data: {e}")
        raise

def init_neo4j():
    """Initialize Neo4j database with data"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    max_retries = 10
    retry_delay = 5
    
    # Get the current directory where the script is located
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to connect to Neo4j at {uri}")
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test the connection
            with driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Successfully connected to Neo4j")
            
            # Use the current directory for data files
            logger.info(f"Using data directory: {current_dir}")
            
            # Import data into Neo4j
            import_csv_data(driver, current_dir)
            
            logger.info("Neo4j database initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Failed to initialize Neo4j after all retries")
                raise
        finally:
            if 'driver' in locals():
                driver.close()

if __name__ == "__main__":
    init_neo4j() 