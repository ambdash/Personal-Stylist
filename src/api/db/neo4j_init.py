from neo4j import GraphDatabase
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def import_csv_data(driver, data_dir):
    """Import CSV data into Neo4j"""
    try:
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing data")
            
            # Import nodes
            nodes_path = data_dir / "filtered_nodes.csv"
            if not nodes_path.exists():
                logger.error(f"Nodes file not found: {nodes_path}")
                return
                
            session.run("""
                LOAD CSV WITH HEADERS FROM 'file:///filtered_nodes.csv' AS row
                MERGE (n:`${row.label}` {id: row.id})
                ON CREATE SET n.name = row.name
            """)
            logger.info("Imported nodes from filtered_nodes.csv")
            
            # Import relationships
            edges_path = data_dir / "filtered_edges.csv"
            if not edges_path.exists():
                logger.error(f"Edges file not found: {edges_path}")
                return
                
            session.run("""
                LOAD CSV WITH HEADERS FROM 'file:///filtered_edges.csv' AS row
                MATCH (a {id: row.start})
                MATCH (b {id: row.end})
                MERGE (a)-[r:`${row.relation}`]->(b)
            """)
            logger.info("Imported relationships from filtered_edges.csv")
            
            # Run any additional Cypher commands
            cypher_path = data_dir / "import.cypher"
            if cypher_path.exists():
                with open(cypher_path, 'r') as f:
                    cypher_commands = f.read()
                    session.run(cypher_commands)
                logger.info("Executed additional Cypher commands")
            
            logger.info("Successfully imported all data into Neo4j")
    except Exception as e:
        logger.error(f"Error importing CSV data: {e}")
        raise

def init_neo4j():
    """Initialize Neo4j database with data"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Use the mounted data directory
        data_dir = Path("/var/lib/neo4j/import")
        
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