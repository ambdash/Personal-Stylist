#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_shoe_relations():
    """Add ЯВЛЯЕТСЯ_ОБУВЬЮ relations to shoe concepts"""
    
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123'))
    
    try:
        with driver.session() as session:
            # Create or get the Обувь category node
            session.run("""
                MERGE (category:Категория {id: 'Категория:обувь', name: 'обувь'})
            """)
            
            # List of shoe concepts that need ЯВЛЯЕТСЯ_ОБУВЬЮ relations
            shoe_concepts = [
                'Концепт:легкие кожаные лоферы',
                'Концепт:бежевые балетки',
                'Концепт:строгие ботинки-оксфорды'
            ]
            
            for concept_id in shoe_concepts:
                # Add ЯВЛЯЕТСЯ_ОБУВЬЮ relation
                result = session.run("""
                    MATCH (concept:Концепт {id: $concept_id})
                    MATCH (category:Категория {id: 'Категория:обувь'})
                    MERGE (concept)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ]->(category)
                    RETURN concept.name as concept_name
                """, concept_id=concept_id)
                
                concept_data = result.data()
                if concept_data:
                    concept_name = concept_data[0]['concept_name']
                    logger.info(f"Added ЯВЛЯЕТСЯ_ОБУВЬЮ relation for: {concept_name}")
                else:
                    logger.warning(f"Concept not found: {concept_id}")
            
            # Verify the relations were added
            logger.info("\nVerifying relations:")
            for concept_id in shoe_concepts:
                result = session.run("""
                    MATCH (concept:Концепт {id: $concept_id})
                    OPTIONAL MATCH (concept)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ]->(category)
                    RETURN concept.name as concept_name, 
                           category.name as category_name,
                           EXISTS((concept)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ]->()) as has_shoe_relation
                """, concept_id=concept_id)
                
                data = result.data()
                if data:
                    concept_data = data[0]
                    logger.info(f"  {concept_data['concept_name']}: has_shoe_relation = {concept_data['has_shoe_relation']}")
                    if concept_data['category_name']:
                        logger.info(f"    -> {concept_data['category_name']}")
                        
    except Exception as e:
        logger.error(f"Error adding shoe relations: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    add_shoe_relations() 