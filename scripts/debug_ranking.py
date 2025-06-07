#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_ranking():
    """Debug the ranking of concepts to see where our targets are"""
    
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
            # Simulate the exact query from enhanced_text_processor
            query = """
            MATCH (c:Концепт)
            WHERE (c.name CONTAINS 'обувь' OR c.name STARTS WITH 'Концепт:обувь' OR EXISTS((c)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ|ЯВЛЯЕТСЯ_ОДЕЖДОЙ|ЯВЛЯЕТСЯ_АКСЕССУАРОМ]->()))
            AND (
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Тренд:2025']
                } OR
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Сезон:лето']
                } OR
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Эстетика:Pinterest']
                } OR
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Случай:офис']
                }
            )
            WITH DISTINCT c
            OPTIONAL MATCH (c)-[r:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(n)
            WHERE n.id IN ['Тренд:2025', 'Сезон:лето', 'Эстетика:Pinterest', 'Случай:офис']
            WITH c, collect(DISTINCT n.id) as intersections
            WITH c, intersections, size(intersections) as intersection_score
            ORDER BY intersection_score DESC, c.name ASC
            RETURN c.name as name,
                   c.id as id,
                   intersections,
                   intersection_score
            """
            
            result = session.run(query)
            concepts = result.data()
            
            print(f"Found {len(concepts)} concepts total")
            print("="*80)
            
            # Find our target concepts
            target_concepts = [
                'легкие кожаные лоферы',
                'бежевые балетки',
                'строгие ботинки-оксфорды'
            ]
            
            target_positions = {}
            
            for i, concept in enumerate(concepts, 1):
                name = concept['name']
                score = concept['intersection_score']
                intersections = concept['intersections']
                
                # Check if this is one of our targets
                is_target = any(target in name for target in target_concepts)
                marker = " ⭐ TARGET" if is_target else ""
                
                if is_target:
                    target_positions[name] = i
                
                print(f"{i:3d}. {name} (score: {score}){marker}")
                print(f"     Intersections: {intersections}")
                
                # Show top 15 and any targets
                if i <= 15 or is_target:
                    continue
                elif i == 16:
                    print("     ... (showing only top 15 and targets)")
                    break
            
            print("="*80)
            print("TARGET CONCEPT POSITIONS:")
            for target in target_concepts:
                found = False
                for name, position in target_positions.items():
                    if target in name:
                        print(f"  {target}: position {position} ({name})")
                        found = True
                        break
                if not found:
                    print(f"  {target}: NOT FOUND")
                    
    except Exception as e:
        logger.error(f"Error debugging ranking: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    debug_ranking() 