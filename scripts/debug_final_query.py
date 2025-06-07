#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_final_query():
    # Connect to database
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
            # Simulate the query that the enhanced processor would run
            # Based on the key nodes found: лето, офис, 2025, Pinterest
            
            query = """
            MATCH (c:Концепт)
            WHERE (c.name CONTAINS 'обувь' OR c.name STARTS WITH 'Концепт:обувь' OR EXISTS((c)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ|ЯВЛЯЕТСЯ_ОДЕЖДОЙ|ЯВЛЯЕТСЯ_АКСЕССУАРОМ]->()))
            AND (
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Случай:офис']
                } OR
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Тренд:2025']
                } OR
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Эстетика:Pinterest']
                } OR
                EXISTS {
                    MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                    WHERE target_node.id IN ['Сезон:лето']
                }
            )
            WITH DISTINCT c
            OPTIONAL MATCH (c)-[r:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(n)
            WITH c, collect({type: type(r), node: n}) as relations
            RETURN c.name as name,
                   c.id as id,
                   relations,
                   2.0 as score
            ORDER BY score DESC, c.name
            """
            
            result = session.run(query)
            concepts = result.data()
            
            print(f"Found {len(concepts)} concepts matching the query:")
            print("="*60)
            
            for i, concept in enumerate(concepts, 1):
                print(f"{i}. {concept['name']} (id: {concept['id']})")
                
                # Calculate intersection score
                relations = concept.get('relations', [])
                key_node_ids = ['Случай:офис', 'Тренд:2025', 'Эстетика:Pinterest', 'Сезон:лето']
                
                concept_related_ids = set()
                for rel in relations:
                    node = rel.get('node')
                    if node and 'id' in node:
                        concept_related_ids.add(node['id'])
                
                intersection_score = sum(1 for node_id in key_node_ids if node_id in concept_related_ids)
                print(f"   Intersection score: {intersection_score}")
                
                # Show relations
                if relations:
                    print("   Relations:")
                    for rel in relations:
                        node = rel.get('node')
                        if node and 'name' in node:
                            print(f"     {rel['type']} -> {node['name']} ({node.get('id', 'no id')})")
                print()
                
            print("="*60)
            print("CHECKING TARGET CONCEPTS SPECIFICALLY:")
            print("="*60)
            
            # Check our target concepts specifically
            target_concepts = [
                'Концепт:легкие кожаные лоферы',
                'Концепт:бежевые балетки', 
                'Концепт:строгие ботинки-оксфорды'
            ]
            
            for concept_id in target_concepts:
                result = session.run('''
                    MATCH (c:Концепт {id: $concept_id})
                    OPTIONAL MATCH (c)-[r:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(n)
                    WITH c, collect({type: type(r), node: n}) as relations
                    RETURN c.name as name,
                           c.id as id,
                           relations
                ''', concept_id=concept_id)
                
                concept_data = result.data()
                if concept_data:
                    concept = concept_data[0]
                    print(f"Target concept: {concept['name']}")
                    
                    # Check if it matches the item type condition
                    name = concept['name']
                    matches_item_type = ('обувь' in name or 
                                       name.startswith('Концепт:обувь') or 
                                       any(rel['type'] in ['ЯВЛЯЕТСЯ_ОБУВЬЮ', 'ЯВЛЯЕТСЯ_ОДЕЖДОЙ', 'ЯВЛЯЕТСЯ_АКСЕССУАРОМ'] 
                                           for rel in concept['relations']))
                    print(f"  Matches item type condition: {matches_item_type}")
                    
                    # Check key node intersections
                    key_node_ids = ['Случай:офис', 'Тренд:2025', 'Эстетика:Pinterest', 'Сезон:лето']
                    concept_related_ids = set()
                    for rel in concept['relations']:
                        node = rel.get('node')
                        if node and 'id' in node:
                            concept_related_ids.add(node['id'])
                    
                    intersections = [node_id for node_id in key_node_ids if node_id in concept_related_ids]
                    print(f"  Key node intersections: {intersections}")
                    print(f"  Intersection score: {len(intersections)}")
                    
                    if matches_item_type and intersections:
                        print("  ✓ Should be included in results")
                    else:
                        print("  ✗ Would be filtered out")
                        if not matches_item_type:
                            print("    - Doesn't match item type condition")
                        if not intersections:
                            print("    - No key node intersections")
                    print()
                else:
                    print(f"Target concept {concept_id} not found in database")
                    
    finally:
        driver.close()

if __name__ == "__main__":
    debug_final_query() 