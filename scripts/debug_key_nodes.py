#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_key_nodes():
    text = "Какую летнюю обувь лучше выбрать для офиса в 2025 году?"
    
    print(f"Original text: {text}")
    print("="*60)
    
    # Connect to database
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
            # Check what key nodes we should find
            key_node_types = ['Сезон', 'Случай', 'Тренд', 'Эстетика', 'Погода']
            
            print("EXPECTED KEY NODES FROM TEXT:")
            print("- лето (from 'летнюю')")
            print("- офис (from 'офиса')")  
            print("- 2025 (from '2025')")
            print()
            
            # Check if these nodes exist
            expected_nodes = [
                ('Сезон', 'лето'),
                ('Случай', 'офис'),
                ('Тренд', '2025')
            ]
            
            print("CHECKING IF EXPECTED NODES EXIST:")
            for node_type, node_name in expected_nodes:
                result = session.run(f'''
                    MATCH (n:{node_type})
                    WHERE n.name = $name
                    RETURN n.name as name, n.id as id
                ''', name=node_name)
                
                nodes = result.data()
                if nodes:
                    print(f"✓ Found {node_type}:{node_name} - {nodes[0]['id']}")
                else:
                    print(f"✗ Missing {node_type}:{node_name}")
            
            print("\n" + "="*60)
            print("CHECKING RELATIONS TO TARGET CONCEPTS:")
            print("="*60)
            
            # Check if our target concepts are related to these key nodes
            target_concepts = [
                'Концепт:легкие кожаные лоферы',
                'Концепт:бежевые балетки', 
                'Концепт:строгие ботинки-оксфорды'
            ]
            
            key_node_ids = ['Сезон:лето', 'Случай:офис', 'Тренд:2025']
            
            for concept_id in target_concepts:
                print(f"\nConcept: {concept_id}")
                
                # Check relations to key nodes
                result = session.run('''
                    MATCH (c:Концепт {id: $concept_id})-[r]->(n)
                    WHERE n.id IN $key_node_ids
                    RETURN type(r) as rel_type, n.name as target_name, n.id as target_id
                    ORDER BY rel_type, target_name
                ''', concept_id=concept_id, key_node_ids=key_node_ids)
                
                relations = result.data()
                if relations:
                    print("  Relations to key nodes:")
                    for rel in relations:
                        print(f"    {rel['rel_type']} -> {rel['target_name']} ({rel['target_id']})")
                else:
                    print("  No relations to key nodes found")
                
                # Check all relations
                all_rels_result = session.run('''
                    MATCH (c:Концепт {id: $concept_id})-[r]->(n)
                    RETURN type(r) as rel_type, n.name as target_name, n.id as target_id
                    ORDER BY rel_type, target_name
                ''', concept_id=concept_id)
                
                all_relations = all_rels_result.data()
                if all_relations:
                    print("  All relations:")
                    for rel in all_relations:
                        print(f"    {rel['rel_type']} -> {rel['target_name']} ({rel['target_id']})")
                        
    finally:
        driver.close()

if __name__ == "__main__":
    debug_key_nodes() 