#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_shoe_concepts():
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
            # Check if our specific shoe concepts exist
            result = session.run('''
                MATCH (c:Концепт) 
                WHERE c.name CONTAINS 'лоферы' OR c.name CONTAINS 'балетки' OR c.name CONTAINS 'оксфорды'
                RETURN c.name as name, c.id as id
                ORDER BY c.name
            ''')
            
            concepts = result.data()
            print(f'Found {len(concepts)} shoe concepts:')
            for concept in concepts:
                print(f'  - {concept["name"]} (id: {concept["id"]})')
            
            # Check their relations
            if concepts:
                print('\nRelations for these concepts:')
                for concept in concepts:
                    rel_result = session.run('''
                        MATCH (c:Концепт {id: $concept_id})-[r]->(n)
                        RETURN c.name as concept_name, type(r) as rel_type, n.name as target_name
                        ORDER BY rel_type, target_name
                    ''', concept_id=concept['id'])
                    
                    relations = rel_result.data()
                    print(f'\n{concept["name"]}:')
                    for rel in relations:
                        print(f'  {rel["rel_type"]} -> {rel["target_name"]}')
            
            # Also check what key nodes we have
            print('\n' + '='*50)
            print('KEY NODES IN DATABASE:')
            print('='*50)
            
            key_node_types = ['Сезон', 'Случай', 'Тренд', 'Эстетика', 'Погода']
            for node_type in key_node_types:
                result = session.run(f'MATCH (n:{node_type}) RETURN n.name as name ORDER BY n.name')
                nodes = [record['name'] for record in result]
                print(f'{node_type}: {", ".join(nodes)}')
                
    finally:
        driver.close()

if __name__ == "__main__":
    check_shoe_concepts() 