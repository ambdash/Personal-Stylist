#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

def add_jeans_concepts():
    """Add jeans concepts for testing specific item filtering"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Добавляем концепты джинсов
            concepts_to_add = [
                {
                    "id": "Концепт:классические джинсы с высокой посадкой",
                    "name": "классические джинсы с высокой посадкой",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "одежда"),
                        ("ПОДХОДИТ_ДЛЯ", "офис"),
                        ("ПОДХОДИТ_ДЛЯ", "работа"),
                        ("ОТНОСИТСЯ_К", "деловой стиль")
                    ]
                },
                {
                    "id": "Концепт:темные джинсы скинни",
                    "name": "темные джинсы скинни",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "одежда"),
                        ("ПОДХОДИТ_ДЛЯ", "офис"),
                        ("ПОДХОДИТ_ДЛЯ", "деловая встреча"),
                        ("ОТНОСИТСЯ_К", "современный стиль")
                    ]
                },
                {
                    "id": "Концепт:прямые джинсы",
                    "name": "прямые джинсы",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "одежда"),
                        ("ПОДХОДИТ_ДЛЯ", "офис"),
                        ("ПОДХОДИТ_ДЛЯ", "повседневная носка"),
                        ("ОТНОСИТСЯ_К", "casual")
                    ]
                },
                {
                    "id": "Концепт:джинсы с завышенной талией",
                    "name": "джинсы с завышенной талией",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "одежда"),
                        ("ПОДХОДИТ_ДЛЯ", "офис"),
                        ("В_СЕЗОНЕ", "осень"),
                        ("В_СЕЗОНЕ", "зима"),
                        ("ОТНОСИТСЯ_К", "винтаж")
                    ]
                },
                {
                    "id": "Концепт:офисные джинсы",
                    "name": "офисные джинсы",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "одежда"),
                        ("ПОДХОДИТ_ДЛЯ", "офис"),
                        ("ПОДХОДИТ_ДЛЯ", "деловая встреча"),
                        ("ОТНОСИТСЯ_К", "деловой стиль"),
                        ("ОТНОСИТСЯ_К", "smart casual")
                    ]
                }
            ]
            
            for concept in concepts_to_add:
                # Создаем концепт
                session.run(
                    "MERGE (c:Концепт {id: $id, name: $name})",
                    id=concept["id"],
                    name=concept["name"]
                )
                logger.info(f"Added concept: {concept['name']}")
                
                # Добавляем связи
                for relation_type, target_name in concept["relations"]:
                    # Создаем целевой узел если его нет
                    if relation_type == "ЯВЛЯЕТСЯ_ОДЕЖДОЙ":
                        session.run(
                            "MERGE (t:Одежда {name: $target_name})",
                            target_name=target_name
                        )
                        session.run(
                            "MATCH (c:Концепт {id: $concept_id}), (t:Одежда {name: $target_name}) "
                            "MERGE (c)-[:ЯВЛЯЕТСЯ_ОДЕЖДОЙ]->(t)",
                            concept_id=concept["id"],
                            target_name=target_name
                        )
                    elif relation_type == "ПОДХОДИТ_ДЛЯ":
                        session.run(
                            "MERGE (t:Случай {name: $target_name})",
                            target_name=target_name
                        )
                        session.run(
                            "MATCH (c:Концепт {id: $concept_id}), (t:Случай {name: $target_name}) "
                            "MERGE (c)-[:ПОДХОДИТ_ДЛЯ]->(t)",
                            concept_id=concept["id"],
                            target_name=target_name
                        )
                    elif relation_type == "ОТНОСИТСЯ_К":
                        session.run(
                            "MERGE (t:Эстетика {name: $target_name})",
                            target_name=target_name
                        )
                        session.run(
                            "MATCH (c:Концепт {id: $concept_id}), (t:Эстетика {name: $target_name}) "
                            "MERGE (c)-[:ОТНОСИТСЯ_К]->(t)",
                            concept_id=concept["id"],
                            target_name=target_name
                        )
                    elif relation_type == "В_СЕЗОНЕ":
                        session.run(
                            "MERGE (t:Сезон {name: $target_name})",
                            target_name=target_name
                        )
                        session.run(
                            "MATCH (c:Концепт {id: $concept_id}), (t:Сезон {name: $target_name}) "
                            "MERGE (c)-[:В_СЕЗОНЕ]->(t)",
                            concept_id=concept["id"],
                            target_name=target_name
                        )
                    
                    logger.info(f"  Added relation: {relation_type} -> {target_name}")
            
            logger.info("✅ All jeans concepts added successfully!")
            
    except Exception as e:
        logger.error(f"Error adding jeans concepts: {str(e)}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    add_jeans_concepts() 