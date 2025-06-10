#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

def add_dress_concepts():
    """Add dress concepts for testing specific item filtering"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Добавляем концепты платьев
            concepts_to_add = [
                {
                    "id": "Концепт:романтическое платье",
                    "name": "романтическое платье",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "Одежда:платье"),
                        ("ПОДХОДИТ_ДЛЯ", "Эстетика:романтика"),
                        ("ПОДХОДИТ_ДЛЯ", "Случай:ужин"),
                        ("СДЕЛАН_ИЗ", "Материал:шифон"),
                        ("ОКРАШЕН_В", "Цвет:розовый")
                    ]
                },
                {
                    "id": "Концепт:кружевное платье",
                    "name": "кружевное платье",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "Одежда:платье"),
                        ("ПОДХОДИТ_ДЛЯ", "Эстетика:романтика"),
                        ("ПОДХОДИТ_ДЛЯ", "Случай:ужин"),
                        ("СДЕЛАН_ИЗ", "Материал:кружево"),
                        ("ОКРАШЕН_В", "Цвет:белый")
                    ]
                },
                {
                    "id": "Концепт:вечернее платье",
                    "name": "вечернее платье",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "Одежда:платье"),
                        ("ПОДХОДИТ_ДЛЯ", "Случай:ужин"),
                        ("ПОДХОДИТ_ДЛЯ", "Эстетика:элегантность"),
                        ("СДЕЛАН_ИЗ", "Материал:шелк"),
                        ("ОКРАШЕН_В", "Цвет:черный")
                    ]
                },
                {
                    "id": "Концепт:летнее платье",
                    "name": "летнее платье",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "Одежда:платье"),
                        ("В_СЕЗОНЕ", "Сезон:лето"),
                        ("СДЕЛАН_ИЗ", "Материал:хлопок"),
                        ("ОКРАШЕН_В", "Цвет:голубой")
                    ]
                },
                {
                    "id": "Концепт:деловое платье",
                    "name": "деловое платье",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОДЕЖДОЙ", "Одежда:платье"),
                        ("ПОДХОДИТ_ДЛЯ", "Случай:офис"),
                        ("ПОДХОДИТ_ДЛЯ", "Эстетика:деловой стиль"),
                        ("СДЕЛАН_ИЗ", "Материал:шерсть"),
                        ("ОКРАШЕН_В", "Цвет:серый")
                    ]
                }
            ]
            
            # Создаем недостающие узлы
            nodes_to_create = [
                ("Одежда:платье", "платье", "Одежда"),
                ("Материал:шифон", "шифон", "Материал"),
                ("Материал:кружево", "кружево", "Материал"),
                ("Материал:шелк", "шелк", "Материал"),
                ("Материал:хлопок", "хлопок", "Материал"),
                ("Материал:шерсть", "шерсть", "Материал"),
                ("Цвет:розовый", "розовый", "Цвет"),
                ("Цвет:белый", "белый", "Цвет"),
                ("Цвет:черный", "черный", "Цвет"),
                ("Цвет:голубой", "голубой", "Цвет"),
                ("Цвет:серый", "серый", "Цвет"),
                ("Эстетика:элегантность", "элегантность", "Эстетика")
            ]
            
            for node_id, name, label in nodes_to_create:
                session.run(f"""
                    MERGE (n:{label} {{id: $id}})
                    ON CREATE SET n.name = $name
                """, id=node_id, name=name)
                logger.info(f"Created/ensured node: {node_id}")
            
            # Создаем концепты и их связи
            for concept in concepts_to_add:
                # Создаем концепт
                session.run("""
                    MERGE (c:Концепт {id: $id})
                    ON CREATE SET c.name = $name
                """, id=concept["id"], name=concept["name"])
                logger.info(f"Created concept: {concept['id']}")
                
                # Создаем связи
                for relation_type, target_id in concept["relations"]:
                    session.run(f"""
                        MATCH (c:Концепт {{id: $concept_id}})
                        MATCH (t {{id: $target_id}})
                        MERGE (c)-[r:{relation_type}]->(t)
                    """, concept_id=concept["id"], target_id=target_id)
                    logger.info(f"Created relation: {concept['id']} -> {relation_type} -> {target_id}")
            
            # Add ЯВЛЯЕТСЯ_ОДЕЖДОЙ relations to dress concepts
            session.run("""
                MERGE (category:Категория {id: 'Категория:одежда', name: 'одежда'})
            """)
            
            for concept in concepts_to_add:
                session.run("""
                    MATCH (concept:Концепт {id: $concept_id})
                    MATCH (category:Категория {id: 'Категория:одежда'})
                    MERGE (concept)-[:ЯВЛЯЕТСЯ_ОДЕЖДОЙ]->(category)
                """, concept_id=concept["id"])
                logger.info(f"Added ЯВЛЯЕТСЯ_ОДЕЖДОЙ relation for: {concept['name']}")
            
            logger.info("Successfully added all dress concepts!")
            
    except Exception as e:
        logger.error(f"Error adding dress concepts: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    add_dress_concepts() 