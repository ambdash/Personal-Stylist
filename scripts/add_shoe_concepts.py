from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

def add_shoe_concepts():
    """Add shoe concepts for the example"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Добавляем концепты обуви
            concepts_to_add = [
                {
                    "id": "Концепт:легкие кожаные лоферы",
                    "name": "легкие кожаные лоферы",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОБУВЬЮ", "Обувь:лоферы"),
                        ("В_СЕЗОНЕ", "Сезон:лето"),
                        ("ПОДХОДИТ_ДЛЯ", "Случай:офис"),
                        ("ОТНОСИТСЯ_К", "Тренд:2025"),
                        ("СДЕЛАН_ИЗ", "Материал:кожа"),
                        ("ОКРАШЕН_В", "Цвет:бежевый")
                    ]
                },
                {
                    "id": "Концепт:бежевые балетки",
                    "name": "бежевые балетки",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОБУВЬЮ", "Обувь:балетки"),
                        ("В_СЕЗОНЕ", "Сезон:лето"),
                        ("ОТНОСИТСЯ_К", "Тренд:2025"),
                        ("ОКРАШЕН_В", "Цвет:бежевый")
                    ]
                },
                {
                    "id": "Концепт:строгие ботинки-оксфорды",
                    "name": "строгие ботинки-оксфорды",
                    "relations": [
                        ("ЯВЛЯЕТСЯ_ОБУВЬЮ", "Обувь:оксфорды"),
                        ("ПОДХОДИТ_ДЛЯ", "Случай:офис"),
                        ("ОТНОСИТСЯ_К", "Тренд:2025"),
                        ("СДЕЛАН_ИЗ", "Материал:кожа")
                    ]
                }
            ]
            
            # Создаем недостающие узлы
            nodes_to_create = [
                ("Обувь:лоферы", "лоферы", "Обувь"),
                ("Обувь:балетки", "балетки", "Обувь"),
                ("Обувь:оксфорды", "оксфорды", "Обувь"),
                ("Материал:кожа", "кожа", "Материал"),
                ("Цвет:бежевый", "бежевый", "Цвет")
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
            
            logger.info("Successfully added all shoe concepts!")
            
    except Exception as e:
        logger.error(f"Error adding shoe concepts: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    add_shoe_concepts() 