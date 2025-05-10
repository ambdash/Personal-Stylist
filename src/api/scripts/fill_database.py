from typing import List, Tuple
from ..utils.triple_extractor import TripleExtractor
from ..db.neo4j_config import neo4j_connection
import logging

logger = logging.getLogger(__name__)

FASHION_STATEMENTS = [
    # Styles and their characteristics
    "Y2K стиль включает розовый топ и подходит для вечеринки",
    "Y2K стиль включает блестящие аксессуары и яркие цвета",
    "Грандж стиль включает черные джинсы и подходит для концерта",
    "Old money стиль включает бежевый тренч и подходит для офиса",
    "Clean girl стиль включает белую рубашку и подходит для работы",
    
    # Color combinations
    "Розовый цвет сочетается с белым и подходит для лета",
    "Черный цвет сочетается с бежевым и подходит для осени",
    "Красный цвет сочетается с белым и подходит для зимы",
    
    # Event recommendations
    "Для вечеринки подходит Y2K стиль и яркие цвета",
    "Для офиса подходит clean girl стиль и нейтральные цвета",
    "Для свидания подходит old money стиль и элегантные вещи",
    
    # Seasonal recommendations
    "Летом подходит розовый топ и белая юбка",
    "Осенью подходит бежевый тренч и черные ботинки",
    "Зимой подходит красный свитер и черные джинсы"
]

def fill_database():
    """Fill the Neo4j database with fashion-related triples"""
    try:
        extractor = TripleExtractor()
        
        # Process each fashion statement
        for statement in FASHION_STATEMENTS:
            triples = extractor.extract_triples(statement)
            
            # Store triples in Neo4j
            for subject, predicate, obj in triples:
                query = """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATIONSHIP {type: $predicate}]->(o)
                RETURN s, r, o
                """
                neo4j_connection.execute_query(
                    query,
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj
                    }
                )
                logger.info(f"Created triple: {subject} - {predicate} -> {obj}")
        
        logger.info("Database filled successfully")
        return True
    except Exception as e:
        logger.error(f"Error filling database: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fill_database() 