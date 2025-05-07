from fastapi import APIRouter, HTTPException
from neo4j import GraphDatabase
from typing import List, Dict
from src.api.schemas.request_models import StyleQuery, OutfitData
from src.api.db.neo4j import db

router = APIRouter(prefix="/v1/db", tags=["database"])

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "neo4j://neo4j:7687",
            auth=("neo4j", "password123")
        )

    def close(self):
        self.driver.close()

db = Neo4jConnection()

@router.post("/outfits")
async def store_outfit(outfit: OutfitData):
    """Store new outfit in Neo4j"""
    query = """
    CREATE (o:Outfit {id: $id, description: $description})
    WITH o
    UNWIND $items as item
    CREATE (i:Item {name: item.name, category: item.category})
    CREATE (o)-[:CONTAINS]->(i)
    """
    try:
        with db.driver.session() as session:
            session.run(query, **outfit.dict())
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema")
async def get_schema():
    try:
        labels = db.execute_query("CALL db.labels()")
        
        relationships = db.execute_query("CALL db.relationshipTypes()")
        
        # Get property keys
        properties = db.execute_query("CALL db.propertyKeys()")
        
        return {
            "labels": [label["label"] for label in labels],
            "relationships": [rel["relationshipType"] for rel in relationships],
            "properties": [prop["propertyKey"] for prop in properties]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    try:
        stats = db.execute_query("""
        MATCH (n)
        RETURN labels(n) as label, count(*) as count
        """)
        
        relationships = db.execute_query("""
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        """)
        
        return {
            "nodes": stats,
            "relationships": relationships
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 