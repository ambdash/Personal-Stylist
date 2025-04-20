from fastapi import APIRouter, HTTPException
from neo4j import GraphDatabase
from typing import List, Dict
from src.api.schemas.request_models import StyleQuery, OutfitData

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

@router.get("/styles/{user_id}")
async def get_user_styles(user_id: str) -> List[Dict]:
    """Get user style preferences from Neo4j"""
    query = """
    MATCH (u:User {id: $user_id})-[r:LIKES]->(s:Style)
    RETURN s.name as style, r.score as preference_score
    """
    with db.driver.session() as session:
        result = session.run(query, user_id=user_id)
        return [record.data() for record in result]

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