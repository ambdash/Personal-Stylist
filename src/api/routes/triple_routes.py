from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ..utils.triple_extractor import TripleExtractor
from ..db.neo4j.config import neo4j_connection
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/triples", tags=["triples"])

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

class TextInput(BaseModel):
    text: str

triple_extractor = TripleExtractor()

@router.post("/extract", response_model=List[Triple])
async def extract_triples(input_data: TextInput):
    """Extract triples from input text and store them in Neo4j"""
    try:
        # Extract triples
        triples = triple_extractor.extract_triples(input_data.text)
        
        # Convert to list of Triple models
        triple_models = []
        for subject, predicate, obj in triples:
            # Create nodes and relationships in Neo4j
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
            
            triple_models.append(Triple(
                subject=subject,
                predicate=predicate,
                object=obj
            ))
            
        return triple_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{entity}")
async def search_entity_relationships(entity: str):
    """Search for all relationships involving an entity"""
    try:
        query = """
        MATCH (n:Entity {name: $entity})-[r]-(m)
        RETURN n.name as entity, type(r) as relationship, m.name as related_entity
        """
        results = neo4j_connection.execute_query(query, {"entity": entity})
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/entities/{entity}")
async def delete_entity(entity: str):
    """Delete an entity and its relationships"""
    try:
        query = """
        MATCH (n:Entity {name: $entity})
        DETACH DELETE n
        """
        neo4j_connection.execute_query(query, parameters={"entity": entity})
        return {"message": f"Entity {entity} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph")
async def get_graph_data():
    """Get all entities and relationships for visualization"""
    try:
        query = """
        MATCH (n:Entity)-[r]->(m:Entity)
        RETURN n.name as source, type(r) as relationship, m.name as target
        """
        results = neo4j_connection.execute_query(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 