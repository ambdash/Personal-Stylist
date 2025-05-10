from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from ..db.neo4j_config import neo4j_connection

router = APIRouter(prefix="/database", tags=["database"])

class FashionItem(BaseModel):
    name: str
    category: str
    style: str
    color: str
    season: str

class Outfit(BaseModel):
    items: List[str]
    style: str
    occasion: str

@router.post("/items/", response_model=dict)
async def create_fashion_item(item: FashionItem):
    try:
        query = """
        CREATE (i:FashionItem {
            name: $name,
            category: $category,
            style: $style,
            color: $color,
            season: $season
        })
        RETURN i
        """
        result = neo4j_connection.execute_query(query, item.dict())
        return {"message": "Item created successfully", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outfits/", response_model=dict)
async def create_outfit(outfit: Outfit):
    try:
        query = """
        CREATE (o:Outfit {style: $style, occasion: $occasion})
        WITH o
        UNWIND $items as item_name
        MATCH (i:FashionItem {name: item_name})
        CREATE (o)-[:CONTAINS]->(i)
        RETURN o
        """
        result = neo4j_connection.execute_query(
            query,
            {
                "items": outfit.items,
                "style": outfit.style,
                "occasion": outfit.occasion
            }
        )
        return {"message": "Outfit created successfully", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/items/{style}/{category}")
async def get_matching_items(style: str, category: str):
    try:
        query = """
        MATCH (i:FashionItem)
        WHERE i.style = $style AND i.category = $category
        RETURN i
        """
        result = neo4j_connection.execute_query(
            query,
            {"style": style, "category": category}
        )
        return {"items": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 