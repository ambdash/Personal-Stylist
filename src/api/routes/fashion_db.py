from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from ..db.neo4j import db

router = APIRouter(prefix="/fashion-db", tags=["fashion-db"])

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
        result = db.create_fashion_item(item.dict())
        return {"message": "Item created successfully", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outfits/", response_model=dict)
async def create_outfit(outfit: Outfit):
    try:
        result = db.create_outfit(outfit.items, outfit.style, outfit.occasion)
        return {"message": "Outfit created successfully", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/items/{style}/{category}")
async def get_matching_items(style: str, category: str):
    try:
        result = db.get_matching_items(style, category)
        return {"items": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 