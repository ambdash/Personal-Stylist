from fastapi import APIRouter, HTTPException
from redis import Redis
import json
from typing import Optional

router = APIRouter(prefix="/v1/cache", tags=["cache"])

redis_client = Redis(host='redis', port=6379, db=0)

@router.get("/style/{style_id}")
async def get_cached_style(style_id: str) -> Optional[dict]:
    """Get cached style recommendations"""
    cached = redis_client.get(f"style:{style_id}")
    if cached:
        return json.loads(cached)
    return None

@router.post("/style/{style_id}")
async def cache_style(style_id: str, data: dict):
    """Cache style recommendations"""
    redis_client.setex(
        f"style:{style_id}",
        3600,  # 1 hour expiration
        json.dumps(data)
    )
    return {"status": "cached"} 