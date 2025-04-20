from fastapi import APIRouter, HTTPException
from models.recommend_model import RecommendRequest, RecommendResponse
from services.recommendation import get_recommendations

router = APIRouter(tags=["Recommendation"])

@router.post("/recommend", response_model=RecommendResponse)
async def recommend_outfit(request: RecommendRequest):
    """
    Generate outfit recommendations using LLM and Neo4j relationships.
    """
    outfits = get_recommendations(request.style)
    
    if not outfits:
        raise HTTPException(status_code=404, detail="No recommendations found.")

    return {"style": request.style, "outfits": outfits}
