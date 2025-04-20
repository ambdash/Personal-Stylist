from fastapi import APIRouter
from models.train_model import TrainRequest, TrainResponse
from services.training import train_model

router = APIRouter(tags=["Training"])

@router.post("/train", response_model=TrainResponse)
async def train_llm(request: TrainRequest):
    """
    Train the LLM model with new fashion datasets.
    """
    success = train_model(request.dataset_url)

    return {"status": "success" if success else "failed"}
