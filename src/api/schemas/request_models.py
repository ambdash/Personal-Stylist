from pydantic import BaseModel, Field
from typing import List, Optional

class InferenceRequest(BaseModel):
    text: str = Field(..., max_length=1000)  # Limit input length
    max_length: Optional[int] = Field(default=100, ge=1, le=500)  # Limit output length
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)

class InferenceResponse(BaseModel):
    generated_text: str

class StyleQuery(BaseModel):
    user_id: str
    style_type: str

class OutfitItem(BaseModel):
    name: str
    category: str

class OutfitData(BaseModel):
    id: str
    description: str
    items: List[OutfitItem]

class TrainingRequest(BaseModel):
    dataset_path: str
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 8

class TrainingResponse(BaseModel):
    status: str
    message: str 