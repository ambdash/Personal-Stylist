from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ModelName(str, Enum):
    STUB = "stub_model"

class InferenceRequest(BaseModel):
    text: str
    model_name: Optional[ModelName] = None
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    generated_text: str
    model_used: str
    generation_time: float

class ModelStatus(BaseModel):
    name: str
    status: str
    loaded: bool
    last_used: Optional[str]
    avg_inference_time: Optional[float]

class HealthResponse(BaseModel):
    status: str
    models: List[ModelStatus]
    api_version: str
    uptime: float
    total_requests: int
    avg_latency: float

class MetricsResponse(BaseModel):
    total_requests: int
    requests_per_model: dict
    average_latency: float
    error_rate: float
    model_load_times: dict

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