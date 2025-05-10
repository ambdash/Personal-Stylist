from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from src.ml.config import MODEL_CONFIGS

class ModelName(str, Enum):
    PHI2 = "microsoft/phi-2"
    MISTRAL = "mistralai/Mistral-7B-v0.1"
    SAIGA_LLAMA = "IlyaGusev/saiga_llama3_8b"
    SAIGA_MISTRAL = "IlyaGusev/saiga_mistral_7b"
    SAIGA2 = "IlyaGusev/saiga2_7b_lora"

class InferenceRequest(BaseModel):
    text: str
    model_name: Optional[ModelName] = None
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.2
    do_sample: Optional[bool] = True

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