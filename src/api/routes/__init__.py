from .triple_routes import router as triple_router
from .inference import router as inference_router
from .rag import router as rag_router
from .fashion_routes import router as fashion_router
from .database import router as db_router

__all__ = [
    'triple_router',
    'inference_router',
    'rag_router',
    'fashion_router',
    'db_router'
] 