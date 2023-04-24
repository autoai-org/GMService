import os
from loguru import logger
import motor.motor_asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.router.models import router as model_router
from src.router.primitives import router as primitive_router
app = FastAPI(title="GMService", description="Generative Models as a Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get("MONGODB_URL", ""))
    db = client.gmservice
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")

app.include_router(model_router)
app.include_router(primitive_router)