import os
from fastapi import FastAPI, Body, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from src.providers import models, GenerativeModel

app = FastAPI(title="GMService", description="Generative Models as a Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.gmservice

@app.get("/api/models", response_description="List all models", response_model=list[GenerativeModel])
async def list_models():
    return models