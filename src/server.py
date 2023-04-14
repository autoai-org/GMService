import os
from typing import Any
import motor.motor_asyncio
from fastapi import FastAPI, Body, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from src.providers import models, GenerativeModel
from src.model import RequestModel, ResponseModel

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

@app.post("/api/predict", response_description="Predict response from a model", response_model=ResponseModel)
async def predict(query: RequestModel):
    model_name = query.model
    model = next((model for model in models if model.name == model_name), None)
    if model is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Model not found"})
    res, err = await model(query.body)
    if err is not None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(err)})
    return res

    