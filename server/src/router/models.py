from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import status

from src.providers import models, GenerativeModel, GenerativeModelInternal
from src.model import RequestModel, ResponseModel, DialogModel
from src.shortcuts.chat import chat_shortcut

router = APIRouter()

@router.get("/api/models", response_description="List all models", response_model=list[GenerativeModel])
async def list_models():
    results = []
    for model in models:
        if isinstance(model, GenerativeModelInternal):
            results.append(model.dantize())
        else:
            results.append(model)
    return results

@router.post("/api/predict", response_description="Predict response from a model", response_model=ResponseModel)
async def predict(query: RequestModel):
    model_name = query.model
    model = next((model for model in models if model.name == model_name), None)
    if model is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Model not found"})
    res, err = await model(query.body)
    if err is not None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(err)})
    return res

@router.post("/api/chat", response_description="Chat with a model", response_model=ResponseModel)
async def chat(query: DialogModel):
    model_name = query.model
    if model_name.lower() == "random":
        import random
        model = random.choice(models)
    else:
        model = next((model for model in models if model.name == model_name), None)
    if model is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Model not found"})
    res, err = await chat_shortcut(model, query)
    if err is not None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(err)})
    return res