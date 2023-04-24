from fastapi import APIRouter
from pydantic import BaseModel
from src.providers import GenerativeModel
from src.providers.utils import get_modelsoup
from src.providers.huggingface import HuggingFaceCausalLM, huggingface_models
router = APIRouter()

class MixModelsRequest(BaseModel):
    models: list[str] = []
    weights: list[float] = []

@router.post("/action/mix", tags=["model_soup"], response_model=list[GenerativeModel])
async def mix_models(query: MixModelsRequest):
    if len(query.weights) == 0:
        query.weights = [1.0 for _ in range(len(query.models))]
    if len(query.models) != len(query.weights):
        raise ValueError("Length of models and weights must be equal")
    # check if all models are in the HF_models
    known_hf_models = [model.name for model in huggingface_models]
    for idx, model in enumerate(query.models):
        if model not in known_hf_models:
            raise ValueError(f"Model {model} is not a known HuggingFace model")
        query.models[idx] = next((hf_model for hf_model in huggingface_models if hf_model.name == model), None)

    new_model = get_modelsoup(query.models, query.weights)
    new_model_name = "mix_"
    for i in range(len(query.models)):
        new_model_name += f"{query.weights[i]} * {query.models[i]}_"
    new_model = HuggingFaceCausalLM(
        name=new_model_name,
        description="Mix of models",
        version="0.1.0",
        model=new_model,
        tokenizer = query.models[0].tokenizer
    )
    huggingface_models.append(new_model)
    return huggingface_models