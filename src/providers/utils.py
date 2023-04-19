from typing import List
from src.providers._base import GenerativeModel

def get_modelsoup(models: List[GenerativeModel], weights: List[float]) -> GenerativeModel:
    # this only works for models with the type MODEL_TYPE.NATIVE_HF
    model_types = [model.model_type for model in models]
    if len(set(model_types)) > 1:
        raise ValueError("All models must be of the same type")

    if len(models) != len(weights):
        raise ValueError("Number of models must equal number of weights")
    
    if sum(weights) != 1.0:
        raise ValueError("Weights must sum to 1.0")
    new_model = models[0].copy()
    # zero out the model
    new_model.weighted(0.0)
    # now calculate the weighted average of the models and assign it to the new model
    for i, model in enumerate(models):
        model.weighted(weights[i])
        new_model.model = new_model.model + model.model
    return new_model