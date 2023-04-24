from typing import List
from src.providers._base import GenerativeModel

def get_modelsoup(models: List[GenerativeModel], weights: List[float]) -> GenerativeModel:
    """
    get_modelsoup takes a list of models and a list of weights and returns a new model that is the weighted average of the models
    """
    # this only works for models with the type MODEL_TYPE.NATIVE_HF
    model_types = [model.model_type for model in models]
    if len(set(model_types)) > 1:
        raise ValueError("All models must be of the same type")

    if len(models) != len(weights):
        raise ValueError("Number of models must equal number of weights")
    
    if sum(weights) != 1.0:
        raise ValueError("Weights must sum to 1.0")
    # zero out the model
    # now calculate the weighted average of the models and assign it to the new model
    for i, model in enumerate(models):
        model.weighted(weights[i])
        if i > 0:
            for param in model.model.parameters():
                param.data += param.data
    return models[0]