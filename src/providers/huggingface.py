import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
from src.providers._base import GenerativeModelInternal, MODEL_TYPE

class HuggingFaceCausalLM(GenerativeModelInternal):
    def __init__(self, name: str, description: str, version="v1"):
        super().__init__(
            name=name,
            description=description,
            prefix='huggingface',
            version = version,
            endpoint = "",
            model_type=MODEL_TYPE.NATIVE_HF
        )
        self.model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(name)

    async def __call__(self, args) -> Any:
        payload = {}
        payload['prompt'] = args.get('prompt', '')
        payload['max_tokens'] = args.get('max_tokens', 128)
        payload['temperature'] = args.get('temperature', 1.0)
        payload['top_k'] = args.get('top_k', 50)
        payload['top_p'] = args.get('top_p', 0.95)
        outputs = self.model.generate(payload['prompt'], max_new_tokens=payload['max_tokens'], do_sample=True, top_k=payload['top_k'], top_p=payload['top_p'], temperature=payload['temperature'])
    
    def weighted(self, weight: float) -> None:
        for param in self.model.parameters():
            param.data = param.data * weight


huggingface_models = [
    HuggingFaceCausalLM(
        "../model-mixture/models/pythia-dolly-2000/",
        "Pythia Dolly 2000",
        version="v1"
    ),
    HuggingFaceCausalLM(
        "../model-mixture/models/pythia-oig-dolly-2000/",
        "Pythia OIG Dolly 2000",
        version="v1"
    ),
    # HuggingFaceCausalLM(
    #     "../model-mixture/models/pythia-oig-sharegpt-gpt4all-12000/",
    #     "Pythia OIG ShareGPT GPT4All 12000",
    #     version="v1"
    # )
]