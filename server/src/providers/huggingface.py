import torch
from typing import Any
from src.model import ResponseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.providers._base import GenerativeModelInternal, MODEL_TYPE

class HuggingFaceCausalLM(GenerativeModelInternal):
    def __init__(self, name: str, description: str, version="v1", model=None, tokenizer=None):
        super().__init__(
            name=name,
            description=description,
            prefix='huggingface',
            version = version,
            endpoint = "",
            model_type=MODEL_TYPE.NATIVE_HF
        )
        if model is None:
            self.model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(name)
            self.model = self.model.half()
        else:
            self.model = model.half()
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to(self):
        self.model.to(self.device)

    @classmethod
    def from_mixed(self, model, name: str, description: str, version: str = "v1"):
        return HuggingFaceCausalLM(name, description, version=version)

    def format_output(self, response: str):
        additional_info = {
            "finish_reason": "length",
            "compute_time": -1,
            'type': 'language-model-inference'
        }
        return ResponseModel(
            model = self.name,
            output = response,
            status = 'finished',
            additional = additional_info
        )

    async def __call__(self, args) -> Any:
        try:
            payload = {}
            payload['prompt'] = args.get('prompt', '')
            payload['max_tokens'] = args.get('max_tokens', 128)
            payload['temperature'] = args.get('temperature', 1.0)
            payload['top_k'] = args.get('top_k', 50)
            payload['top_p'] = args.get('top_p', 0.95)
            
            inputs = self.tokenizer(payload['prompt'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=payload['max_tokens'],
                do_sample=True,
                top_k=payload['top_k'],
                top_p=payload['top_p'],
                temperature=payload['temperature'],
                return_dict_in_generate=True,
                output_scores=False, # return logit score
                output_hidden_states=True,
            )
            token = outputs.sequences[0, input_length:]
            output = self.tokenizer.decode(token)
            return self.format_output(output), None
        except Exception as e:
            return str(e), e

    def weighted(self, weight: float) -> None:
        for param in self.model.parameters():
            param.data = param.data * weight

huggingface_models = [
    HuggingFaceCausalLM(
        "../.cache/models/pythia-gpt4all-6000/",
        "Pythia gpt4all 6000",
        version="v1"
    ),
    HuggingFaceCausalLM(
        "../.cache/models/pythia-sharegpt-6000/",
        "Pythia sharegpt 6000",
        version="v1"
    ),
    HuggingFaceCausalLM(
        "../model-mixture/models/pythia-sharegpt-gpt4all-12000/",
        "Pythia OIG ShareGPT GPT4All 12000",
        version="v1"
    )
]