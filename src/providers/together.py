import os
import aiohttp
from typing import Any
from src.model import ResponseModel
from src.providers._base import GenerativeModel

class TogetherModel(GenerativeModel):
    def __init__(self, name: str, description: str, version="v1"):
        KEY_PATH = "TOGETHER_APIKEY"
        headers = {
            'Authorization': 'Bearer ' + os.environ[KEY_PATH],
        }
        super().__init__(
            name=name,
            description=description,
            prefix='together',
            version = version,
            endpoint = "https://api.together.xyz/inference",
            headers=headers
        )

    def format_output(self, response: dict):
        additional_info = {
            "finish_reason": response['output']['choices'][0]['finish_reason'],
            "compute_time": response['output']['raw_compute_time'],
            'type': response['output']['result_type']
        }
        return ResponseModel(
            model = self.name,
            output = response['output']['choices'][0]['text'],
            status = response['status'],
            additional = additional_info
        )

    async def __call__(self, args) -> Any:
        payload = {
            "model": self.name,
        }
        for key, value in args.items():
            payload[key] = value
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(self.endpoint, json=payload) as r:
                if r.status == 200:
                    try:
                        res = await r.json()
                        return self.format_output(res), None
                    except Exception as e:
                        return str(e), e
                else:
                    return f"{r.status} {r.reason}", ValueError(f"{r.status} {r.reason}")

together_models = [
    TogetherModel("pythia-openalign", "Pythia OpenAlign", version="2023-04-14"),
    TogetherModel("GPT-NeoXT-Chat-Base-20B-v0.16", "OCK-20B", version="2023-04-14"),
    TogetherModel("Pythia-Chat-Base-7B-v0.16", "OCK-7B", version="2023-04-14"),
    TogetherModel("Pythia-Chat-Base-7B-v0.16-int8", "OCK-7B-int8", version="2023-04-14"),
]