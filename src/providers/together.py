import os
import aiohttp
from typing import Any
from src.providers._base import GenerativeModel

class TogetherModel(GenerativeModel):
    def __init__(self, name: str, description: str, version="v1"):
        super().__init__(
            name=name,
            description=description
        )
        self.prefix='together'
        self.version = version
        self.endpoint = "https://api.together.xyz/inference"
        self.KEY_PATH = "TOGETHER_APIKEY"
        self.headers = {
            'Authorization': 'Bearer ' + os.environ[self.KEY_PATH],
        }
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        payload = {
            "model": self.model,
        }
        for key, value in kwargs.items():
            payload[key] = value
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload) as r:
                if r.status == 200:
                    try:
                        res = await r.json()
                        return res, None
                    except Exception as e:
                        return str(e), e
                else:
                    return r.status + " " + r.reason, ValueError("Server Error"+r.status + " " + r.reason)

together_models = [
    TogetherModel("pythia-openalign", "Pythia OpenAlign", version="2023-04-14"),
    TogetherModel("GPT-NeoXT-Chat-Base-20B-v0.16", "OCK-20B", version="2023-04-14"),
    TogetherModel("Pythia-Chat-Base-7B-v0.16", "OCK-7B", version="2023-04-14"),
    TogetherModel("Pythia-Chat-Base-7B-v0.16-int8", "OCK-7B-int8", version="2023-04-14"),
]