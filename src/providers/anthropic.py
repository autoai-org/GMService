import os
import aiohttp
from typing import Any
from src.model import ResponseModel
from src.providers._base import GenerativeModel

class AnthropicModel(GenerativeModel):
    def __init__(self, name: str, description: str, version="v1"):
        KEY_PATH = "ANTHROPIC_APIKEY"
        headers = {
            'x-api-key': os.environ[KEY_PATH],
            'content-type': 'application/json',
        }
        super().__init__(
            name=name,
            description=description,
            prefix='anthropic',
            version = version,
            endpoint='https://api.anthropic.com/v1/complete',
            headers=headers
        )

    def format_output(self, response: dict):
        additional_info = {
            "finish_reason": response['stop_reason'],
            "compute_time": -1,
            'type': 'language-model-inference'
        }
        return ResponseModel(
            model = self.name,
            output = response['completion'],
            status = 'finished',
            additional = additional_info
        )

    async def __call__(self,  **kwargs: Any) -> Any:
        payload = {
            "model": self.model,
        }
        for key, value in kwargs.items():
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
                    return r.status + " " + r.reason, ValueError("Server Error"+r.status + " " + r.reason)

anthropic_models = [
    AnthropicModel(
        "claude-v1.3",
        "A significantly improved version of claude-v1. Compared to claude-v1.2, it's more robust against red-team inputs, better at precise instruction-following, better at code, and better and non-English dialogue and writing.",
        version="2023-04-14"
    ),
    AnthropicModel(
        "claude-v1.2",
        "An improved version of claude-v1. It is slightly improved at general helpfulness, instruction following, coding, and other tasks. It is also considerably better with non-English languages. This model also has the ability to role play (in harmless ways) more consistently, and it defaults to writing somewhat longer and more thorough responses",
        version="2023-04-14"
    ),
    AnthropicModel(
        "claude-v1.0",
        "An earlier version of claude-v1",
        version="2023-04-14"
    ),
    AnthropicModel(
        "claude-instant-v1.0",
        "A smaller model with far lower latency, sampling at roughly 40 words/sec! Its output quality is somewhat lower than claude-v1 models, particularly for complex tasks. However, it is much less expensive and blazing fast. We believe that this model provides more than adequate performance on a range of tasks including text classification, summarization, and lightweight chat applications, as well as search result summarization. Using this model name will automatically switch you to newer versions of claude-instant-v1 as they are released.",
        version="2023-04-14"
    )
]