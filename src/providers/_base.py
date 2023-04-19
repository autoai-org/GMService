import enum
from typing import Any
from pydantic import BaseModel, Field

class GenerativeModel(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    prefix: str = Field(...)
    version: str = Field(...)
    endpoint: str = Field(...)
    headers: dict = Field(default_factory=dict)
    model: Any = Field(default_factory=dict)
    def __call__(self, **kwds: Any) -> Any:
        raise NotImplementedError

# types of model is a list of literal strings: MODEL_TYPE.NATIVE_HF, MODEL_TYPE.REMOTE_TOGETHER, MODEL_TYPE.REMOTE_ANTHROPIC
class MODEL_TYPE(str, enum.Enum):
    NATIVE_HF = "native_hf"
    REMOTE_TOGETHER = "remote_together"
    REMOTE_ANTHROPIC = "remote_anthropic"
