import enum
from typing import Any, Optional
from pydantic import BaseModel, Field

class GenerativeModel(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    prefix: str = Field(...)
    version: str = Field(...)
    endpoint: str = Field(...)
    headers: dict = Field(default_factory=dict)
    model: Optional[Any] = Field(default_factory=dict)
    model_type: Optional[str] = Field(None)

    def __call__(self, **kwds: Any) -> Any:
        raise NotImplementedError

class GenerativeModelInternal():
    name: str
    description: str
    prefix: str
    version: str
    endpoint: str
    headers: dict
    model: Optional[Any]
    model_type: Optional[str]

    def __init__(self, name: str, description: str, prefix: str, version: str, endpoint: str, model_type: Optional[str], headers: dict={}):
        self.name = name
        self.description = description
        self.prefix = prefix
        self.version = version
        self.endpoint = endpoint
        self.headers = headers
        self.model_type = model_type
    
    def dantize(self):
        return GenerativeModel(
            name=self.name,
            description=self.description,
            prefix=self.prefix,
            version=self.version,
            endpoint=self.endpoint,
            headers=self.headers,
            model_type=self.model_type
        )

# types of model is a list of literal strings: MODEL_TYPE.NATIVE_HF, MODEL_TYPE.REMOTE_TOGETHER, MODEL_TYPE.REMOTE_ANTHROPIC
class MODEL_TYPE(str, enum.Enum):
    NATIVE_HF = "native_hf"
    REMOTE_TOGETHER = "remote_together"
    REMOTE_ANTHROPIC = "remote_anthropic"