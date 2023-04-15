from typing import Any
from pydantic import BaseModel, Field

class GenerativeModel(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    prefix: str = Field(...)
    version: str = Field(...)
    endpoint: str = Field(...)
    headers: dict = Field(default_factory=dict)

    def __call__(self, **kwds: Any) -> Any:
        raise NotImplementedError