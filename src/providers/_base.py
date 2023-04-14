from typing import Any

class GenerativeModel():
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)