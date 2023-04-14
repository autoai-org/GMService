from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Any

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class ChatHistory(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    dialogs: list = Field(...)
    model: str = Field(...)
    tags: list = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "chat_id": "session_01",
                "dialogs": [{"role":"user", "text":"hello"}, {"role":"bot", "text":"hello"}],
                "tags": ["greeting", "introduction"]
            }
        }

