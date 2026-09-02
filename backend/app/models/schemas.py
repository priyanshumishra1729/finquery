"""Pydantic schemas for the FinQuery chat API."""

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    """Request body for a future chat endpoint."""

    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        stripped_message = value.strip()
        if not stripped_message:
            raise ValueError("message cannot be empty")
        return stripped_message


class ChatResponse(BaseModel):
    """Response body for a future chat endpoint."""

    success: bool
    response: str
    model: str
