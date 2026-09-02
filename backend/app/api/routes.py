"""API routes for the FinQuery backend."""

import logging

from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import settings
from backend.app.models.schemas import ChatRequest, ChatResponse
from backend.app.services.application_service import ApplicationService, ApplicationServiceError


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Generate a FinQuery chat response using Ollama."""
    service = ApplicationService()

    try:
        generated_response = service.process_chat(request.message)
    except ApplicationServiceError as exc:
        logger.exception("Chat orchestration failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The chat service is currently unavailable.",
        ) from exc

    return ChatResponse(
        success=True,
        response=generated_response,
        model=settings.OLLAMA_MODEL,
    )
