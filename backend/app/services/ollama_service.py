"""Service for communicating with the local Ollama API."""

from typing import Any

import httpx

from backend.app.core.config import settings


class OllamaServiceError(Exception):
    """Raised when Ollama cannot return a usable response."""


class OllamaService:
    """Small client for Ollama text generation."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_response(self, prompt: str) -> str:
        """Generate a response from Ollama for the given prompt."""
        return self.generate_response_with_metadata(prompt)["response"]

    def generate_response_with_metadata(self, prompt: str) -> dict[str, Any]:
        """Generate a response and return the text alongside Ollama metadata."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaServiceError("Unable to connect to Ollama. Please make sure Ollama is running.") from exc
        except httpx.TimeoutException as exc:
            raise OllamaServiceError("Ollama did not respond before the request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaServiceError("Ollama returned an unsuccessful response.") from exc
        except httpx.HTTPError as exc:
            raise OllamaServiceError("An error occurred while communicating with Ollama.") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaServiceError("Ollama returned an invalid JSON response.") from exc

        generated_text = data.get("response")
        if not isinstance(generated_text, str):
            raise OllamaServiceError("Ollama response did not contain generated text.")

        return {
            "response": generated_text,
            "model": data.get("model", self.model),
            "created_at": data.get("created_at"),
            "done": data.get("done"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
            "context": data.get("context"),
        }
