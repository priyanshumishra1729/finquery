"""Main FastAPI application for FinQuery."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as api_router


app = FastAPI(
    title="FinQuery — Financial Document Intelligence Assistant",
    description=(
        "FinQuery is currently an Exercise 1 LLM application that "
        "communicates with Code Llama through Ollama."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a small status message for the API root."""
    return {
        "message": "FinQuery API is running",
        "status": "ok",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "healthy"}
