# FinQuery Architecture

## Overview

FinQuery is an academic financial document intelligence assistant built around a small, explicit service pipeline. The application uses FastAPI for the external HTTP API, an internal application orchestration layer for chat flow coordination, a RAG pipeline for retrieval and context building, Ollama for embeddings and text generation, and ChromaDB for persistent knowledge-base vector storage.

The architecture is intentionally simple: each service has one responsibility, and the request path is easy to trace from user input to the generated response. FinQuery uses modular internal Python services rather than unnecessary network microservices.

## Components

### 1. Application/API Service

The Application/API Service is the FastAPI layer in `backend/app/main.py` and `backend/app/api/routes.py`.

Responsibilities:
- Receives HTTP requests.
- Validates `ChatRequest` payloads.
- Exposes the existing `POST /api/v1/chat` endpoint.
- Returns `ChatResponse` to the client.
- Keeps the route thin by delegating orchestration work to `ApplicationService`.

### 1b. Application Service

The Application Service is implemented in `backend/app/services/application_service.py`.

Responsibilities:
- Orchestrates the end-to-end chat workflow.
- Accepts the user message from the FastAPI route.
- Calls `RAGService` to build context.
- Builds the final FinQuery prompt.
- Calls `OllamaService.generate_response()` to produce the final answer.
- Converts lower-level service failures into a clean application-level error.

### 2. RAG Service

The RAG Service is implemented in `backend/app/services/rag_service.py`.

Responsibilities:
- Coordinates the retrieval pipeline for a user question.
- Receives the user's question from the API layer.
- Requests relevant chunks through `RetrievalService`.
- Formats those chunks into a readable context string.
- Produces the financial context that will later be passed to the LLM.

### 3. Retrieval Service

The Retrieval Service is implemented in `backend/app/services/retrieval_service.py`.

Responsibilities:
- Validates and normalizes the user query.
- Generates a query embedding using `EmbeddingService`.
- Passes the query embedding to `VectorStoreService.search()`.
- Returns the most relevant knowledge-base chunks.
- Keeps embedding generation and similarity search out of the route and out of the RAG formatter.

### 4. Embedding Service

The Embedding Service is implemented in `backend/app/services/embedding_service.py`.

Responsibilities:
- Communicates with Ollama's embedding API.
- Uses the `nomic-embed-text` model.
- Produces 768-dimensional embedding vectors.
- Embeds document chunks for storage and query text for retrieval.
- Does not know anything about FastAPI, ChromaDB, or prompt construction.

### 5. Vector Store Service

The Vector Store Service is implemented in `backend/app/services/vector_store_service.py`.

Responsibilities:
- Communicates with ChromaDB.
- Stores chunk content, chunk metadata, and embeddings.
- Searches the `finquery_knowledge_base` collection for relevant chunks.
- Returns similarity-search results with `filename`, `chunk_index`, `content`, and `distance`.
- Keeps storage and search logic independent from the API and from retrieval orchestration.

### 6. LLM Service

The LLM Service is the existing `OllamaService` in `backend/app/services/ollama_service.py`.

Responsibilities:
- Communicates with Ollama's generate API.
- Uses Code Llama as configured in the application settings.
- Generates the final response from the prompt produced by the chat route.
- Does not perform retrieval or context formatting.

### 7. Data / Knowledge Base

The knowledge base lives in `backend/data/knowledge_base/`.

Supporting services:
- `DocumentLoader` loads the source text documents.
- `ChunkingService` splits documents into overlapping chunks.

Responsibilities:
- Provide the raw financial reference material used by the rest of the pipeline.
- Supply stable, local educational content for embeddings, storage, retrieval, and context building.

## Current Request Flow

User
→ API Service
→ Application Service
→ RAG Service
→ Retrieval Service
→ Embedding Service
→ Vector Store
→ Relevant Context
→ LLM Service
→ Code Llama
→ Response

## Exercise 4 Target Architecture

Exercise 4 focuses on service separation, clear orchestration, and a clean application structure without unnecessary complexity.

The target direction is:
- Keep the existing FastAPI entry point.
- Keep the RAG pipeline organized into small services.
- Add an internal application orchestration layer between the route and the RAG/LLM services.
- Make the flow easy to test and reason about.
- Preserve the academic scope of the project.
- Avoid introducing unnecessary microservices, brokers, caches, Kubernetes, or additional databases.

The application should remain simple, local, and understandable while still demonstrating how API handling, retrieval, context building, and LLM generation fit together as a coherent system.

## API and Service Communication

HTTP Request
↓
FastAPI /api/v1/chat
↓
ApplicationService
↓
RAGService
↓
RetrievalService
↓
EmbeddingService
↓
VectorStoreService / ChromaDB
↓
Context
↓
OllamaService
↓
Ollama API
↓
Code Llama
↓
ApplicationService
↓
ChatResponse
↓
HTTP Response

Communication types:
- HTTP/API communication: the external request between the client and FastAPI, and the HTTP calls from OllamaService to Ollama.
- Internal Python service communication: calls between ApplicationService, RAGService, RetrievalService, and EmbeddingService.
- Vector database communication: requests from VectorStoreService to ChromaDB for storing and searching vectors.

## Notes

- The architecture is intentionally centered on local services and a single persistent vector store.
- ChromaDB remains the only storage layer for the knowledge-base vectors.
- The current implementation is already suitable for demonstrating a full retrieval-augmented workflow in a compact academic project.
