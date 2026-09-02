# FinQuery — Financial Document Intelligence Assistant

This is the FinQuery college project repository.

Exercise 1 is currently under development. This stage establishes the initial project structure only.

## Frontend

The frontend is a React + Vite interface for the Exercise 1 FinQuery chat API.

Install frontend dependencies:

```bash
cd frontend
npm install
```

Start the frontend:

```bash
npm run dev
```

Before using the frontend, make sure the backend is running:

```bash
uvicorn backend.app.main:app --reload
```

The frontend expects the backend at:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Exercise 1 also requires Ollama to be running locally with the Code Llama model available:

```bash
ollama pull codellama
```
