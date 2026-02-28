# REST API Usage Guide

The GAM REST API provides a standalone, high-performance RESTful service built on **FastAPI + Uvicorn**. It offers automatic request validation, interactive API documentation, and CORS support out of the box.

## Starting the API Server

```bash
# Default (port 5001, no pre-configured LLM)
python examples/run_api.py

# With a default LLM model
python examples/run_api.py --model gpt-4o-mini --api-key sk-xxx --port 5001
```

Once running, the interactive documentation is available at:

| URL | Description |
|---|---|
| `http://localhost:5001/docs` | Swagger UI (interactive) |
| `http://localhost:5001/redoc` | ReDoc (read-only) |

## API Endpoints

### 1. Add Content (`POST /api/v1/add`)

Add text or video content to a GAM.

**Request Body (JSON):**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"text"` | GAM type: `"text"` or `"video"` |
| `gam_dir` | string | (auto) | Path to the GAM directory |
| `input` | string / array | `null` | Input file paths or video directory |
| `content` | string / array | `null` | [Text] Raw text content to add |
| `model` | string | (env) | LLM model name |
| `api_base` | string | (env) | API base URL |
| `api_key` | string | (env) | API key |
| `use_chunking` | boolean | `true` | [Text] Use intelligent chunking |
| `max_splits` | integer | `120` | [Text] Maximum number of chunks |
| `force_reorganize` | boolean | `false` | [Text] Force hierarchical reorganization |

**Example:**

```bash
curl -X POST http://localhost:5001/api/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text",
    "content": ["New research data...", "Secondary findings..."],
    "model": "gpt-4o",
    "use_chunking": true
  }'
```

```python
import requests

payload = {
    "type": "text",
    "content": ["New research data...", "Secondary findings..."],
    "model": "gpt-4o",
    "use_chunking": True,
}
resp = requests.post("http://localhost:5001/api/v1/add", json=payload)
print(resp.json())
```

**Response:**

```json
{
  "success": true,
  "type": "text",
  "gam_dir": "/path/to/gam",
  "output_dir": "/path/to/chunks",
  "created_files": 5,
  "new_directories": 2
}
```

---

### 2. Query / QA (`POST /api/v1/query`)

Query an existing GAM knowledge base.

**Request Body (JSON):**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"text"` | GAM type: `"text"` or `"video"` |
| `gam_dir` | string | (required) | Path to the GAM directory |
| `question` | string | (required) | User question |
| `model` | string | (env) | LLM model name |
| `max_iter` | integer | `10` | Maximum exploration iterations |
| `system_prompt` | string | `""` | Optional system prompt override |

**Example:**

```bash
curl -X POST http://localhost:5001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text",
    "gam_dir": "./my_gam",
    "question": "Summarize the key findings.",
    "model": "gpt-4o-mini"
  }'
```

```python
import requests

payload = {
    "type": "text",
    "gam_dir": "./my_gam",
    "question": "Summarize the key findings.",
    "model": "gpt-4o-mini",
}
resp = requests.post("http://localhost:5001/api/v1/query", json=payload)
print(resp.json()["answer"])
```

**Response:**

```json
{
  "success": true,
  "question": "Summarize the key findings.",
  "answer": "The key findings are...",
  "sources": ["file1.md", "file2.md"],
  "confidence": 0.85,
  "notes": "",
  "files_read": ["dir/file1.md"],
  "dirs_explored": ["dir/"]
}
```

---

### 3. Health Check (`GET /`)

```bash
curl http://localhost:5001/
```

```json
{
  "service": "GAM REST API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

## Python Client Example

A complete example is available at [examples/rest_api_client.py](../rest_api_client.py).
