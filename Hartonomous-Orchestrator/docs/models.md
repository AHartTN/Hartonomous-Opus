# Models API

The Models API endpoint provides information about available language models in the local self-hosted deployment, compatible with OpenAI's models interface.

## Endpoints

### List Models
```
GET /v1/models
```

### Get Model
```
GET /v1/models/{model}
```

## Description

The Models API allows clients to discover and inspect the language models available in the local deployment. This includes text generation models, embedding models, and reranking models used for retrieval-augmented generation.

## Available Models

The current deployment includes the following models:

### Text Generation Models
- **`qwen3-coder-30b`**: Large language model optimized for code generation and programming tasks
  - Context window: Large (implementation dependent)
  - Best for: Code completion, debugging, technical writing
  - Architecture: Transformer-based with 30 billion parameters

### Embedding Models
- **`qwen3-embedding-4b`**: Specialized model for generating vector embeddings
  - Output dimensions: Variable (configurable)
  - Best for: Semantic search, document similarity, clustering
  - Architecture: Optimized for embedding generation

### Reranking Models
- **`qwen3-reranker-4b`**: Model for reranking retrieved documents by relevance
  - Input: Query + candidate documents
  - Output: Relevance scores
  - Best for: Improving search result quality in RAG applications

## List Models Response

### Endpoint: GET /v1/models

Returns a list of all available models in the deployment.

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-coder-30b",
      "object": "model",
      "created": 1703123456,
      "owned_by": "local"
    },
    {
      "id": "qwen3-embedding-4b",
      "object": "model",
      "created": 1703123456,
      "owned_by": "local"
    },
    {
      "id": "qwen3-reranker-4b",
      "object": "model",
      "created": 1703123456,
      "owned_by": "local"
    }
  ]
}
```

### Response Fields

- **`object`** (string): Always `"list"`
- **`data`** (array): Array of model objects
  - **`id`** (string): Unique model identifier
  - **`object`** (string): Always `"model"`
  - **`created`** (integer): Unix timestamp of model creation/deployment
  - **`owned_by`** (string): Always `"local"` for self-hosted models

## Get Model Response

### Endpoint: GET /v1/models/{model}

Returns detailed information about a specific model.

### Request Parameters

- **`model`** (path parameter): The model identifier to retrieve

### Response Format

```json
{
  "id": "qwen3-coder-30b",
  "object": "model",
  "created": 1703123456,
  "owned_by": "local"
}
```

### Error Responses

#### Model Not Found
```json
{
  "detail": "Model 'nonexistent-model' not found"
}
```

Status code: `404 Not Found`

## Model Capabilities and Usage

### Text Generation Models

Used by:
- **Chat Completions** (`/v1/chat/completions`): Conversational AI with optional RAG
- **Completions** (`/v1/completions`): Legacy text completion tasks

Example usage:
```python
import requests

# List available models
response = requests.get("http://localhost:8000/v1/models")
models = response.json()["data"]
print("Available models:", [model["id"] for model in models])

# Use specific model for generation
chat_response = requests.post("http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-coder-30b",
        "messages": [{"role": "user", "content": "Write a Python function to sort a list"}]
    })
```

### Embedding Models

Used by:
- **Embeddings** (`/v1/embeddings`): Generate vector representations
- **RAG Search**: Document indexing and retrieval

Example usage:
```python
# Generate embeddings for documents
docs = ["Machine learning algorithms", "Neural network architectures"]
response = requests.post("http://localhost:8000/v1/embeddings",
    json={"input": docs, "model": "qwen3-embedding-4b"})
embeddings = [item["embedding"] for item in response.json()["data"]]
```

### Reranking Models

Used by:
- **RAG Reranking**: Improve relevance of retrieved documents
- **Search Quality Enhancement**: Post-processing of search results

Example integration:
```python
# Rerank retrieved documents
query = "natural language processing"
candidates = ["NLP techniques", "Machine learning", "Deep learning papers"]
# Implementation depends on specific reranking endpoint
```

## Model Management

### Adding New Models

To add new models to the deployment:

1. **Model Acquisition**: Obtain the model files/weights
2. **Backend Integration**: Configure llama.cpp or compatible server
3. **API Registration**: Update the models endpoint to include new model
4. **Testing**: Verify model works with all relevant endpoints

### Model Configuration

Models are registered in the gateway with:
- Unique identifier
- Model type classification
- Capability flags
- Resource requirements

### Performance Considerations

- **Memory Usage**: Larger models require more VRAM
- **Inference Speed**: Model size affects response latency
- **Concurrent Requests**: Resource sharing across multiple requests

## Authentication

The Models API requires authentication for certain operations, though basic model listing may be publicly accessible depending on deployment configuration.

## Error Handling

- **Network Issues**: Backend connectivity problems
- **Model Unavailable**: Temporary model loading issues
- **Invalid Requests**: Malformed model identifiers

## Future Extensions

Potential enhancements:
- **Model Metadata**: Detailed capability descriptions
- **Performance Metrics**: Latency and throughput information
- **Usage Statistics**: Request counts and resource usage
- **Health Monitoring**: Model availability status
- **Dynamic Loading**: On-demand model loading/unloading

## Integration Examples

### Model Selection Logic

```python
def select_model_for_task(task_type: str) -> str:
    """Select appropriate model based on task"""
    model_mapping = {
        "code_generation": "qwen3-coder-30b",
        "embedding": "qwen3-embedding-4b",
        "reranking": "qwen3-reranker-4b"
    }
    return model_mapping.get(task_type, "qwen3-coder-30b")

def get_available_models(endpoint: str) -> List[str]:
    """Fetch available models from API"""
    response = requests.get(f"{endpoint}/v1/models")
    return [model["id"] for model in response.json()["data"]]
```

### Health Check Integration

```python
def check_model_health(model_id: str, endpoint: str) -> bool:
    """Check if a model is available and responding"""
    try:
        response = requests.get(f"{endpoint}/v1/models/{model_id}")
        return response.status_code == 200
    except:
        return False

def get_healthy_models(endpoint: str) -> List[str]:
    """Get list of currently healthy models"""
    all_models = get_available_models(endpoint)
    healthy = []
    for model in all_models:
        if check_model_health(model, endpoint):
            healthy.append(model)
    return healthy
```

This API provides essential model discovery capabilities for applications using the local AI gateway.