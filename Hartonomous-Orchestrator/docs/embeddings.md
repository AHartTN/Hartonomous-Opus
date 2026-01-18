# Embeddings API

The Embeddings API endpoint generates vector representations (embeddings) for text inputs, providing OpenAI-compatible functionality for local self-hosted models. This endpoint is essential for semantic search, document retrieval, and similarity-based operations.

## Endpoint

```
POST /v1/embeddings
```

## Description

Transforms text inputs into high-dimensional vector representations that capture semantic meaning. These embeddings can be used for various NLP tasks including:

- Semantic similarity comparison
- Document clustering and classification
- Information retrieval and search
- Retrieval-Augmented Generation (RAG) context retrieval
- Recommendation systems
- Anomaly detection

The endpoint supports both single text inputs and batch processing of multiple texts.

## Request Parameters

### Required Parameters

- **`input`** (string or array of strings): The text(s) to embed. Maximum length varies by model but is typically constrained by context window limits.
  - Single string: `"Hello, world!"`
  - Array: `["First text", "Second text", "Third text"]`

### Optional Parameters

- **`model`** (string): Identifier for the embedding model to use. Must match an available model in the local deployment. If not specified, uses the default configured model.
- **`dimensions`** (integer): Reduces the output embedding dimensionality. This parameter allows truncation of the embedding vector to a specified length, which can:
  - Reduce storage requirements
  - Improve computational efficiency
  - Maintain semantic quality for many use cases
  - Must be less than or equal to the model's native embedding dimension

## Request Examples

### Single Text Embedding
```json
{
  "input": "The quick brown fox jumps over the lazy dog.",
  "model": "text-embedding-ada-002",
  "dimensions": 512
}
```

### Batch Embedding Processing
```json
{
  "input": [
    "Natural language processing with local models",
    "Vector embeddings for semantic search",
    "Retrieval-augmented generation techniques",
    "OpenAI-compatible API interfaces"
  ],
  "model": "local-embedding-model-v1",
  "dimensions": 768
}
```

### Large Batch Example
```json
{
  "input": [
    "First document text...",
    "Second document text...",
    "Third document text...",
    "Fourth document text...",
    "Fifth document text..."
  ],
  "model": "text-embedding-large"
}
```

## Response Format

### Successful Response
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        -0.006929283,
        -0.005336422,
        -0.009327292,
        -0.024047505,
        0.0005430635,
        0.014238915,
        0.017470544,
        0.012780987,
        -0.010690838,
        0.008867529,
        // ... (truncated for brevity, actual length depends on model and dimensions)
      ],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [
        -0.003456789,
        0.002345678,
        -0.008765432,
        // ... (additional vector elements)
      ],
      "index": 1
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

### Response Fields

- **`object`** (string): Always `"list"` for embedding responses
- **`data`** (array): Array of embedding objects
  - **`object`** (string): Always `"embedding"`
  - **`embedding`** (array of floats): The embedding vector
  - **`index`** (integer): Position of this embedding in the request input array
- **`model`** (string): The model used for generation
- **`usage`** (object): Token usage information
  - **`prompt_tokens`** (integer): Tokens in the input text(s)
  - **`total_tokens`** (integer): Total tokens processed (same as prompt_tokens for embeddings)

## Features and Capabilities

### Batch Processing Optimization

- Processes multiple texts efficiently in a single request
- Maintains order correspondence between input texts and output embeddings
- Reduces network overhead for large-scale embedding generation

### Dimensionality Control

- Allows dynamic dimension reduction without retraining models
- Useful for adapting embeddings to specific application requirements
- Can improve performance in downstream tasks

### Memory and Performance Considerations

- Large batch sizes may impact response times
- Consider splitting very large requests into smaller batches
- Embedding generation is computationally intensive; monitor backend resources

## Error Responses

### Invalid Input
```json
{
  "error": {
    "message": "Invalid input: input must be a string or array of strings",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

### Model Not Found
```json
{
  "error": {
    "message": "Model 'nonexistent-model' not found",
    "type": "invalid_request_error",
    "code": 404
  }
}
```

### Backend Unavailable
```json
{
  "error": {
    "message": "Embedding service temporarily unavailable",
    "type": "service_unavailable_error",
    "code": 503
  }
}
```

## Authentication and Security

Requires valid API key authentication via the `Authorization` header:
```
Authorization: Bearer your-api-key-here
```

## Usage Patterns and Best Practices

### Semantic Similarity Search
```python
import requests
import numpy as np

def find_similar_texts(query_text, documents, api_key, endpoint="http://localhost:8000"):
    # Generate embeddings for query and documents
    payload = {
        "input": [query_text] + documents,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(f"{endpoint}/v1/embeddings",
                           json=payload,
                           headers={"Authorization": f"Bearer {api_key}"})
    
    embeddings = [item["embedding"] for item in response.json()["data"]]
    query_embedding = np.array(embeddings[0])
    doc_embeddings = np.array(embeddings[1:])
    
    # Calculate cosine similarities
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Return sorted results
    ranked_indices = np.argsort(similarities)[::-1]
    return [(documents[i], similarities[i]) for i in ranked_indices]

# Example usage
documents = ["Machine learning algorithms", "Neural network architectures", "Data processing pipelines"]
results = find_similar_texts("artificial intelligence models", documents, "your-api-key")
```

### Integration with Vector Databases

```python
import requests

def store_embeddings_in_qdrant(texts, collection_name, api_key):
    # Generate embeddings
    response = requests.post("http://localhost:8000/v1/embeddings",
                           json={"input": texts, "model": "text-embedding-ada-002"},
                           headers={"Authorization": f"Bearer {api_key}"})
    
    embeddings = [item["embedding"] for item in response.json()["data"]]
    
    # Store in vector database (implementation depends on client)
    points = [
        {"id": i, "vector": emb, "payload": {"text": text}}
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    
    # qdrant_client.upsert(collection_name, points)
    return points
```

### RAG Context Retrieval

Embeddings are automatically utilized by the Chat Completions endpoint when RAG is enabled, providing context-aware responses through:

1. Query embedding generation
2. Similarity search against document embeddings
3. Context injection into conversation prompts
4. Enhanced response generation

## Performance Optimization

- **Batch Processing**: Group multiple texts into single requests
- **Dimension Reduction**: Use `dimensions` parameter for storage optimization
- **Caching**: Cache embeddings for frequently accessed texts
- **Preprocessing**: Clean and normalize text inputs for better embeddings

## Limitations

- Maximum input length constrained by model context window
- Batch size limits may apply for very large requests
- Embedding quality depends on the underlying model
- Real-time performance may vary with model size and hardware