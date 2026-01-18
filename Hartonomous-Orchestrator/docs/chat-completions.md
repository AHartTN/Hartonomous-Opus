# Chat Completions API

The Chat Completions API endpoint provides conversational AI capabilities compatible with OpenAI's chat completion interface, designed for local self-hosted models.

## Endpoint

```
POST /v1/chat/completions
```

## Description

Generates conversational responses using chat-based language models. This endpoint supports multi-turn conversations, streaming responses, and optional Retrieval-Augmented Generation (RAG) for enhanced context-aware responses.

## Request Parameters

### Required Parameters

- **`messages`** (array): A list of messages comprising the conversation. Each message has a `role` ("system", "user", or "assistant") and `content` (string or array of content blocks).

### Optional Parameters

- **`model`** (string): The model to use for generation. Defaults to the configured default model.
- **`temperature`** (float): Controls randomness in the response. Range: 0.0 to 2.0. Default: 1.0.
- **`max_tokens`** (integer): Maximum number of tokens to generate. Alternative: `max_completion_tokens`.
- **`stream`** (boolean): Enable streaming responses. Default: false.
- **`top_p`** (float): Nucleus sampling parameter. Range: 0.0 to 1.0.
- **`stop`** (string or array): Sequences where the API will stop generating further tokens.
- **`seed`** (integer): Seed for deterministic sampling.
- **`frequency_penalty`** (float): Penalize frequent tokens. Range: -2.0 to 2.0.
- **`logprobs`** (boolean): Return log probabilities of the output tokens.
- **`top_logprobs`** (integer): Number of most likely tokens to return at each token position.

### Custom RAG Parameters

- **`rag_enabled`** (boolean): Override global RAG settings for this request.
- **`rag_top_k`** (integer): Number of documents to retrieve for RAG.
- **`rag_rerank_top_n`** (integer): Number of documents to rerank from retrieved set.

## Request Example

```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": false,
  "rag_enabled": true
}
```

## Response Format

### Non-Streaming Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  }
}
```

### Streaming Response

Returns Server-Sent Events (SSE) with partial responses. Each event contains a `data` field with JSON similar to the non-streaming format, but with `"object": "chat.completion.chunk"`.

## Features

### Retrieval-Augmented Generation (RAG)

When enabled, the endpoint automatically retrieves relevant context from a knowledge base to enhance responses:

- Searches for relevant documents based on the user's query
- Reranks retrieved documents for relevance
- Injects context into the conversation prompt
- Falls back gracefully if RAG is unavailable

### Streaming Support

Supports real-time streaming of responses for interactive applications.

### Parameter Mapping

Maps OpenAI-compatible parameters to the underlying llama.cpp backend parameters for seamless local model integration.

## Error Handling

- Returns standard HTTP status codes
- Provides detailed error messages in JSON format
- Handles backend connectivity issues gracefully

## Authentication

Requires API key in the `Authorization` header (Bearer token format).