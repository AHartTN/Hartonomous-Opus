# Completions API (Legacy)

The Completions API provides legacy text completion functionality compatible with OpenAI's completions endpoint, designed for completing text prompts with local self-hosted models.

## Endpoint

```
POST /v1/completions
```

## Description

Generates text completions for given prompts. This is the legacy completion endpoint that predates the Chat Completions API. It is suitable for single-turn text generation tasks where conversational formatting is not required.

**Note**: For conversational AI applications, consider using the Chat Completions API instead, which provides better formatting and context handling.

## Request Parameters

### Required Parameters

- **`prompt`** (string or array of strings): The text prompt(s) to complete. Can be a single string or an array of strings for batch processing.

### Optional Parameters

- **`model`** (string): The model to use for completion. Defaults to the configured default model.
- **`max_tokens`** (integer): Maximum number of tokens to generate. Default varies by model.
- **`temperature`** (float): Controls randomness in the response. Range: 0.0 to 2.0. Default: 1.0.
- **`top_p`** (float): Nucleus sampling parameter. Range: 0.0 to 1.0.
- **`stream`** (boolean): Enable streaming responses. Default: false. Note: Streaming only supported for single prompts.
- **`stop`** (string or array): Sequence(s) where the API will stop generating further tokens.
- **`frequency_penalty`** (float): Penalize frequent tokens. Range: -2.0 to 2.0. Default: 0.0.
- **`presence_penalty`** (float): Penalize new tokens based on presence in text. Range: -2.0 to 2.0. Default: 0.0.
- **`logprobs`** (integer): Include log probabilities in response. Default: null.
- **`echo`** (boolean): Include the prompt in the completion response. Default: false.

### Unsupported Parameters (Logged as Warnings)

The following OpenAI parameters are not supported by the llama.cpp backend and will generate warnings:

- **`best_of`** (integer): Generate multiple completions and return the best.
- **`logit_bias`** (object): Modify logit scores for specific tokens.
- **`suffix`** (string): Suffix to append after completion.

## Request Examples

### Single Prompt Completion
```json
{
  "model": "text-davinci-003",
  "prompt": "The future of artificial intelligence is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 1.0
}
```

### Batch Prompt Completion
```json
{
  "model": "local-model",
  "prompt": [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The quick brown fox"
  ],
  "max_tokens": 50,
  "temperature": 0.8
}
```

### Streaming Completion
```json
{
  "model": "local-model",
  "prompt": "Write a short story about a robot who learns to paint:",
  "max_tokens": 200,
  "temperature": 0.9,
  "stream": true
}
```

## Response Format

### Non-Streaming Response (Single Prompt)
```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "local-model",
  "choices": [
    {
      "text": " likely to be shaped by advances in machine learning, quantum computing, and ethical AI development. Organizations worldwide are investing heavily in AI research, focusing on applications in healthcare, transportation, and environmental sustainability.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 45,
    "total_tokens": 52
  }
}
```

### Non-Streaming Response (Multiple Prompts)
```json
{
  "id": "cmpl-batch-123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "local-model",
  "choices": [
    {
      "text": " there was a magical kingdom...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    },
    {
      "text": " a young Jedi named Luke Skywalker...",
      "index": 1,
      "logprobs": null,
      "finish_reason": "stop"
    },
    {
      "text": " jumps over the lazy dog. This pangram...",
      "index": 2,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 28,
    "total_tokens": 43
  }
}
```

### Streaming Response

Returns Server-Sent Events (SSE) with partial completions. Each event contains JSON with `"object": "text_completion"` and incremental text updates.

```
data: {"id": "cmpl-123", "object": "text_completion", "created": 1677652288, "model": "local-model", "choices": [{"text": "The", "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-123", "object": "text_completion", "created": 1677652288, "model": "local-model", "choices": [{"text": " future", "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-123", "object": "text_completion", "created": 1677652288, "model": "local-model", "choices": [{"text": " of", "index": 0, "finish_reason": null}]}

...

data: {"id": "cmpl-123", "object": "text_completion", "created": 1677652288, "model": "local-model", "choices": [{"text": " AI", "index": 0, "finish_reason": "stop"}]}
```

## Response Fields

- **`id`** (string): Unique identifier for the completion request
- **`object`** (string): Always `"text_completion"`
- **`created`** (integer): Unix timestamp of creation
- **`model`** (string): Model used for generation
- **`choices`** (array): Array of completion choices
  - **`text`** (string): The completed text
  - **`index`** (integer): Index of the choice (useful for multiple prompts)
  - **`logprobs`** (object or null): Log probability information if requested
  - **`finish_reason`** (string): Reason completion stopped (`"stop"`, `"length"`, `"content_filter"`, or null)
- **`usage`** (object): Token usage statistics
  - **`prompt_tokens`** (integer): Tokens in the prompt(s)
  - **`completion_tokens`** (integer): Tokens generated
  - **`total_tokens`** (integer): Total tokens processed

## Features

### Batch Processing

- Supports multiple prompts in a single request
- Processes prompts sequentially for optimal resource usage
- Returns results in the same order as input prompts

### Streaming Support

- Real-time text generation for single prompts
- Compatible with Server-Sent Events (SSE)
- Useful for interactive applications and real-time user feedback

### Parameter Compatibility

- Maps OpenAI parameters to llama.cpp equivalents
- Logs warnings for unsupported parameters
- Maintains backward compatibility with legacy applications

## Error Handling

### Common Error Responses

#### Invalid Prompt Format
```json
{
  "error": {
    "message": "Invalid prompt format",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

#### Streaming with Multiple Prompts Not Supported
```json
{
  "error": {
    "message": "Streaming is only supported for single prompts",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

#### Backend Unavailable
```json
{
  "error": {
    "message": "Completion service temporarily unavailable",
    "type": "service_unavailable_error",
    "code": 503
  }
}
```

## Authentication

Requires API key in the `Authorization` header (Bearer token format).

## Usage Guidelines

### When to Use Completions vs Chat Completions

**Use Completions for:**
- Single-turn text generation
- Legacy applications requiring OpenAI compatibility
- Creative writing, code generation, or other non-conversational tasks
- Simple prompt-response scenarios

**Use Chat Completions for:**
- Multi-turn conversations
- Applications requiring message history
- Better formatting and context management
- RAG-enhanced responses

### Best Practices

- **Prompt Engineering**: Craft clear, specific prompts for best results
- **Token Limits**: Monitor token usage to avoid unexpected truncation
- **Temperature Tuning**: Adjust temperature based on desired creativity vs. consistency
- **Stop Sequences**: Use stop sequences to control completion length and structure

### Performance Considerations

- Batch processing is more efficient than multiple individual requests
- Streaming may have higher latency for very short completions
- Large prompt arrays may impact response times

## Migration from OpenAI

This endpoint provides full compatibility with OpenAI's completions API, making migration straightforward:

1. Update the base URL to point to your local gateway
2. Ensure authentication headers are configured
3. No code changes required for basic usage
4. Review unsupported parameters for advanced use cases

## Limitations

- Streaming only supported for single prompts
- Some advanced OpenAI parameters not supported by llama.cpp backend
- No built-in content filtering or moderation
- Performance depends on local hardware capabilities