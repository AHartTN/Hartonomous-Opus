"""
Response formatting utilities for OpenAI-compatible responses
"""
import json
import time
import uuid
from typing import Dict, Any, AsyncGenerator, List
from ..models import CompletionChoice, CompletionUsage, CompletionResponse


def create_chat_completion_response(result: Dict[str, Any], request_model: str, logprobs: bool = False, seed: int = None) -> Dict[str, Any]:
    """Create OpenAI-compatible chat completion response"""
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result.get("content", "")
            },
            "logprobs": result.get("logprobs") if logprobs else None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": result.get("tokens_evaluated", 0),
            "completion_tokens": result.get("tokens_predicted", 0),
            "total_tokens": result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0)
        }
    }

    if seed is not None:
        response["system_fingerprint"] = f"fp_{seed}_{uuid.uuid4().hex[:8]}"

    return response


def create_completion_response(result: Dict[str, Any], request_model: str, prompt: str, logprobs: bool = False, echo: bool = False) -> CompletionResponse:
    """Create OpenAI-compatible text completion response"""
    completion_text = result.get("content", "")
    if echo and isinstance(prompt, str):
        completion_text = prompt + completion_text

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request_model,
        choices=[
            CompletionChoice(
                text=completion_text,
                index=0,
                logprobs=result.get("logprobs") if logprobs else None,
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=result.get("tokens_evaluated", 0),
            completion_tokens=result.get("tokens_predicted", 0),
            total_tokens=result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0)
        )
    )


def create_batch_completion_response(results_and_prompts: List[tuple], request_model: str, logprobs: bool = False, echo: bool = False) -> CompletionResponse:
    """Create OpenAI-compatible text completion response for multiple prompts"""
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, (result, prompt) in enumerate(results_and_prompts):
        completion_text = result.get("content", "")
        if echo and isinstance(prompt, str):
            completion_text = prompt + completion_text

        choices.append(CompletionChoice(
            text=completion_text,
            index=idx,
            logprobs=result.get("logprobs") if logprobs else None,
            finish_reason="stop"
        ))

        total_prompt_tokens += result.get("tokens_evaluated", 0)
        total_completion_tokens += result.get("tokens_predicted", 0)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request_model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens
        )
    )


def create_embedding_response(embeddings: List[List[float]], request_model: str, input_texts: List[str]) -> Dict[str, Any]:
    """Create OpenAI-compatible embedding response"""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": idx
            }
            for idx, embedding in enumerate(embeddings)
        ],
        "model": request_model,
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in input_texts),
            "total_tokens": sum(len(text.split()) for text in input_texts)
        }
    }


async def create_streaming_chat_generator(stream_generator, request_model: str, logprobs: bool = False) -> AsyncGenerator[str, None]:
    """Create streaming response generator for chat completions"""
    async for chunk in stream_generator:
        if chunk.startswith(b"data: "):
            try:
                data = json.loads(chunk[6:])
                openai_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": data.get("content", "")},
                        "logprobs": data.get("logprobs") if logprobs else None,
                        "finish_reason": "stop" if data.get("stop", False) else None
                    }]
                }
                yield f"data: {json.dumps(openai_chunk)}\n\n"

                if data.get("stop", False):
                    yield "data: [DONE]\n\n"
            except json.JSONDecodeError:
                pass


async def create_streaming_completion_generator(stream_generator, request_model: str, logprobs: bool = False) -> AsyncGenerator[str, None]:
    """Create streaming response generator for text completions"""
    async for chunk in stream_generator:
        if chunk.startswith(b"data: "):
            try:
                data = json.loads(chunk[6:])
                openai_chunk = {
                    "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request_model,
                    "choices": [{
                        "text": data.get("content", ""),
                        "index": 0,
                        "logprobs": data.get("logprobs") if logprobs else None,
                        "finish_reason": "stop" if data.get("stop", False) else None
                    }]
                }
                yield f"data: {json.dumps(openai_chunk)}\n\n"

                if data.get("stop", False):
                    yield "data: [DONE]\n\n"
            except json.JSONDecodeError:
                pass