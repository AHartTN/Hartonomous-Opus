"""
Text completions API routes
"""
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional
import logging

from ..models import CompletionRequest
from ..clients.llamacpp_client import llamacpp_client
from ..utils.response_formatters import (
    create_completion_response,
    create_streaming_completion_generator
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible text completions endpoint"""
    logger.info(f"Completion request received: model={request.model}, stream={request.stream}, prompt_type={type(request.prompt).__name__}")

    # Handle both single prompt and array of prompts
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    logger.info(f"Processing {len(prompts)} prompt(s)")

    # Handle streaming with multiple prompts - OpenAI doesn't support this, so warn and take first prompt
    if request.stream and len(prompts) > 1:
        logger.warning("Streaming with multiple prompts not supported, using first prompt only")
        prompts = [prompts[0]]

    # Process multiple prompts sequentially for non-streaming
    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")

        # Build llama.cpp payload
        llamacpp_payload = {
            "prompt": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }

        # Add optional parameters
        if request.max_tokens:
            llamacpp_payload["n_predict"] = request.max_tokens

        if request.stop:
            llamacpp_payload["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop

        if request.frequency_penalty:
            llamacpp_payload["repeat_penalty"] = 1.0 + request.frequency_penalty

        if request.presence_penalty and request.presence_penalty != 0.0:
            # Note: llama.cpp may not support presence_penalty directly
            logger.warning("presence_penalty not fully supported by llama.cpp backend")

        if request.logprobs:
            llamacpp_payload["logprobs"] = request.logprobs

        if request.echo:
            llamacpp_payload["echo"] = request.echo

        if request.best_of and request.best_of > 1:
            logger.warning("best_of parameter not supported by llama.cpp backend")

        if request.logit_bias:
            logger.warning("logit_bias parameter not supported by llama.cpp backend")

        if request.suffix:
            logger.warning("suffix parameter not supported by llama.cpp backend")

        logger.info(f"About to call generative server for prompt {i+1}: payload keys: {list(llamacpp_payload.keys())}")

        # Handle streaming (only for single prompt)
        if request.stream:
            logger.info("Handling streaming completion response")
            try:
                generator = await llamacpp_client.generate_completion(llamacpp_payload, stream=True)
                logger.info("Streaming generator created successfully")
                streaming_generator = create_streaming_completion_generator(generator, request.model, request.logprobs)
                return StreamingResponse(streaming_generator, media_type="text/event-stream")
            except Exception as e:
                logger.error(f"Streaming completion error: {e}")
                raise

        # Non-streaming response
        logger.info("Handling non-streaming completion response")
        try:
            result = await llamacpp_client.generate_completion(llamacpp_payload)
            logger.info(f"Received result from generative server: content_length={len(result.get('content', ''))}, tokens_predicted={result.get('tokens_predicted', 0)}")
        except Exception as e:
            logger.error(f"Error calling generative server for prompt {i+1}: {e}")
            raise

        results.append((result, prompt))

    # Create combined response for multiple prompts
    from ..utils.response_formatters import create_batch_completion_response
    response = create_batch_completion_response(results, request.model, request.logprobs, request.echo)
    return response