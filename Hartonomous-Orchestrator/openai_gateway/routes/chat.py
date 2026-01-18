"""
Chat completions API routes
"""
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional
import logging

from ..models import ChatCompletionRequest
from ..config import RAG_ENABLED, RAG_TOP_K, RAG_RERANK_TOP_N, COLLECTION_NAME
from ..clients.llamacpp_client import llamacpp_client
from ..clients.qdrant_client import qdrant_vector_client
from ..rag.search import rag_search
from ..rag.prompt_builder import prompt_builder
from ..utils.text_processing import convert_messages_to_prompt
from ..utils.response_formatters import (
    create_chat_completion_response,
    create_streaming_chat_generator
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completions with automatic RAG"""
    logger.info(f"Chat completion request received: model={request.model}, stream={request.stream}, messages_count={len(request.messages)}")
    logger.info(f"Request details: temperature={request.temperature}, max_tokens={request.max_tokens}, rag_enabled={request.rag_enabled}")

    # Determine RAG settings: use request override if provided, otherwise global config
    use_rag = request.rag_enabled if request.rag_enabled is not None else RAG_ENABLED
    logger.info(f"RAG enabled: {use_rag}")
    top_k = request.rag_top_k or RAG_TOP_K  # Documents to retrieve
    rerank_top_n = request.rag_rerank_top_n or RAG_RERANK_TOP_N  # Documents to rerank

    # Check if knowledge base is available for RAG
    messages = request.messages
    collection_info = None
    try:
        collection_info = qdrant_vector_client.get_collection_info(COLLECTION_NAME)
        logger.info(f"Collection info retrieved: {collection_info.points_count if collection_info else 0} points")
    except Exception as e:
        logger.warning(f"Failed to get collection info (continuing without RAG): {e}")
        collection_info = None

    # Perform RAG if enabled and collection has documents
    if use_rag and collection_info and collection_info.points_count > 0:
        logger.info("RAG conditions met, performing search")
        # Extract query from the last user message in the conversation
        user_messages = [msg for msg in messages if msg.role == "user"]
        if user_messages:
            last_user_msg = user_messages[-1]
            # Handle both string and structured content formats
            query = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)

            logger.info(f"Performing RAG search for query: {query[:100]}...")
            try:
                # Search and rerank documents relevant to the query
                context_docs = await rag_search(query, top_k, rerank_top_n)
                logger.info(f"RAG search completed, got {len(context_docs) if context_docs else 0} documents")

                if context_docs:
                    logger.info(f"Found {len(context_docs)} relevant documents")
                    # Inject retrieved context into the conversation messages
                    messages = prompt_builder.build_prompt(messages, context_docs)
                else:
                    logger.info("No relevant context found")
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
                logger.info("Continuing without RAG context")
        else:
            logger.info("No user messages found for RAG")
    else:
        logger.info(f"RAG skipped: use_rag={use_rag}, collection_exists={collection_info is not None}, points={collection_info.points_count if collection_info else 0}")

    # Build payload for llama.cpp backend, converting OpenAI parameters to llama.cpp format
    llamacpp_payload = {
        "prompt": convert_messages_to_prompt(messages),  # Convert chat messages to single prompt string
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream,
    }

    # Map OpenAI parameters to llama.cpp equivalents
    if request.max_tokens or request.max_completion_tokens:
        llamacpp_payload["n_predict"] = request.max_tokens or request.max_completion_tokens

    if request.stop:
        # Ensure stop sequences are always a list
        llamacpp_payload["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop

    if request.seed is not None:
        llamacpp_payload["seed"] = request.seed

    if request.frequency_penalty:
        # Convert frequency penalty to llama.cpp repeat penalty (higher values = more penalty)
        llamacpp_payload["repeat_penalty"] = 1.0 + request.frequency_penalty

    if request.logprobs:
        # llama.cpp uses logprobs count, default to top_logprobs or 0
        llamacpp_payload["logprobs"] = request.top_logprobs or 0

    logger.info(f"About to call generative server: payload keys: {list(llamacpp_payload.keys())}")
    logger.info(f"Prompt content: {llamacpp_payload.get('prompt', '')[:200]}...")

    # Handle streaming
    if request.stream:
        logger.info("Handling streaming response")
        try:
            generator = await llamacpp_client.generate_completion(llamacpp_payload, stream=True)
            logger.info("Streaming generator created successfully")
            streaming_generator = create_streaming_chat_generator(generator, request.model, request.logprobs)
            return StreamingResponse(streaming_generator, media_type="text/event-stream")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    # Non-streaming response
    logger.info("Handling non-streaming response")
    try:
        result = await llamacpp_client.generate_completion(llamacpp_payload)
        logger.info(f"Received result from generative server: content_length={len(result.get('content', ''))}, tokens_predicted={result.get('tokens_predicted', 0)}")
    except Exception as e:
        logger.error(f"Error calling generative server: {e}")
        raise

    response = create_chat_completion_response(result, request.model, request.logprobs, request.seed)
    return response