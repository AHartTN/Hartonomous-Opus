"""
Chat completions API routes
"""
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional
import logging

from ..models import ChatCompletionRequest
from ..config import RAG_ENABLED, RAG_TOP_K, RAG_RERANK_TOP_N, COLLECTION_NAME, USE_OPUS_DB
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

    # ============================================================================
    # HARTONOMOUS PURE DATABASE-NATIVE GENERATION
    # No llama.cpp, no neural networks, just database queries
    # ============================================================================

    logger.info("Using Hartonomous database-native generation (THE DATABASE IS THE MODEL)")

    try:
        from ..clients.hartonomous_client import get_hartonomous_client
        hartonomous_client = get_hartonomous_client()

        # Extract starting text from prompt (use last user message)
        user_messages = [msg for msg in messages if msg.role == "user"]
        start_text = user_messages[-1].content if user_messages else "The"

        max_tokens = request.max_tokens or request.max_completion_tokens or 100

        if request.stream:
            # TODO: Implement streaming for Hartonomous
            logger.warning("Streaming not yet implemented for Hartonomous mode, using non-streaming")

        # Generate using database-native inference (no forward pass!)
        generated_text = hartonomous_client.generate_text(
            start_text=start_text,
            max_tokens=max_tokens
        )

        # Format as OpenAI response
        result = {
            "content": generated_text,
            "tokens_predicted": max_tokens,
            "stopped_word": False,
            "stopped_eos": True,
            "stopped_limit": False
        }

        logger.info(f"Hartonomous generation complete: {len(generated_text)} chars")
        response = create_chat_completion_response(result, request.model, request.logprobs, request.seed)
        return response

    except Exception as e:
        logger.error(f"Hartonomous generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hartonomous generation error: {str(e)}")