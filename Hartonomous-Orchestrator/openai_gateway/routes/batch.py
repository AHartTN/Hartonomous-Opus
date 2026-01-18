"""
Batch API routes
"""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional, List
import logging
import asyncio
import json
import time
import uuid

from ..models import (
    CreateBatchRequest, Batch, BatchListResponse, BatchRequestInput, BatchResult,
    ChatCompletionRequest, EmbeddingRequest, CompletionRequest, ModerationRequest
)
from ..routes.files import file_manager
from ..clients.llamacpp_client import llamacpp_client
from ..clients.qdrant_client import qdrant_vector_client
from ..utils.text_processing import convert_messages_to_prompt
from ..utils.response_formatters import (
    create_chat_completion_response,
    create_completion_response,
    create_embedding_response
)
from ..config import RAG_ENABLED, RAG_TOP_K, RAG_RERANK_TOP_N, COLLECTION_NAME
from ..rag.search import rag_search
from ..rag.prompt_builder import prompt_builder

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory batch storage (in production, this should be persisted)
batches_db = {}

# Supported endpoints for batch processing
SUPPORTED_ENDPOINTS = [
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/completions",
    "/v1/moderations"
]


async def process_batch_requests(batch_id: str, input_file_id: str, endpoint: str):
    """Process batch requests asynchronously"""
    try:
        # Get the input file
        input_file_obj = file_manager.get_file(input_file_id)
        if not input_file_obj:
            raise ValueError(f"Input file {input_file_id} not found")

        # Read file content
        input_file_path = file_manager.get_file_path(input_file_id)
        if not input_file_path:
            raise ValueError(f"Input file {input_file_id} path not found")

        with open(input_file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        lines = file_content.strip().split('\n')

        # Parse JSONL input
        requests = []
        for line in lines:
            if line.strip():
                try:
                    request = json.loads(line.strip())
                    requests.append(BatchRequestInput(**request))
                except Exception as e:
                    logger.error(f"Failed to parse request line: {e}")
                    continue

        # Update batch status to in_progress
        batch = batches_db[batch_id]
        batch.status = "in_progress"
        batch.in_progress_at = int(time.time())
        batch.request_counts.total = len(requests)

        results = []
        for req in requests:
            try:
                # Route to appropriate handler based on endpoint
                if endpoint == "/v1/chat/completions":
                    response = await process_chat_completion(req.body)
                elif endpoint == "/v1/embeddings":
                    response = await process_embeddings(req.body)
                elif endpoint == "/v1/completions":
                    response = await process_completions(req.body)
                elif endpoint == "/v1/moderations":
                    response = await process_moderations(req.body)
                else:
                    raise ValueError(f"Unsupported endpoint: {endpoint}")

                results.append(BatchResult(custom_id=req.custom_id, response=response))
                batch.request_counts.completed += 1

            except Exception as e:
                logger.error(f"Request {req.custom_id} failed: {e}")
                results.append(BatchResult(custom_id=req.custom_id, error={"message": str(e)}))
                batch.request_counts.failed += 1

        # Store results in a file using file_manager
        output_content = '\n'.join(json.dumps(result.dict()) for result in results)
        output_file_id = f"file_batch_output_{batch_id}"

        # Create a file-like object for the output
        from io import BytesIO
        output_file_like = BytesIO(output_content.encode('utf-8'))
        output_file_like.filename = f"batch_{batch_id}_results.jsonl"

        # Create the file using file_manager (need to adapt)
        # For now, save directly to file_manager's directory
        import os
        from ..config import FILES_DIR
        output_path = os.path.join(FILES_DIR, output_file_id)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)

        # Add to metadata
        file_manager.metadata[output_file_id] = {
            "id": output_file_id,
            "object": "file",
            "bytes": len(output_content.encode('utf-8')),
            "created_at": int(time.time()),
            "filename": f"batch_{batch_id}_results.jsonl",
            "purpose": "batch_output"
        }
        file_manager._save_metadata()

        # Update batch as completed
        batch.status = "completed"
        batch.completed_at = int(time.time())
        batch.output_file_id = output_file_id

        logger.info(f"Batch {batch_id} completed: {batch.request_counts.completed} succeeded, {batch.request_counts.failed} failed")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        batch = batches_db[batch_id]
        batch.status = "failed"
        batch.failed_at = int(time.time())


async def process_chat_completion(body):
    """Process chat completion request"""
    # Convert body to ChatCompletionRequest
    request = ChatCompletionRequest(**body)

    # Determine RAG settings
    use_rag = request.rag_enabled if request.rag_enabled is not None else RAG_ENABLED
    top_k = request.rag_top_k or RAG_TOP_K
    rerank_top_n = request.rag_rerank_top_n or RAG_RERANK_TOP_N

    # Check if knowledge base is available for RAG
    messages = request.messages
    collection_info = None
    try:
        collection_info = qdrant_vector_client.get_collection_info(COLLECTION_NAME)
    except Exception:
        collection_info = None

    # Perform RAG if enabled
    if use_rag and collection_info and collection_info.points_count > 0:
        user_messages = [msg for msg in messages if msg.role == "user"]
        if user_messages:
            last_user_msg = user_messages[-1]
            query = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)
            try:
                context_docs = await rag_search(query, top_k, rerank_top_n)
                if context_docs:
                    messages = prompt_builder.build_prompt(messages, context_docs)
            except Exception as e:
                logger.error(f"RAG search failed: {e}")

    # Build payload for llama.cpp backend
    llamacpp_payload = {
        "prompt": convert_messages_to_prompt(messages),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": False,  # Batches don't support streaming
    }

    if request.max_tokens or request.max_completion_tokens:
        llamacpp_payload["n_predict"] = request.max_tokens or request.max_completion_tokens

    if request.stop:
        llamacpp_payload["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop

    if request.seed is not None:
        llamacpp_payload["seed"] = request.seed

    if request.frequency_penalty:
        llamacpp_payload["repeat_penalty"] = 1.0 + request.frequency_penalty

    if request.logprobs:
        llamacpp_payload["logprobs"] = request.top_logprobs or 0

    # Call the backend
    result = await llamacpp_client.generate_completion(llamacpp_payload)

    # Format response
    response = create_chat_completion_response(result, request.model, request.logprobs, request.seed)
    return response


async def process_embeddings(body):
    """Process embeddings request"""
    request = EmbeddingRequest(**body)

    inputs = [request.input] if isinstance(request.input, str) else request.input

    embeddings_data = []
    for idx, text in enumerate(inputs):
        embedding = await llamacpp_client.get_embedding(text)

        # Handle dimension reduction if requested
        if request.dimensions and request.dimensions < len(embedding):
            embedding = embedding[:request.dimensions]

        embeddings_data.append({
            "object": "embedding",
            "embedding": embedding,
            "index": idx
        })

    response = create_embedding_response([item["embedding"] for item in embeddings_data], request.model, inputs)
    return response


async def process_completions(body):
    """Process completions request"""
    request = CompletionRequest(**body)

    # Build payload for llama.cpp backend
    llamacpp_payload = {
        "prompt": request.prompt if isinstance(request.prompt, str) else request.prompt[0],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": False,
    }

    if request.max_tokens:
        llamacpp_payload["n_predict"] = request.max_tokens

    if request.stop:
        llamacpp_payload["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop

    if request.frequency_penalty:
        llamacpp_payload["repeat_penalty"] = 1.0 + request.frequency_penalty

    # Call the backend
    result = await llamacpp_client.generate_completion(llamacpp_payload)

    # Format response
    response = create_completion_response(result, request.model, request.logprobs)
    return response


async def process_moderations(body):
    """Process moderations request"""
    # For now, return a simple response - moderation would need actual implementation
    return {
        "id": f"modr_{uuid.uuid4()}",
        "model": "text-moderation-stable",
        "results": [{"flagged": False, "categories": {"sexual": False, "hate": False, "harassment": False, "self-harm": False, "sexual/minors": False, "hate/threatening": False, "violence/graphic": False, "self-harm/intent": False, "self-harm/instructions": False, "harassment/threatening": False, "violence": False}, "category_scores": {"sexual": 0.0, "hate": 0.0, "harassment": 0.0, "self-harm": 0.0, "sexual/minors": 0.0, "hate/threatening": 0.0, "violence/graphic": 0.0, "self-harm/intent": 0.0, "self-harm/instructions": 0.0, "harassment/threatening": 0.0, "violence": 0.0}}]
    }


@router.post("/v1/batches", response_model=Batch)
async def create_batch(
    request: CreateBatchRequest,
    authorization: Optional[str] = Header(None)
):
    """Create a new batch job"""
    logger.info(f"Creating batch: endpoint={request.endpoint}, input_file={request.input_file_id}")

    # Validate endpoint
    if request.endpoint not in SUPPORTED_ENDPOINTS:
        raise HTTPException(status_code=400, detail=f"Unsupported endpoint: {request.endpoint}")

    # Validate input file exists
    input_file = file_manager.get_file(request.input_file_id)
    if not input_file:
        raise HTTPException(status_code=404, detail=f"Input file {request.input_file_id} not found")

    # Validate input file purpose
    if input_file.purpose != "batch":
        raise HTTPException(status_code=400, detail=f"Input file purpose must be 'batch', got '{input_file.purpose}'")

    # Create batch
    batch_id = f"batch_{uuid.uuid4()}"
    created_at = int(time.time())
    expires_at = created_at + (24 * 60 * 60)  # 24 hours from now

    batch = Batch(
        id=batch_id,
        endpoint=request.endpoint,
        input_file_id=request.input_file_id,
        completion_window=request.completion_window,
        created_at=created_at,
        expires_at=expires_at,
        metadata=request.metadata
    )

    batches_db[batch_id] = batch

    # Start async processing
    asyncio.create_task(process_batch_requests(batch_id, request.input_file_id, request.endpoint))

    return batch


@router.get("/v1/batches", response_model=BatchListResponse)
async def list_batches(
    after: Optional[str] = None,
    limit: Optional[int] = 20,
    authorization: Optional[str] = Header(None)
):
    """List batches"""
    logger.info(f"Listing batches: after={after}, limit={limit}")

    # Simple pagination - in production, implement proper pagination
    batches = list(batches_db.values())

    # Sort by created_at descending
    batches.sort(key=lambda x: x.created_at, reverse=True)

    # Apply pagination
    start_idx = 0
    if after:
        for i, batch in enumerate(batches):
            if batch.id == after:
                start_idx = i + 1
                break

    end_idx = start_idx + (limit or 20)
    paginated_batches = batches[start_idx:end_idx]

    response = BatchListResponse(
        data=paginated_batches,
        has_more=end_idx < len(batches)
    )

    if paginated_batches:
        response.first_id = paginated_batches[0].id
        response.last_id = paginated_batches[-1].id

    return response


@router.get("/v1/batches/{batch_id}", response_model=Batch)
async def get_batch(
    batch_id: str,
    authorization: Optional[str] = Header(None)
):
    """Get batch details"""
    logger.info(f"Getting batch: {batch_id}")

    if batch_id not in batches_db:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    return batches_db[batch_id]


@router.post("/v1/batches/{batch_id}/cancel", response_model=Batch)
async def cancel_batch(
    batch_id: str,
    authorization: Optional[str] = Header(None)
):
    """Cancel a batch"""
    logger.info(f"Cancelling batch: {batch_id}")

    if batch_id not in batches_db:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    batch = batches_db[batch_id]

    # Only allow cancelling if not already completed/failed
    if batch.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel batch with status: {batch.status}")

    batch.status = "cancelling"
    batch.cancelling_at = int(time.time())

    # In a real implementation, we'd need to actually cancel the running task
    # For now, just mark as cancelled
    batch.status = "cancelled"
    batch.cancelled_at = int(time.time())

    return batch


@router.get("/v1/batches/{batch_id}/results")
async def get_batch_results(
    batch_id: str,
    authorization: Optional[str] = Header(None)
):
    """Get batch results"""
    logger.info(f"Getting batch results: {batch_id}")

    if batch_id not in batches_db:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    batch = batches_db[batch_id]

    if batch.status != "completed":
        raise HTTPException(status_code=400, detail=f"Batch is not completed. Current status: {batch.status}")

    if not batch.output_file_id:
        raise HTTPException(status_code=404, detail="Batch results file not found")

    # Read the file content
    output_file_path = file_manager.get_file_path(batch.output_file_id)
    if not output_file_path or not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="Batch results file not found")

    with open(output_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content