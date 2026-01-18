"""
Stub endpoints for unimplemented OpenAI API features
"""
from fastapi import APIRouter, HTTPException
import logging
import hashlib

logger = logging.getLogger(__name__)

router = APIRouter()

_stub_counter = 0

def create_stub_handler(endpoint_name: str, detail: str):
    """Create a stub handler function for unimplemented endpoints"""
    global _stub_counter

    async def stub_handler(**kwargs):
        logger.info(f"Stub endpoint called: {endpoint_name}")
        raise HTTPException(status_code=501, detail=detail)

    # Generate unique function name using hash of endpoint
    endpoint_hash = hashlib.md5(endpoint_name.encode()).hexdigest()[:8]
    stub_handler.__name__ = f"stub_{endpoint_hash}_{_stub_counter}"
    _stub_counter += 1
    return stub_handler


# Image generation endpoints
router.add_api_route(
    "/v1/images/generations",
    create_stub_handler("/v1/images/generations", "Image generation is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/images/edits",
    create_stub_handler("/v1/images/edits", "Image editing is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/images/variations",
    create_stub_handler("/v1/images/variations", "Image variations are not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

# Audio endpoints
router.add_api_route(
    "/v1/audio/transcriptions",
    create_stub_handler("/v1/audio/transcriptions", "Audio transcription is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/audio/translations",
    create_stub_handler("/v1/audio/translations", "Audio translation is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/audio/speech",
    create_stub_handler("/v1/audio/speech", "Text-to-speech is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)



# Files endpoints - now implemented in routes/files.py

# Fine-tuning endpoints
router.add_api_route(
    "/v1/fine_tuning/jobs",
    create_stub_handler("/v1/fine_tuning/jobs", "Fine-tuning is not implemented in the local llama.cpp gateway"),
    methods=["POST", "GET"]
)

router.add_api_route(
    "/v1/fine_tuning/jobs/{fine_tuning_job_id}",
    create_stub_handler("/v1/fine_tuning/jobs/{fine_tuning_job_id}", "Fine-tuning is not implemented in the local llama.cpp gateway"),
    methods=["GET"]
)

router.add_api_route(
    "/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
    create_stub_handler("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel", "Fine-tuning is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/fine_tuning/jobs/{fine_tuning_job_id}/events",
    create_stub_handler("/v1/fine_tuning/jobs/{fine_tuning_job_id}/events", "Fine-tuning is not implemented in the local llama.cpp gateway"),
    methods=["GET"]
)

# Assistants endpoints - now implemented in routes/assistants.py

# Threads endpoints
router.add_api_route(
    "/v1/threads",
    create_stub_handler("/v1/threads", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/threads/{thread_id}",
    create_stub_handler("/v1/threads/{thread_id}", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["GET", "POST", "DELETE"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/runs",
    create_stub_handler("/v1/threads/{thread_id}/runs", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["POST", "GET"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/runs/{run_id}",
    create_stub_handler("/v1/threads/{thread_id}/runs/{run_id}", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["GET", "POST"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/runs/{run_id}/cancel",
    create_stub_handler("/v1/threads/{thread_id}/runs/{run_id}/cancel", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/runs/{run_id}/steps",
    create_stub_handler("/v1/threads/{thread_id}/runs/{run_id}/steps", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["GET"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}",
    create_stub_handler("/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["GET"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/messages",
    create_stub_handler("/v1/threads/{thread_id}/messages", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["POST", "GET"]
)

router.add_api_route(
    "/v1/threads/{thread_id}/messages/{message_id}",
    create_stub_handler("/v1/threads/{thread_id}/messages/{message_id}", "Threads API is not implemented in the local llama.cpp gateway"),
    methods=["GET", "POST", "DELETE"]
)

# Vector stores endpoints
router.add_api_route(
    "/v1/vector_stores",
    create_stub_handler("/v1/vector_stores", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["POST", "GET"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}",
    create_stub_handler("/v1/vector_stores/{vector_store_id}", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["GET", "POST", "DELETE"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}/files",
    create_stub_handler("/v1/vector_stores/{vector_store_id}/files", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["POST", "GET"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    create_stub_handler("/v1/vector_stores/{vector_store_id}/files/{file_id}", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["GET", "POST", "DELETE"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}/file_batches",
    create_stub_handler("/v1/vector_stores/{vector_store_id}/file_batches", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["POST", "GET"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
    create_stub_handler("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["GET"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
    create_stub_handler("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

router.add_api_route(
    "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
    create_stub_handler("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files", "Vector stores API is not implemented in the local llama.cpp gateway"),
    methods=["GET"]
)

# Realtime API endpoints
router.add_api_route(
    "/v1/realtime",
    create_stub_handler("/v1/realtime", "Realtime API is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

# Responses API endpoints
router.add_api_route(
    "/v1/responses",
    create_stub_handler("/v1/responses", "Responses API is not implemented in the local llama.cpp gateway"),
    methods=["POST"]
)

