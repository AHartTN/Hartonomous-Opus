"""
Main application entry point for OpenAI Gateway
"""
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import initialize_collections
from .middleware.auth import AuthMiddleware

# Import all route modules
from .routes import (
    chat, completions, embeddings, ingestion, search, rerank,
    models_endpoints, collections, stubs, moderations, files, fine_tuning, assistants,
    threads, runs, vector_stores, batch
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI-compatible RAG Gateway",
    version="2.0.0",
    description="An OpenAI API-compatible gateway with RAG capabilities using llama.cpp and Qdrant"
)

# Add middleware
app.add_middleware(AuthMiddleware)


# Custom exception handlers for OpenAI-compatible error responses

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with OpenAI-compatible format"""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request body",
                "type": "invalid_request_error",
                "param": None,
                "code": None
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with OpenAI-compatible format"""
    # Map HTTP status codes to OpenAI error types
    error_types = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_denied_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "internal_error"
    }

    error_type = error_types.get(exc.status_code, "api_error")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": error_type,
                "param": getattr(exc, 'param', None),
                "code": getattr(exc, 'code', None)
            }
        }
    )

# Include all routers
app.include_router(chat.router)
app.include_router(completions.router)
app.include_router(embeddings.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(rerank.router)
app.include_router(models_endpoints.router)
app.include_router(collections.router)
app.include_router(stubs.router)
app.include_router(moderations.router)
app.include_router(files.router)
app.include_router(fine_tuning.router)
app.include_router(assistants.router)
app.include_router(threads.router)
app.include_router(runs.router)
app.include_router(vector_stores.router)
app.include_router(batch.router)


@app.get("/health")
async def health_check():
    """Health check endpoint returning status of all backend services.

    Returns:
        dict: Health status including:
            - status: Overall health ("healthy" if all services are up)
            - backends: Status of generative, embedding, and reranker services
            - vector_store: Status and info of the Qdrant vector database
    """
    from .clients.llamacpp_client import llamacpp_client
    from .clients.qdrant_client import qdrant_vector_client
    from .config import COLLECTION_NAME

    status = {
        "status": "healthy",
        "backends": {},
        "vector_store": {}
    }

    # Check Qdrant vector database
    try:
        collection_info = qdrant_vector_client.get_collection_info(COLLECTION_NAME)
        status["vector_store"] = {
            "name": COLLECTION_NAME,
            "count": collection_info.points_count if collection_info else 0,
            "status": "up"
        }
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        status["vector_store"] = {
            "name": COLLECTION_NAME,
            "status": "down"
        }

    # Check llama.cpp backend services
    for name in ["generative", "embedding", "reranker"]:
        try:
            status["backends"][name] = "up" if await llamacpp_client.health_check(name) else "down"
        except Exception as e:
            logger.warning(f"Backend {name} health check failed: {e}")
            status["backends"][name] = "down"

    # Update overall status if any service is down
    if any(s == "down" for s in status["backends"].values()) or status["vector_store"]["status"] == "down":
        status["status"] = "degraded"

    return status


@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup.

    This event handler runs when the FastAPI application starts up.
    It initializes the vector database collections and sets up
    auto-discovery of available collections for multi-collection search.
    """
    logger.info("Initializing OpenAI Gateway...")
    initialize_collections()
    logger.info("OpenAI Gateway initialized successfully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700, log_level="info")