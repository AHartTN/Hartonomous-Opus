"""
Models API routes
"""
from fastapi import APIRouter, HTTPException
import time

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {"id": "qwen3-coder-30b", "object": "model", "created": int(time.time()), "owned_by": "local"},
            {"id": "qwen3-embedding-4b", "object": "model", "created": int(time.time()), "owned_by": "local"},
            {"id": "qwen3-reranker-4b", "object": "model", "created": int(time.time()), "owned_by": "local"}
        ]
    }


@router.get("/v1/models/{model}")
async def get_model(model: str):
    """Get information about a specific model"""
    # Available models
    available_models = {
        "qwen3-coder-30b": {"id": "qwen3-coder-30b", "object": "model", "created": int(time.time()), "owned_by": "local"},
        "qwen3-embedding-4b": {"id": "qwen3-embedding-4b", "object": "model", "created": int(time.time()), "owned_by": "local"},
        "qwen3-reranker-4b": {"id": "qwen3-reranker-4b", "object": "model", "created": int(time.time()), "owned_by": "local"}
    }

    if model not in available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    return available_models[model]