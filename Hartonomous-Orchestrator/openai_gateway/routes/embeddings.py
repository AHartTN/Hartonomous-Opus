"""
Embeddings API routes
"""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from ..models import EmbeddingRequest
from ..clients.llamacpp_client import llamacpp_client
from ..utils.response_formatters import create_embedding_response

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible embeddings endpoint"""
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

    return create_embedding_response([item["embedding"] for item in embeddings_data], request.model, inputs)