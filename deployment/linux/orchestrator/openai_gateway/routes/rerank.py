"""
Reranking API routes
"""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from ..models import RerankRequest
from ..clients.llamacpp_client import llamacpp_client

router = APIRouter()


@router.post("/v1/rerank")
async def rerank_endpoint(
    request: RerankRequest,
    authorization: Optional[str] = Header(None)
):
    """Reranking endpoint"""
    results = await llamacpp_client.rerank_documents(
        request.query,
        request.documents,
        request.top_n or len(request.documents)
    )

    return {
        "results": results,
        "model": request.model
    }