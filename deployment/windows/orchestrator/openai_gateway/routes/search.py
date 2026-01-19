"""
Vector search API routes
"""
from fastapi import APIRouter, HTTPException, Header
from qdrant_client.models import FieldCondition, MatchValue, Filter
from typing import Optional

from ..models import SearchRequest
from ..config import COLLECTION_NAME
from ..clients.llamacpp_client import llamacpp_client
from ..clients.qdrant_client import qdrant_vector_client

router = APIRouter()


@router.post("/v1/search")
async def search_endpoint(request: SearchRequest):
    """Search vector store"""
    try:
        query_embedding = await llamacpp_client.get_embedding(request.query)

        # Build Qdrant filter if provided
        qdrant_filter = None
        if request.filter:
            # Simple filter conversion - extend as needed
            conditions = []
            for key, value in request.filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                qdrant_filter = Filter(must=conditions)

        search_results = qdrant_vector_client.search(
            COLLECTION_NAME,
            query_embedding,
            limit=request.top_k,
            query_filter=qdrant_filter
        )

        return {
            "results": [
                {
                    "document": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                    "score": hit.score,
                    "id": hit.id
                }
                for hit in search_results
            ],
            "total": len(search_results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")