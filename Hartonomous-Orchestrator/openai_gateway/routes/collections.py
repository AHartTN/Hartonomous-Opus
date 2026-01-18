"""
Collection management API routes
"""
from fastapi import APIRouter, HTTPException

from ..config import COLLECTION_NAME, VECTOR_SIZE, RAG_ENABLED, RAG_TOP_K, RAG_RERANK_TOP_N, SEARCH_COLLECTIONS
from ..clients.qdrant_client import qdrant_vector_client

router = APIRouter()


@router.delete("/v1/collection")
async def clear_collection():
    """Clear the vector store collection"""
    try:
        success = qdrant_vector_client.recreate_collection(COLLECTION_NAME, VECTOR_SIZE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to recreate collection")
        return {"status": "success", "message": f"Collection {COLLECTION_NAME} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing collection: {str(e)}")


@router.get("/v1/collection/stats")
async def collection_stats():
    """Get collection statistics"""
    collection_info = qdrant_vector_client.get_collection_info(COLLECTION_NAME)
    return {
        "name": COLLECTION_NAME,
        "count": collection_info.points_count if collection_info else 0,
        "vector_size": VECTOR_SIZE,
        "rag_enabled": RAG_ENABLED,
        "rag_top_k": RAG_TOP_K,
        "rag_rerank_top_n": RAG_RERANK_TOP_N,
        "search_collections": SEARCH_COLLECTIONS
    }


@router.get("/v1/collections")
async def list_collections():
    """List all available collections in Qdrant"""
    try:
        collections = qdrant_vector_client.list_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "vectors_count": qdrant_vector_client.get_collection_info(col.name).points_count,
                    "vector_size": (
                        qdrant_vector_client.get_collection_info(col.name).config.params.vectors.size
                        if hasattr(qdrant_vector_client.get_collection_info(col.name).config.params.vectors, 'size')
                        else "named_vectors"
                    ),
                    "in_search": col.name in SEARCH_COLLECTIONS
                }
                for col in collections
            ],
            "currently_searching": SEARCH_COLLECTIONS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")