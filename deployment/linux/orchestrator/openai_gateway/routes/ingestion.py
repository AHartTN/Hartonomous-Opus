"""
Document ingestion API routes
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from qdrant_client.models import PointStruct
from typing import Optional
import uuid

from ..models import IngestRequest
from ..config import COLLECTION_NAME
from ..clients.llamacpp_client import llamacpp_client
from ..clients.qdrant_client import qdrant_vector_client
from ..utils.text_processing import chunk_text

router = APIRouter()


@router.post("/v1/ingest")
async def ingest_documents(request: IngestRequest):
    """Ingest documents into vector store"""
    try:
        points = []

        for idx, doc in enumerate(request.documents):
            # Chunk if requested
            chunks = chunk_text(doc) if request.chunk else [doc]

            for chunk_idx, chunk in enumerate(chunks):
                # Get embedding
                embedding = await llamacpp_client.get_embedding(chunk)

                # Metadata
                metadata = request.metadata[idx] if request.metadata and idx < len(request.metadata) else {}
                metadata["chunk_index"] = chunk_idx
                metadata["total_chunks"] = len(chunks)
                metadata["text"] = chunk

                # ID
                if request.ids and idx < len(request.ids):
                    point_id = f"{request.ids[idx]}_chunk_{chunk_idx}"
                else:
                    point_id = str(uuid.uuid4())

                # Ensure vector is a flat list of floats
                if isinstance(embedding, list) and len(embedding) > 0:
                    # Check if it's nested (list of lists)
                    if isinstance(embedding[0], list):
                        embedding = embedding[0]  # Take first inner list

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                ))

        # Add to Qdrant collection
        success = qdrant_vector_client.upsert_points(COLLECTION_NAME, points)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to upsert points to vector store")

        collection_info = qdrant_vector_client.get_collection_info(COLLECTION_NAME)

        return {
            "status": "success",
            "documents_ingested": len(request.documents),
            "chunks_created": len(points),
            "collection_size": collection_info.points_count if collection_info else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@router.post("/v1/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a text file into vector store"""
    try:
        content = await file.read()
        text = content.decode("utf-8")

        result = await ingest_documents(IngestRequest(
            documents=[text],
            metadata=[{"filename": file.filename}],
            chunk=True
        ))

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion error: {str(e)}")