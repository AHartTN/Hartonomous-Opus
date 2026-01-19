"""
Vector Stores API endpoints for OpenAI Gateway

This module implements the OpenAI Vector Stores API, providing CRUD operations
for vector stores, file attachments, and batch processing operations.
"""
import json
import os
import time
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError
from qdrant_client.models import PointStruct

from ..models import (
    CreateVectorStoreRequest, VectorStore, ModifyVectorStoreRequest,
    VectorStoreListResponse, CreateVectorStoreFileRequest, VectorStoreFile,
    VectorStoreFileListResponse, VectorStoreFileDeleteResponse,
    CreateVectorStoreFileBatchRequest, VectorStoreFileBatch,
    VectorStoreFileBatchListResponse, ChunkingStrategy
)
from ..clients.llamacpp_client import llamacpp_client
from ..clients.qdrant_client import qdrant_vector_client
from ..utils.text_processing import chunk_text
from ..routes.files import file_manager

router = APIRouter()

# Data directory for vector stores
DATA_DIR = Path("data")
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
VECTOR_STORE_FILES_DIR = DATA_DIR / "vector_store_files"
VECTOR_STORE_BATCHES_DIR = DATA_DIR / "vector_store_batches"

# Ensure directories exist
VECTOR_STORES_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_FILES_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_BATCHES_DIR.mkdir(parents=True, exist_ok=True)

# Metadata files
VECTOR_STORES_METADATA = VECTOR_STORES_DIR / "metadata.json"
VECTOR_STORE_FILES_METADATA = VECTOR_STORE_FILES_DIR / "metadata.json"
VECTOR_STORE_BATCHES_METADATA = VECTOR_STORE_BATCHES_DIR / "metadata.json"


def load_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    if not metadata_file.exists():
        return {}
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_metadata(metadata_file: Path, data: Dict[str, Any]) -> None:
    """Save metadata to JSON file."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(data, f, indent=2)


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def update_vector_store_file_counts(vector_store_id: str) -> None:
    """Update file counts for a vector store based on its files."""
    vector_stores = load_metadata(VECTOR_STORES_METADATA)
    vector_store_files = load_metadata(VECTOR_STORE_FILES_METADATA)

    if vector_store_id not in vector_stores:
        return

    # Count files by status
    file_counts = {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}

    for file_id, file_data in vector_store_files.items():
        if file_data.get("vector_store_id") == vector_store_id:
            status = file_data.get("status", "completed")
            if status in file_counts:
                file_counts[status] += 1
            file_counts["total"] += 1

    vector_stores[vector_store_id]["file_counts"] = file_counts
    save_metadata(VECTOR_STORES_METADATA, vector_stores)


@router.post("/v1/vector_stores", response_model=VectorStore)
async def create_vector_store(request: CreateVectorStoreRequest) -> VectorStore:
    """Create a new vector store."""
    vector_stores = load_metadata(VECTOR_STORES_METADATA)

    # Generate unique ID
    vector_store_id = generate_id("vs")

    # Create Qdrant collection for this vector store
    from ..config import VECTOR_SIZE
    collection_name = f"vector_store_{vector_store_id}"
    success = qdrant_vector_client.create_collection(collection_name, VECTOR_SIZE)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create vector store collection")

    # Create vector store object
    current_time = int(time.time())
    vector_store_data = {
        "id": vector_store_id,
        "object": "vector_store",
        "created_at": current_time,
        "name": request.name,
        "description": request.description,
        "usage_bytes": 0,
        "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        "status": "completed",
        "expires_after": request.expires_after.dict() if request.expires_after else None,
        "expires_at": None,
        "last_active_at": current_time,
        "metadata": request.metadata
    }

    # Calculate expires_at if expires_after is provided
    if request.expires_after:
        if request.expires_after.anchor == "created_at":
            vector_store_data["expires_at"] = current_time + request.expires_after.seconds

    # Save to metadata
    vector_stores[vector_store_id] = vector_store_data
    save_metadata(VECTOR_STORES_METADATA, vector_stores)

    # Process initial file attachments if provided
    if request.file_ids:
        for file_id in request.file_ids:
            await create_vector_store_file(vector_store_id, CreateVectorStoreFileRequest(
                file_id=file_id,
                chunking_strategy=request.chunking_strategy
            ))

    return VectorStore(**vector_store_data)


@router.get("/v1/vector_stores", response_model=VectorStoreListResponse)
async def list_vector_stores(
    before: Optional[str] = Query(None, description="Cursor for pagination"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to retrieve"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
) -> VectorStoreListResponse:
    """List vector stores with pagination."""
    vector_stores = load_metadata(VECTOR_STORES_METADATA)

    # Convert to list and sort
    stores_list = []
    for store_id, store_data in vector_stores.items():
        try:
            stores_list.append(VectorStore(**store_data))
        except ValidationError:
            continue  # Skip invalid entries

    # Sort by created_at
    reverse = order == "desc"
    stores_list.sort(key=lambda x: x.created_at, reverse=reverse)

    # Apply pagination
    start_idx = 0
    if after:
        for i, store in enumerate(stores_list):
            if store.id == after:
                start_idx = i + 1
                break
    elif before:
        for i, store in enumerate(stores_list):
            if store.id == before:
                start_idx = i
                break

    end_idx = start_idx + limit
    paginated_stores = stores_list[start_idx:end_idx]

    return VectorStoreListResponse(
        data=paginated_stores,
        first_id=paginated_stores[0].id if paginated_stores else None,
        last_id=paginated_stores[-1].id if paginated_stores else None,
        has_more=len(stores_list) > end_idx
    )


@router.get("/v1/vector_stores/{vector_store_id}", response_model=VectorStore)
async def get_vector_store(vector_store_id: str) -> VectorStore:
    """Retrieve a specific vector store."""
    vector_stores = load_metadata(VECTOR_STORES_METADATA)

    if vector_store_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Vector store not found")

    try:
        return VectorStore(**vector_stores[vector_store_id])
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Invalid vector store data: {e}")


@router.post("/v1/vector_stores/{vector_store_id}", response_model=VectorStore)
async def modify_vector_store(
    vector_store_id: str,
    request: ModifyVectorStoreRequest
) -> VectorStore:
    """Modify an existing vector store."""
    vector_stores = load_metadata(VECTOR_STORES_METADATA)

    if vector_store_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Vector store not found")

    store_data = vector_stores[vector_store_id]

    # Update fields if provided
    if request.name is not None:
        store_data["name"] = request.name
    if request.description is not None:
        store_data["description"] = request.description
    if request.metadata is not None:
        store_data["metadata"] = request.metadata
    if request.expires_after is not None:
        store_data["expires_after"] = request.expires_after.dict()
        # Recalculate expires_at
        if request.expires_after.anchor == "created_at":
            store_data["expires_at"] = store_data["created_at"] + request.expires_after.seconds

    # Update last_active_at
    store_data["last_active_at"] = int(time.time())

    # Save changes
    save_metadata(VECTOR_STORES_METADATA, vector_stores)

    try:
        return VectorStore(**store_data)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Invalid vector store data: {e}")


@router.delete("/v1/vector_stores/{vector_store_id}", response_model=Dict[str, Any])
async def delete_vector_store(vector_store_id: str) -> Dict[str, Any]:
    """Delete a vector store."""
    vector_stores = load_metadata(VECTOR_STORES_METADATA)
    vector_store_files = load_metadata(VECTOR_STORE_FILES_METADATA)
    vector_store_batches = load_metadata(VECTOR_STORE_BATCHES_METADATA)

    if vector_store_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Vector store not found")

    # Delete the Qdrant collection
    collection_name = f"vector_store_{vector_store_id}"
    try:
        qdrant_vector_client.delete_collection(collection_name)
    except Exception:
        # Collection might not exist or deletion might fail, but continue
        pass

    # Remove all associated files
    files_to_remove = []
    for file_id, file_data in vector_store_files.items():
        if file_data.get("vector_store_id") == vector_store_id:
            files_to_remove.append(file_id)

    for file_id in files_to_remove:
        del vector_store_files[file_id]

    # Remove all associated batches
    batches_to_remove = []
    for batch_id, batch_data in vector_store_batches.items():
        if batch_data.get("vector_store_id") == vector_store_id:
            batches_to_remove.append(batch_id)

    for batch_id in batches_to_remove:
        del vector_store_batches[batch_id]

    # Remove the vector store
    del vector_stores[vector_store_id]

    # Save all changes
    save_metadata(VECTOR_STORES_METADATA, vector_stores)
    save_metadata(VECTOR_STORE_FILES_METADATA, vector_store_files)
    save_metadata(VECTOR_STORE_BATCHES_METADATA, vector_store_batches)

    return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}


async def create_vector_store_file(
    vector_store_id: str,
    request: CreateVectorStoreFileRequest
) -> VectorStoreFile:
    """Create a vector store file (internal helper)."""
    vector_store_files = load_metadata(VECTOR_STORE_FILES_METADATA)

    # Verify vector store exists
    vector_stores = load_metadata(VECTOR_STORES_METADATA)
    if vector_store_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Vector store not found")

    # Verify file exists in the files system
    file_obj = file_manager.get_file(request.file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    # Check if file already exists in this vector store
    for existing_file_id, file_data in vector_store_files.items():
        if (file_data.get("vector_store_id") == vector_store_id and
            existing_file_id == request.file_id):
            raise HTTPException(
                status_code=409,
                detail="File already exists in this vector store"
            )

    # Read file content
    file_path = file_manager.get_file_path(request.file_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File content not found")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-encoded")

    # Process the file content
    collection_name = f"vector_store_{vector_store_id}"

    # Determine chunking strategy
    chunking_strategy = request.chunking_strategy or ChunkingStrategy(type="auto")

    # Chunk the content
    if chunking_strategy.type == "static":
        # For static chunking, we'd implement custom chunking logic
        # For now, use the default chunking
        chunks = chunk_text(file_content, chunk_size=chunking_strategy.max_chunk_size_tokens or 800)
    else:
        # Auto chunking
        chunks = chunk_text(file_content)

    # Create points for Qdrant
    points = []
    for chunk_idx, chunk in enumerate(chunks):
        # Get embedding
        embedding = await llamacpp_client.get_embedding(chunk)

        # Ensure vector is a flat list of floats
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                embedding = embedding[0]  # Take first inner list

        # Create point
        point_id = f"{request.file_id}_chunk_{chunk_idx}"
        metadata = {
            "file_id": request.file_id,
            "chunk_index": chunk_idx,
            "total_chunks": len(chunks),
            "text": chunk,
            "filename": file_obj.filename,
            "vector_store_id": vector_store_id
        }

        # Add attributes if provided
        if request.attributes:
            metadata.update(request.attributes)

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata
        ))

    # Store in Qdrant
    success = qdrant_vector_client.upsert_points(collection_name, points)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store file in vector store")

    # Create vector store file object
    current_time = int(time.time())
    file_data = {
        "id": request.file_id,
        "object": "vector_store.file",
        "created_at": current_time,
        "vector_store_id": vector_store_id,
        "status": "completed",
        "usage_bytes": file_obj.bytes,
        "attributes": request.attributes,
        "last_error": None
    }

    # Save to metadata
    vector_store_files[request.file_id] = file_data
    save_metadata(VECTOR_STORE_FILES_METADATA, vector_store_files)

    # Update vector store file counts
    update_vector_store_file_counts(vector_store_id)

    return VectorStoreFile(**file_data)


@router.post("/v1/vector_stores/{vector_store_id}/files", response_model=VectorStoreFile)
async def create_vector_store_file_endpoint(
    vector_store_id: str,
    request: CreateVectorStoreFileRequest
) -> VectorStoreFile:
    """Create a vector store file by attaching a file to a vector store."""
    return await create_vector_store_file(vector_store_id, request)


@router.get("/v1/vector_stores/{vector_store_id}/files", response_model=VectorStoreFileListResponse)
async def list_vector_store_files(
    vector_store_id: str,
    before: Optional[str] = Query(None, description="Cursor for pagination"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    filter: Optional[str] = Query(None, regex="^(in_progress|completed|failed|cancelled)$",
                                 description="Filter by file status"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to retrieve"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
) -> VectorStoreFileListResponse:
    """List files in a vector store."""
    vector_store_files = load_metadata(VECTOR_STORE_FILES_METADATA)

    # Filter files for this vector store
    store_files = []
    for file_id, file_data in vector_store_files.items():
        if file_data.get("vector_store_id") == vector_store_id:
            if filter and file_data.get("status") != filter:
                continue
            try:
                store_files.append(VectorStoreFile(**file_data))
            except ValidationError:
                continue

    # Sort by created_at
    reverse = order == "desc"
    store_files.sort(key=lambda x: x.created_at, reverse=reverse)

    # Apply pagination
    start_idx = 0
    if after:
        for i, file in enumerate(store_files):
            if file.id == after:
                start_idx = i + 1
                break
    elif before:
        for i, file in enumerate(store_files):
            if file.id == before:
                start_idx = i
                break

    end_idx = start_idx + limit
    paginated_files = store_files[start_idx:end_idx]

    return VectorStoreFileListResponse(
        data=paginated_files,
        first_id=paginated_files[0].id if paginated_files else None,
        last_id=paginated_files[-1].id if paginated_files else None,
        has_more=len(store_files) > end_idx
    )


@router.get("/v1/vector_stores/{vector_store_id}/files/{file_id}", response_model=VectorStoreFile)
async def get_vector_store_file(vector_store_id: str, file_id: str) -> VectorStoreFile:
    """Retrieve a specific vector store file."""
    vector_store_files = load_metadata(VECTOR_STORE_FILES_METADATA)

    if file_id not in vector_store_files:
        raise HTTPException(status_code=404, detail="Vector store file not found")

    file_data = vector_store_files[file_id]
    if file_data.get("vector_store_id") != vector_store_id:
        raise HTTPException(status_code=404, detail="Vector store file not found")

    try:
        return VectorStoreFile(**file_data)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Invalid vector store file data: {e}")


@router.delete("/v1/vector_stores/{vector_store_id}/files/{file_id}", response_model=VectorStoreFileDeleteResponse)
async def delete_vector_store_file(vector_store_id: str, file_id: str) -> VectorStoreFileDeleteResponse:
    """Delete a vector store file."""
    vector_store_files = load_metadata(VECTOR_STORE_FILES_METADATA)

    if file_id not in vector_store_files:
        raise HTTPException(status_code=404, detail="Vector store file not found")

    file_data = vector_store_files[file_id]
    if file_data.get("vector_store_id") != vector_store_id:
        raise HTTPException(status_code=404, detail="Vector store file not found")

    # Delete vectors from Qdrant collection
    collection_name = f"vector_store_{vector_store_id}"
    try:
        # Delete all points with this file_id
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="file_id",
                    match=MatchValue(value=file_id)
                )
            ]
        )
        qdrant_vector_client.delete_points(collection_name, filter_condition)
    except Exception:
        # If deletion fails, continue with metadata cleanup
        pass

    # Remove the file from metadata
    del vector_store_files[file_id]
    save_metadata(VECTOR_STORE_FILES_METADATA, vector_store_files)

    # Update vector store file counts
    update_vector_store_file_counts(vector_store_id)

    return VectorStoreFileDeleteResponse(id=file_id, deleted=True)


@router.post("/v1/vector_stores/{vector_store_id}/file_batches", response_model=VectorStoreFileBatch)
async def create_vector_store_file_batch(
    vector_store_id: str,
    request: CreateVectorStoreFileBatchRequest
) -> VectorStoreFileBatch:
    """Create a vector store file batch."""
    # Verify vector store exists
    vector_stores = load_metadata(VECTOR_STORES_METADATA)
    if vector_store_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Vector store not found")

    # Validate request (mutually exclusive file_ids and files)
    if request.file_ids and request.files:
        raise HTTPException(
            status_code=400,
            detail="file_ids and files are mutually exclusive"
        )
    if not request.file_ids and not request.files:
        raise HTTPException(
            status_code=400,
            detail="Either file_ids or files must be provided"
        )

    vector_store_batches = load_metadata(VECTOR_STORE_BATCHES_METADATA)

    # Generate batch ID
    batch_id = generate_id("vsfb")

    current_time = int(time.time())

    # Prepare file list
    files_to_process = []
    if request.file_ids:
        files_to_process = [{"file_id": fid} for fid in request.file_ids]
    else:
        files_to_process = [{"file_id": f.file_id, "attributes": f.attributes,
                           "chunking_strategy": f.chunking_strategy}
                           for f in request.files]

    # Create batch object
    batch_data = {
        "id": batch_id,
        "object": "vector_store.file_batch",
        "created_at": current_time,
        "vector_store_id": vector_store_id,
        "status": "completed",  # Simplified - would be async in real implementation
        "file_counts": {
            "in_progress": 0,
            "completed": len(files_to_process),
            "failed": 0,
            "cancelled": 0,
            "total": len(files_to_process)
        },
        "completes_at": current_time,  # Simplified
        "expires_at": None
    }

    # Save batch
    vector_store_batches[batch_id] = batch_data
    save_metadata(VECTOR_STORE_BATCHES_METADATA, vector_store_batches)

    # Process files (simplified synchronous processing)
    for file_spec in files_to_process:
        file_request = CreateVectorStoreFileRequest(
            file_id=file_spec["file_id"],
            attributes=file_spec.get("attributes"),
            chunking_strategy=file_spec.get("chunking_strategy") or request.chunking_strategy
        )
        try:
            await create_vector_store_file(vector_store_id, file_request)
        except Exception as e:
            # In real implementation, would track failed files
            pass

    return VectorStoreFileBatch(**batch_data)


@router.get("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}", response_model=VectorStoreFileBatch)
async def get_vector_store_file_batch(vector_store_id: str, batch_id: str) -> VectorStoreFileBatch:
    """Retrieve a specific vector store file batch."""
    vector_store_batches = load_metadata(VECTOR_STORE_BATCHES_METADATA)

    if batch_id not in vector_store_batches:
        raise HTTPException(status_code=404, detail="Vector store file batch not found")

    batch_data = vector_store_batches[batch_id]
    if batch_data.get("vector_store_id") != vector_store_id:
        raise HTTPException(status_code=404, detail="Vector store file batch not found")

    try:
        return VectorStoreFileBatch(**batch_data)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Invalid batch data: {e}")


@router.post("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel", response_model=VectorStoreFileBatch)
async def cancel_vector_store_file_batch(vector_store_id: str, batch_id: str) -> VectorStoreFileBatch:
    """Cancel a vector store file batch."""
    vector_store_batches = load_metadata(VECTOR_STORE_BATCHES_METADATA)

    if batch_id not in vector_store_batches:
        raise HTTPException(status_code=404, detail="Vector store file batch not found")

    batch_data = vector_store_batches[batch_id]
    if batch_data.get("vector_store_id") != vector_store_id:
        raise HTTPException(status_code=404, detail="Vector store file batch not found")

    # Update status to cancelled
    batch_data["status"] = "cancelled"

    # Save changes
    save_metadata(VECTOR_STORE_BATCHES_METADATA, vector_store_batches)

    try:
        return VectorStoreFileBatch(**batch_data)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Invalid batch data: {e}")


@router.get("/v1/vector_stores/{vector_store_id}/file_batches", response_model=VectorStoreFileBatchListResponse)
async def list_vector_store_file_batches(
    vector_store_id: str,
    before: Optional[str] = Query(None, description="Cursor for pagination"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to retrieve"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
) -> VectorStoreFileBatchListResponse:
    """List file batches for a vector store."""
    vector_store_batches = load_metadata(VECTOR_STORE_BATCHES_METADATA)

    # Filter batches for this vector store
    store_batches = []
    for batch_id, batch_data in vector_store_batches.items():
        if batch_data.get("vector_store_id") == vector_store_id:
            try:
                store_batches.append(VectorStoreFileBatch(**batch_data))
            except ValidationError:
                continue

    # Sort by created_at
    reverse = order == "desc"
    store_batches.sort(key=lambda x: x.created_at, reverse=reverse)

    # Apply pagination
    start_idx = 0
    if after:
        for i, batch in enumerate(store_batches):
            if batch.id == after:
                start_idx = i + 1
                break
    elif before:
        for i, batch in enumerate(store_batches):
            if batch.id == before:
                start_idx = i
                break

    end_idx = start_idx + limit
    paginated_batches = store_batches[start_idx:end_idx]

    return VectorStoreFileBatchListResponse(
        data=paginated_batches,
        first_id=paginated_batches[0].id if paginated_batches else None,
        last_id=paginated_batches[-1].id if paginated_batches else None,
        has_more=len(store_batches) > end_idx
    )


@router.get("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files", response_model=VectorStoreFileListResponse)
async def list_vector_store_file_batch_files(
    vector_store_id: str,
    batch_id: str,
    before: Optional[str] = Query(None, description="Cursor for pagination"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to retrieve"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
) -> VectorStoreFileListResponse:
    """List files in a specific vector store file batch."""
    # For simplicity, this returns all files in the vector store
    # In a real implementation, you'd track which files belong to which batch
    return await list_vector_store_files(
        vector_store_id=vector_store_id,
        before=before,
        after=after,
        limit=limit,
        order=order
    )