"""
Files API routes following OpenAI specifications
"""
import os
import json
import time
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from typing import Optional, List
import logging

from ..models import FileObject, FileListResponse, FileDeleteResponse, ExpiresAfter
from ..config import FILES_DIR, FILES_METADATA_FILE

logger = logging.getLogger(__name__)

router = APIRouter()


class FileManager:
    """Simple file manager using JSON metadata store"""

    def __init__(self, files_dir: str, metadata_file: str):
        self.files_dir = files_dir
        self.metadata_file = metadata_file
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def create_file(self, file: UploadFile, purpose: str, expires_after: Optional[ExpiresAfter] = None) -> FileObject:
        """Create and save a new file"""
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        # Calculate expiration
        expires_at = None
        if expires_after:
            if expires_after.anchor == "created_at":
                expires_at = created_at + expires_after.seconds

        # Save file
        file_path = os.path.join(self.files_dir, file_id)
        try:
            with open(file_path, 'wb') as f:
                content = file.file.read()
                f.write(content)
            file_size = len(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # Create metadata
        file_obj = FileObject(
            id=file_id,
            bytes=file_size,
            created_at=created_at,
            expires_at=expires_at,
            filename=file.filename,
            purpose=purpose
        )

        self.metadata[file_id] = file_obj.dict()
        self._save_metadata()

        return file_obj

    def get_file(self, file_id: str) -> Optional[FileObject]:
        """Get file metadata by ID"""
        if file_id not in self.metadata:
            return None
        return FileObject(**self.metadata[file_id])

    def list_files(self, purpose: Optional[str] = None, after: Optional[str] = None,
                   limit: int = 10000, order: str = "desc") -> FileListResponse:
        """List files with optional filtering and pagination"""
        files = []
        for file_data in self.metadata.values():
            file_obj = FileObject(**file_data)

            # Filter by purpose
            if purpose and file_obj.purpose != purpose:
                continue

            # Check expiration
            if file_obj.expires_at and file_obj.expires_at < time.time():
                continue

            files.append(file_obj)

        # Sort by created_at
        reverse = order == "desc"
        files.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, f in enumerate(files):
                if f.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_files = files[start_idx:end_idx]

        has_more = end_idx < len(files)
        first_id = paginated_files[0].id if paginated_files else None
        last_id = paginated_files[-1].id if paginated_files else None

        return FileListResponse(
            data=paginated_files,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )

    def delete_file(self, file_id: str) -> bool:
        """Delete file and its metadata"""
        if file_id not in self.metadata:
            return False

        # Delete file from disk
        file_path = os.path.join(self.files_dir, file_id)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")

        # Delete metadata
        del self.metadata[file_id]
        self._save_metadata()
        return True

    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get the filesystem path for a file"""
        if file_id not in self.metadata:
            return None
        return os.path.join(self.files_dir, file_id)


# Initialize file manager
file_manager = FileManager(FILES_DIR, FILES_METADATA_FILE)


@router.post("/v1/files", response_model=FileObject)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(..., description="The intended purpose of the uploaded file"),
    expires_after_anchor: Optional[str] = Form(None, alias="expires_after[anchor]"),
    expires_after_seconds: Optional[int] = Form(None, alias="expires_after[seconds]")
):
    """Upload a file that can be used across various endpoints"""
    logger.info(f"File upload request: filename={file.filename}, purpose={purpose}")

    # Validate purpose
    valid_purposes = ["assistants", "batch", "fine-tune", "vision", "user_data", "evals"]
    if purpose not in valid_purposes:
        raise HTTPException(status_code=400, detail=f"Invalid purpose. Must be one of: {', '.join(valid_purposes)}")

    # Validate file size (512 MB limit)
    max_size = 512 * 1024 * 1024  # 512 MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds 512 MB limit")

    # Reset file pointer
    await file.seek(0)

    # Handle expires_after
    expires_after = None
    if expires_after_anchor and expires_after_seconds is not None:
        expires_after = ExpiresAfter(anchor=expires_after_anchor, seconds=expires_after_seconds)

    try:
        file_obj = file_manager.create_file(file, purpose, expires_after)
        logger.info(f"File uploaded successfully: id={file_obj.id}, size={file_obj.bytes} bytes")
        return file_obj
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@router.get("/v1/files", response_model=FileListResponse)
async def list_files(
    purpose: Optional[str] = Query(None, description="Only return files with the given purpose"),
    after: Optional[str] = Query(None, description="A cursor for use in pagination"),
    limit: int = Query(10000, ge=1, le=10000, description="A limit on the number of objects to be returned"),
    order: str = Query("desc", description="Sort order by created_at timestamp")
):
    """Returns a list of files"""
    logger.info(f"File list request: purpose={purpose}, limit={limit}, order={order}")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")

    return file_manager.list_files(purpose=purpose, after=after, limit=limit, order=order)


@router.get("/v1/files/{file_id}", response_model=FileObject)
async def get_file(file_id: str):
    """Returns information about a specific file"""
    logger.info(f"File info request: file_id={file_id}")

    file_obj = file_manager.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    # Check expiration
    if file_obj.expires_at and file_obj.expires_at < time.time():
        raise HTTPException(status_code=404, detail="File has expired")

    return file_obj


@router.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    """Delete a file"""
    logger.info(f"File delete request: file_id={file_id}")

    deleted = file_manager.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return FileDeleteResponse(id=file_id, deleted=True)


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Returns the contents of the specified file"""
    logger.info(f"File content request: file_id={file_id}")

    file_obj = file_manager.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    # Check expiration
    if file_obj.expires_at and file_obj.expires_at < time.time():
        raise HTTPException(status_code=404, detail="File has expired")

    file_path = file_manager.get_file_path(file_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File content not found")

    # Return file as binary response
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_obj.filename
    )