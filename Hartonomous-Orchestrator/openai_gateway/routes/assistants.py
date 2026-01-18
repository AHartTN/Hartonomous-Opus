"""
Assistants API routes following OpenAI specifications
"""
import os
import json
import time
import uuid
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..models import (
    Assistant, CreateAssistantRequest, ModifyAssistantRequest,
    AssistantListResponse, AssistantFile, AssistantFileListResponse
)
from ..config import ASSISTANTS_DIR, ASSISTANTS_METADATA_FILE

logger = logging.getLogger(__name__)

router = APIRouter()


class AssistantManager:
    """Simple assistant manager using JSON metadata store"""

    def __init__(self, assistants_dir: str, metadata_file: str):
        self.assistants_dir = assistants_dir
        self.metadata_file = metadata_file
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load assistants metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save assistants metadata: {e}")

    def create_assistant(self, request: CreateAssistantRequest) -> Assistant:
        """Create a new assistant"""
        assistant_id = f"asst_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        assistant = Assistant(
            id=assistant_id,
            created_at=created_at,
            name=request.name,
            description=request.description,
            model=request.model,
            instructions=request.instructions,
            tools=request.tools or [],
            file_ids=request.file_ids or [],
            metadata=request.metadata
        )

        self.metadata[assistant_id] = assistant.dict()
        self._save_metadata()

        logger.info(f"Created assistant: {assistant_id}")
        return assistant

    def get_assistant(self, assistant_id: str) -> Optional[Assistant]:
        """Get assistant by ID"""
        if assistant_id not in self.metadata:
            return None
        return Assistant(**self.metadata[assistant_id])

    def update_assistant(self, assistant_id: str, request: ModifyAssistantRequest) -> Assistant:
        """Update an existing assistant"""
        if assistant_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Assistant not found")

        current_data = self.metadata[assistant_id]

        # Update only provided fields
        updates = request.dict(exclude_unset=True)
        current_data.update(updates)

        assistant = Assistant(**current_data)
        self.metadata[assistant_id] = assistant.dict()
        self._save_metadata()

        logger.info(f"Updated assistant: {assistant_id}")
        return assistant

    def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant"""
        if assistant_id not in self.metadata:
            return False

        del self.metadata[assistant_id]
        self._save_metadata()

        logger.info(f"Deleted assistant: {assistant_id}")
        return True

    def list_assistants(self, after: Optional[str] = None, limit: int = 100,
                       order: str = "desc") -> AssistantListResponse:
        """List assistants with pagination"""
        assistants = []
        for assistant_data in self.metadata.values():
            assistant = Assistant(**assistant_data)
            assistants.append(assistant)

        # Sort by created_at
        reverse = order == "desc"
        assistants.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, a in enumerate(assistants):
                if a.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_assistants = assistants[start_idx:end_idx]

        has_more = end_idx < len(assistants)
        first_id = paginated_assistants[0].id if paginated_assistants else None
        last_id = paginated_assistants[-1].id if paginated_assistants else None

        return AssistantListResponse(
            data=paginated_assistants,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )

    def create_assistant_file(self, assistant_id: str, file_id: str) -> AssistantFile:
        """Attach a file to an assistant"""
        if assistant_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Assistant not found")

        assistant = Assistant(**self.metadata[assistant_id])

        # Check if file is already attached
        if file_id in assistant.file_ids:
            raise HTTPException(status_code=400, detail="File already attached to assistant")

        # Add file to assistant
        assistant.file_ids.append(file_id)
        self.metadata[assistant_id] = assistant.dict()
        self._save_metadata()

        assistant_file = AssistantFile(
            id=file_id,
            created_at=int(time.time()),
            assistant_id=assistant_id
        )

        logger.info(f"Attached file {file_id} to assistant {assistant_id}")
        return assistant_file

    def get_assistant_file(self, assistant_id: str, file_id: str) -> Optional[AssistantFile]:
        """Get an assistant file"""
        assistant = self.get_assistant(assistant_id)
        if not assistant or file_id not in assistant.file_ids:
            return None

        return AssistantFile(
            id=file_id,
            created_at=assistant.created_at,  # Approximate creation time
            assistant_id=assistant_id
        )

    def delete_assistant_file(self, assistant_id: str, file_id: str) -> bool:
        """Remove a file from an assistant"""
        if assistant_id not in self.metadata:
            return False

        assistant = Assistant(**self.metadata[assistant_id])

        if file_id not in assistant.file_ids:
            return False

        assistant.file_ids.remove(file_id)
        self.metadata[assistant_id] = assistant.dict()
        self._save_metadata()

        logger.info(f"Detached file {file_id} from assistant {assistant_id}")
        return True

    def list_assistant_files(self, assistant_id: str, after: Optional[str] = None,
                           limit: int = 100, order: str = "desc") -> AssistantFileListResponse:
        """List files attached to an assistant"""
        assistant = self.get_assistant(assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        assistant_files = []
        for file_id in assistant.file_ids:
            assistant_file = AssistantFile(
                id=file_id,
                created_at=assistant.created_at,  # Approximate
                assistant_id=assistant_id
            )
            assistant_files.append(assistant_file)

        # Sort by created_at (approximate)
        reverse = order == "desc"
        assistant_files.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, af in enumerate(assistant_files):
                if af.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_files = assistant_files[start_idx:end_idx]

        has_more = end_idx < len(assistant_files)
        first_id = paginated_files[0].id if paginated_files else None
        last_id = paginated_files[-1].id if paginated_files else None

        return AssistantFileListResponse(
            data=paginated_files,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )


# Initialize assistant manager
assistant_manager = AssistantManager(ASSISTANTS_DIR, ASSISTANTS_METADATA_FILE)


@router.post("/v1/assistants", response_model=Assistant)
async def create_assistant(request: CreateAssistantRequest):
    """Create a new assistant"""
    logger.info(f"Creating assistant: model={request.model}, name={request.name}")

    # Validate model - for now, accept any string
    if not request.model:
        raise HTTPException(status_code=400, detail="Model is required")

    # Validate tools
    if request.tools:
        for tool in request.tools:
            if tool.type not in ["code_interpreter", "file_search"]:
                raise HTTPException(status_code=400, detail=f"Unsupported tool type: {tool.type}")

    try:
        assistant = assistant_manager.create_assistant(request)
        return assistant
    except Exception as e:
        logger.error(f"Failed to create assistant: {e}")
        raise HTTPException(status_code=500, detail="Failed to create assistant")


@router.get("/v1/assistants", response_model=AssistantListResponse)
async def list_assistants(
    after: Optional[str] = Query(None, description="A cursor for use in pagination"),
    limit: int = Query(100, ge=1, le=100, description="A limit on the number of objects to be returned"),
    order: str = Query("desc", description="Sort order by created_at timestamp")
):
    """Returns a list of assistants"""
    logger.info(f"Listing assistants: limit={limit}, order={order}")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")

    return assistant_manager.list_assistants(after=after, limit=limit, order=order)


@router.get("/v1/assistants/{assistant_id}", response_model=Assistant)
async def get_assistant(assistant_id: str):
    """Retrieve an assistant"""
    logger.info(f"Retrieving assistant: {assistant_id}")

    assistant = assistant_manager.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    return assistant


@router.post("/v1/assistants/{assistant_id}", response_model=Assistant)
async def update_assistant(assistant_id: str, request: ModifyAssistantRequest):
    """Modify an assistant"""
    logger.info(f"Updating assistant: {assistant_id}")

    # Validate tools if provided
    if request.tools:
        for tool in request.tools:
            if tool.type not in ["code_interpreter", "file_search"]:
                raise HTTPException(status_code=400, detail=f"Unsupported tool type: {tool.type}")

    try:
        assistant = assistant_manager.update_assistant(assistant_id, request)
        return assistant
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update assistant: {e}")
        raise HTTPException(status_code=500, detail="Failed to update assistant")


@router.delete("/v1/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """Delete an assistant"""
    logger.info(f"Deleting assistant: {assistant_id}")

    deleted = assistant_manager.delete_assistant(assistant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Return empty response with 204 status
    return {"object": "assistant.deleted", "id": assistant_id, "deleted": True}


@router.post("/v1/assistants/{assistant_id}/files", response_model=AssistantFile)
async def create_assistant_file(assistant_id: str, file_id: str = Query(..., description="The ID of the file to attach")):
    """Attach a file to an assistant"""
    logger.info(f"Attaching file {file_id} to assistant {assistant_id}")

    try:
        assistant_file = assistant_manager.create_assistant_file(assistant_id, file_id)
        return assistant_file
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to attach file: {e}")
        raise HTTPException(status_code=500, detail="Failed to attach file")


@router.get("/v1/assistants/{assistant_id}/files", response_model=AssistantFileListResponse)
async def list_assistant_files(
    assistant_id: str,
    after: Optional[str] = Query(None, description="A cursor for use in pagination"),
    limit: int = Query(100, ge=1, le=100, description="A limit on the number of objects to be returned"),
    order: str = Query("desc", description="Sort order by created_at timestamp")
):
    """Returns a list of files attached to an assistant"""
    logger.info(f"Listing files for assistant: {assistant_id}")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")

    return assistant_manager.list_assistant_files(assistant_id, after=after, limit=limit, order=order)


@router.get("/v1/assistants/{assistant_id}/files/{file_id}", response_model=AssistantFile)
async def get_assistant_file(assistant_id: str, file_id: str):
    """Retrieve an assistant file"""
    logger.info(f"Retrieving assistant file: assistant={assistant_id}, file={file_id}")

    assistant_file = assistant_manager.get_assistant_file(assistant_id, file_id)
    if not assistant_file:
        raise HTTPException(status_code=404, detail="Assistant file not found")

    return assistant_file


@router.delete("/v1/assistants/{assistant_id}/files/{file_id}")
async def delete_assistant_file(assistant_id: str, file_id: str):
    """Remove a file from an assistant"""
    logger.info(f"Detaching file {file_id} from assistant {assistant_id}")

    deleted = assistant_manager.delete_assistant_file(assistant_id, file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Assistant file not found")

    return {"object": "assistant.file.deleted", "id": file_id, "deleted": True}