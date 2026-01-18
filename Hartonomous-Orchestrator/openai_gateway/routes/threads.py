"""
Threads API routes following OpenAI specifications
"""
import os
import json
import time
import uuid
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..models import (
    Thread, CreateThreadRequest, ModifyThreadRequest,
    ThreadMessage, CreateMessageRequest, ModifyMessageRequest, MessageListResponse
)
from ..config import THREADS_DIR, THREADS_METADATA_FILE

logger = logging.getLogger(__name__)

router = APIRouter()


class ThreadManager:
    """Simple thread manager using JSON metadata store"""

    def __init__(self, threads_dir: str, metadata_file: str):
        self.threads_dir = threads_dir
        self.metadata_file = metadata_file
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load threads metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save threads metadata: {e}")

    def create_thread(self, request: CreateThreadRequest) -> Thread:
        """Create a new thread"""
        thread_id = f"thread_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        thread = Thread(
            id=thread_id,
            created_at=created_at,
            metadata=request.metadata
        )

        self.metadata[thread_id] = {
            "thread": thread.dict(),
            "messages": {}
        }

        # Create initial messages if provided
        if request.messages:
            for msg_data in request.messages:
                self.create_message(thread_id, CreateMessageRequest(**msg_data))

        self._save_metadata()

        logger.info(f"Created thread: {thread_id}")
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        if thread_id not in self.metadata:
            return None
        return Thread(**self.metadata[thread_id]["thread"])

    def update_thread(self, thread_id: str, request: ModifyThreadRequest) -> Thread:
        """Update an existing thread"""
        if thread_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Thread not found")

        thread_data = self.metadata[thread_id]["thread"]

        # Update only provided fields
        updates = request.dict(exclude_unset=True)
        thread_data.update(updates)

        thread = Thread(**thread_data)
        self.metadata[thread_id]["thread"] = thread.dict()
        self._save_metadata()

        logger.info(f"Updated thread: {thread_id}")
        return thread

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        if thread_id not in self.metadata:
            return False

        del self.metadata[thread_id]
        self._save_metadata()

        logger.info(f"Deleted thread: {thread_id}")
        return True

    def list_threads(self, after: Optional[str] = None, limit: int = 100,
                    order: str = "desc") -> List[Thread]:
        """List threads with pagination"""
        threads = []
        for thread_data in self.metadata.values():
            thread = Thread(**thread_data["thread"])
            threads.append(thread)

        # Sort by created_at
        reverse = order == "desc"
        threads.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, t in enumerate(threads):
                if t.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        return threads[start_idx:end_idx]

    def create_message(self, thread_id: str, request: CreateMessageRequest) -> ThreadMessage:
        """Create a new message in a thread"""
        if thread_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Thread not found")

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        # Convert content to content blocks format
        content_blocks = [{"type": "text", "text": {"value": request.content}}]

        message = ThreadMessage(
            id=message_id,
            created_at=created_at,
            thread_id=thread_id,
            role=request.role,
            content=content_blocks,
            file_ids=request.file_ids or [],
            metadata=request.metadata
        )

        self.metadata[thread_id]["messages"][message_id] = message.dict()
        self._save_metadata()

        logger.info(f"Created message {message_id} in thread {thread_id}")
        return message

    def get_message(self, thread_id: str, message_id: str) -> Optional[ThreadMessage]:
        """Get message by ID"""
        if thread_id not in self.metadata or message_id not in self.metadata[thread_id]["messages"]:
            return None
        return ThreadMessage(**self.metadata[thread_id]["messages"][message_id])

    def update_message(self, thread_id: str, message_id: str, request: ModifyMessageRequest) -> ThreadMessage:
        """Update an existing message"""
        if thread_id not in self.metadata or message_id not in self.metadata[thread_id]["messages"]:
            raise HTTPException(status_code=404, detail="Message not found")

        message_data = self.metadata[thread_id]["messages"][message_id]

        # Update only provided fields
        updates = request.dict(exclude_unset=True)
        message_data.update(updates)

        message = ThreadMessage(**message_data)
        self.metadata[thread_id]["messages"][message_id] = message.dict()
        self._save_metadata()

        logger.info(f"Updated message {message_id} in thread {thread_id}")
        return message

    def list_messages(self, thread_id: str, after: Optional[str] = None, limit: int = 100,
                     order: str = "desc") -> MessageListResponse:
        """List messages in a thread with pagination"""
        if thread_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages = []
        for message_data in self.metadata[thread_id]["messages"].values():
            message = ThreadMessage(**message_data)
            messages.append(message)

        # Sort by created_at
        reverse = order == "desc"
        messages.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, m in enumerate(messages):
                if m.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_messages = messages[start_idx:end_idx]

        has_more = end_idx < len(messages)
        first_id = paginated_messages[0].id if paginated_messages else None
        last_id = paginated_messages[-1].id if paginated_messages else None

        return MessageListResponse(
            data=paginated_messages,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )


# Initialize thread manager
thread_manager = ThreadManager(THREADS_DIR, THREADS_METADATA_FILE)


@router.post("/v1/threads", response_model=Thread)
async def create_thread(request: CreateThreadRequest):
    """Create a new thread"""
    logger.info(f"Creating thread")

    try:
        thread = thread_manager.create_thread(request)
        return thread
    except Exception as e:
        logger.error(f"Failed to create thread: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thread")


@router.get("/v1/threads/{thread_id}", response_model=Thread)
async def get_thread(thread_id: str):
    """Retrieve a thread"""
    logger.info(f"Retrieving thread: {thread_id}")

    thread = thread_manager.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    return thread


@router.post("/v1/threads/{thread_id}", response_model=Thread)
async def update_thread(thread_id: str, request: ModifyThreadRequest):
    """Modify a thread"""
    logger.info(f"Updating thread: {thread_id}")

    try:
        thread = thread_manager.update_thread(thread_id, request)
        return thread
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update thread: {e}")
        raise HTTPException(status_code=500, detail="Failed to update thread")


@router.delete("/v1/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread"""
    logger.info(f"Deleting thread: {thread_id}")

    deleted = thread_manager.delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Return empty response with 204 status
    return {"object": "thread.deleted", "id": thread_id, "deleted": True}


@router.post("/v1/threads/{thread_id}/messages", response_model=ThreadMessage)
async def create_message(thread_id: str, request: CreateMessageRequest):
    """Create a message"""
    logger.info(f"Creating message in thread: {thread_id}")

    try:
        message = thread_manager.create_message(thread_id, request)
        return message
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create message: {e}")
        raise HTTPException(status_code=500, detail="Failed to create message")


@router.get("/v1/threads/{thread_id}/messages", response_model=MessageListResponse)
async def list_messages(
    thread_id: str,
    after: Optional[str] = Query(None, description="A cursor for use in pagination"),
    limit: int = Query(100, ge=1, le=100, description="A limit on the number of objects to be returned"),
    order: str = Query("desc", description="Sort order by created_at timestamp")
):
    """Returns a list of messages for a given thread"""
    logger.info(f"Listing messages for thread: {thread_id}")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")

    return thread_manager.list_messages(thread_id, after=after, limit=limit, order=order)


@router.get("/v1/threads/{thread_id}/messages/{message_id}", response_model=ThreadMessage)
async def get_message(thread_id: str, message_id: str):
    """Retrieve a message"""
    logger.info(f"Retrieving message: thread={thread_id}, message={message_id}")

    message = thread_manager.get_message(thread_id, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    return message


@router.post("/v1/threads/{thread_id}/messages/{message_id}", response_model=ThreadMessage)
async def update_message(thread_id: str, message_id: str, request: ModifyMessageRequest):
    """Modify a message"""
    logger.info(f"Updating message: thread={thread_id}, message={message_id}")

    try:
        message = thread_manager.update_message(thread_id, message_id, request)
        return message
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update message: {e}")
        raise HTTPException(status_code=500, detail="Failed to update message")