"""
Runs API routes following OpenAI specifications
"""
import os
import json
import time
import uuid
import asyncio
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List, Dict, Any, Union
import logging

from ..models import (
    Run, CreateRunRequest, ModifyRunRequest, RunListResponse,
    RunStep, RunStepListResponse, CreateThreadAndRunRequest,
    Thread, ThreadMessage, Assistant, AssistantTool
)
from ..config import RUNS_DIR, RUNS_METADATA_FILE, THREADS_DIR, THREADS_METADATA_FILE, ASSISTANTS_DIR, ASSISTANTS_METADATA_FILE
from ..clients.llamacpp_client import llamacpp_client
from ..clients.qdrant_client import qdrant_vector_client
from ..rag.search import rag_search
from ..utils.text_processing import chunk_text

logger = logging.getLogger(__name__)

router = APIRouter()


class RunManager:
    """Simple run manager using JSON metadata store"""

    def __init__(self, runs_dir: str, metadata_file: str):
        self.runs_dir = runs_dir
        self.metadata_file = metadata_file
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load runs metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save runs metadata: {e}")

    def _load_thread_metadata(self):
        """Load thread metadata"""
        if os.path.exists(THREADS_METADATA_FILE):
            try:
                with open(THREADS_METADATA_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load threads metadata: {e}")
                return {}
        return {}

    def _save_thread_metadata(self, metadata):
        """Save thread metadata"""
        try:
            with open(THREADS_METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save threads metadata: {e}")

    def _load_assistant_metadata(self):
        """Load assistant metadata"""
        if os.path.exists(ASSISTANTS_METADATA_FILE):
            try:
                with open(ASSISTANTS_METADATA_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load assistants metadata: {e}")
                return {}
        return {}

    def create_run(self, thread_id: str, request: CreateRunRequest) -> Run:
        """Create a new run"""
        run_id = f"run_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())
        expires_at = created_at + 3600  # 1 hour expiry

        # Load assistant to get defaults
        assistant_metadata = self._load_assistant_metadata()
        assistant_data = assistant_metadata.get(request.assistant_id)
        if not assistant_data:
            raise HTTPException(status_code=404, detail="Assistant not found")

        assistant = Assistant(**assistant_data)

        # Merge assistant defaults with request overrides
        model = request.model or assistant.model
        instructions = request.instructions or assistant.instructions
        tools = request.tools or assistant.tools
        file_ids = request.file_ids or assistant.file_ids

        run = Run(
            id=run_id,
            created_at=created_at,
            thread_id=thread_id,
            assistant_id=request.assistant_id,
            status="queued",
            expires_at=expires_at,
            model=model,
            instructions=instructions,
            tools=tools,
            file_ids=file_ids,
            metadata=request.metadata,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_prompt_tokens=request.max_prompt_tokens,
            max_completion_tokens=request.max_completion_tokens,
            truncation_strategy=request.truncation_strategy,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            response_format=request.response_format
        )

        self.metadata[run_id] = {
            "run": run.dict(),
            "steps": {}
        }
        self._save_metadata()

        logger.info(f"Created run: {run_id} for thread: {thread_id}")
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get run by ID"""
        if run_id not in self.metadata:
            return None
        return Run(**self.metadata[run_id]["run"])

    def update_run(self, run_id: str, updates: Dict[str, Any]) -> Run:
        """Update a run"""
        if run_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Run not found")

        run_data = self.metadata[run_id]["run"]
        run_data.update(updates)

        run = Run(**run_data)
        self.metadata[run_id]["run"] = run.dict()
        self._save_metadata()

        return run

    def delete_run(self, run_id: str) -> bool:
        """Delete a run"""
        if run_id not in self.metadata:
            return False

        del self.metadata[run_id]
        self._save_metadata()

        logger.info(f"Deleted run: {run_id}")
        return True

    def list_runs(self, thread_id: str, after: Optional[str] = None, limit: int = 100,
                  order: str = "desc") -> RunListResponse:
        """List runs for a thread"""
        runs = []
        for run_data in self.metadata.values():
            run = Run(**run_data["run"])
            if run.thread_id == thread_id:
                runs.append(run)

        # Sort by created_at
        reverse = order == "desc"
        runs.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, r in enumerate(runs):
                if r.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_runs = runs[start_idx:end_idx]

        has_more = end_idx < len(runs)
        first_id = paginated_runs[0].id if paginated_runs else None
        last_id = paginated_runs[-1].id if paginated_runs else None

        return RunListResponse(
            data=paginated_runs,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )

    def create_run_step(self, run_id: str, step_type: str, step_details: Dict[str, Any]) -> RunStep:
        """Create a run step"""
        step_id = f"step_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        run = self.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        step = RunStep(
            id=step_id,
            created_at=created_at,
            run_id=run_id,
            thread_id=run.thread_id,
            type=step_type,
            status="in_progress",
            step_details=step_details
        )

        self.metadata[run_id]["steps"][step_id] = step.dict()
        self._save_metadata()

        return step

    def update_run_step(self, step_id: str, run_id: str, updates: Dict[str, Any]) -> RunStep:
        """Update a run step"""
        if run_id not in self.metadata or step_id not in self.metadata[run_id]["steps"]:
            raise HTTPException(status_code=404, detail="Run step not found")

        step_data = self.metadata[run_id]["steps"][step_id]
        step_data.update(updates)

        step = RunStep(**step_data)
        self.metadata[run_id]["steps"][step_id] = step.dict()
        self._save_metadata()

        return step

    def list_run_steps(self, run_id: str, after: Optional[str] = None, limit: int = 100,
                      order: str = "desc") -> RunStepListResponse:
        """List steps for a run"""
        if run_id not in self.metadata:
            raise HTTPException(status_code=404, detail="Run not found")

        steps = []
        for step_data in self.metadata[run_id]["steps"].values():
            step = RunStep(**step_data)
            steps.append(step)

        # Sort by created_at
        reverse = order == "desc"
        steps.sort(key=lambda x: x.created_at, reverse=reverse)

        # Pagination
        start_idx = 0
        if after:
            for i, s in enumerate(steps):
                if s.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_steps = steps[start_idx:end_idx]

        has_more = end_idx < len(steps)
        first_id = paginated_steps[0].id if paginated_steps else None
        last_id = paginated_steps[-1].id if paginated_steps else None

        return RunStepListResponse(
            data=paginated_steps,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )


# Global instance
run_manager = RunManager(RUNS_DIR, RUNS_METADATA_FILE)


class ToolExecutor:
    """Executes tools for runs"""

    def __init__(self):
        self.max_execution_time = 30  # seconds

    async def execute_file_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Execute file search tool"""
        try:
            # Search using existing RAG functionality
            search_results = await rag_search(query, top_k=max_results)

            # Format results
            results = []
            for i, content in enumerate(search_results):
                results.append({
                    "file_id": f"doc_{i}",
                    "file_name": f"document_{i}.txt",
                    "content": content,
                    "score": 1.0 - (i * 0.1)  # Decreasing score
                })

            return {
                "type": "file_search_results",
                "results": results
            }
        except Exception as e:
            logger.error(f"File search error: {e}")
            return {
                "type": "error",
                "error": str(e)
            }

    async def execute_code_interpreter(self, code: str) -> Dict[str, Any]:
        """Execute code interpreter tool (basic sandboxed execution)"""
        try:
            # WARNING: This is NOT truly sandboxed - use Docker or proper sandbox in production
            # Restricted globals for basic safety
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'range': range,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                    'enumerate': enumerate,
                    'zip': zip,
                    # Math functions
                    'math': __import__('math'),
                    # Add more safe functions as needed
                }
            }

            # Capture stdout
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Execute with timeout
            exec_globals = restricted_globals.copy()
            exec_globals.update({
                '__name__': '__main__',
                '__doc__': None,
                '__annotations__': {},
            })

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Use asyncio to enforce timeout
                def execute_code():
                    exec(code, exec_globals)

                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, execute_code),
                        timeout=self.max_execution_time
                    )
                except asyncio.TimeoutError:
                    return {
                        "type": "error",
                        "error": f"Code execution timed out after {self.max_execution_time} seconds"
                    }

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            return {
                "type": "code_execution_results",
                "stdout": stdout,
                "stderr": stderr
            }

        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                "type": "error",
                "error": str(e)
            }


async def execute_run(run_id: str):
    """Execute a run with tool calling"""
    tool_executor = ToolExecutor()

    try:
        run = run_manager.get_run(run_id)
        if not run:
            logger.error(f"Run {run_id} not found")
            return

        # Update status to in_progress
        run_manager.update_run(run_id, {"status": "in_progress", "started_at": int(time.time())})

        # Load thread messages
        thread_metadata = run_manager._load_thread_metadata()
        if run.thread_id not in thread_metadata:
            run_manager.update_run(run_id, {
                "status": "failed",
                "last_error": {"message": "Thread not found"}
            })
            return

        thread_messages = []
        for msg_data in thread_metadata[run.thread_id]["messages"].values():
            msg = ThreadMessage(**msg_data)
            thread_messages.append({
                "role": msg.role,
                "content": msg.content[0]["text"]["value"] if msg.content else ""
            })

        # Build conversation for LLM
        messages = []

        # Add system instructions
        if run.instructions:
            messages.append({"role": "system", "content": run.instructions})

        # Add thread messages
        messages.extend(thread_messages)

        # Tool calling loop
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM
            payload = {
                "model": run.model,
                "messages": messages,
                "temperature": run.temperature,
                "top_p": run.top_p,
                "max_tokens": run.max_completion_tokens,
                "tools": run.tools if run.tools else None,
                "tool_choice": run.tool_choice or "auto"
            }

            try:
                response = await llamacpp_client.generate_completion(payload)
                response_content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])

                # Create message creation step
                step_details = {
                    "type": "message_creation",
                    "message_creation": {
                        "message_id": f"msg_{uuid.uuid4().hex[:24]}"
                    }
                }
                run_manager.create_run_step(run_id, "message_creation", step_details)

                # Add assistant response to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": response_content
                }
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                # Create thread message
                thread_metadata = run_manager._load_thread_metadata()
                message_id = f"msg_{uuid.uuid4().hex[:24]}"
                message = ThreadMessage(
                    id=message_id,
                    created_at=int(time.time()),
                    thread_id=run.thread_id,
                    role="assistant",
                    content=[{"type": "text", "text": {"value": response_content}}],
                    assistant_id=run.assistant_id,
                    run_id=run_id
                )
                thread_metadata[run.thread_id]["messages"][message_id] = message.dict()
                run_manager._save_thread_metadata(thread_metadata)

                # Execute tool calls if any
                if tool_calls:
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name")
                        tool_args = tool_call.get("function", {}).get("arguments", "{}")

                        try:
                            args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        except:
                            args = {}

                        # Execute tool
                        if tool_name == "file_search":
                            result = await tool_executor.execute_file_search(args.get("query", ""))
                        elif tool_name == "code_interpreter":
                            result = await tool_executor.execute_code_interpreter(args.get("code", ""))
                        else:
                            result = {"type": "error", "error": f"Unknown tool: {tool_name}"}

                        tool_results.append({
                            "tool_call_id": tool_call.get("id"),
                            "content": json.dumps(result)
                        })

                        # Create tool call step
                        step_details = {
                            "type": "tool_calls",
                            "tool_calls": [{
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type"),
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_args
                                }
                            }]
                        }
                        run_manager.create_run_step(run_id, "tool_calls", step_details)

                    # Add tool results to conversation
                    for result in tool_results:
                        messages.append({
                            "role": "tool",
                            "content": result["content"],
                            "tool_call_id": result["tool_call_id"]
                        })

                        # Create thread message for tool result
                        message_id = f"msg_{uuid.uuid4().hex[:24]}"
                        message = ThreadMessage(
                            id=message_id,
                            created_at=int(time.time()),
                            thread_id=run.thread_id,
                            role="assistant",  # Tool results are from assistant context
                            content=[{"type": "text", "text": {"value": result["content"]}}],
                            assistant_id=run.assistant_id,
                            run_id=run_id
                        )
                        thread_metadata[run.thread_id]["messages"][message_id] = message.dict()
                        run_manager._save_thread_metadata(thread_metadata)

                    # Continue the loop for next LLM call with tool results
                    continue
                else:
                    # No tool calls, run is complete
                    break

            except Exception as e:
                logger.error(f"LLM call error: {e}")
                run_manager.update_run(run_id, {
                    "status": "failed",
                    "last_error": {"message": str(e)}
                })
                return

        # Mark run as completed
        run_manager.update_run(run_id, {
            "status": "completed",
            "completed_at": int(time.time())
        })

    except Exception as e:
        logger.error(f"Run execution error: {e}")
        run_manager.update_run(run_id, {
            "status": "failed",
            "last_error": {"message": str(e)}
        })


@router.post("/v1/threads/{thread_id}/runs", response_model=Run)
async def create_run(thread_id: str, request: CreateRunRequest, background_tasks: BackgroundTasks):
    """Create and start a run for a thread"""
    logger.info(f"Creating run for thread: {thread_id}, assistant: {request.assistant_id}")

    try:
        run = run_manager.create_run(thread_id, request)

        # Start execution in background
        background_tasks.add_task(execute_run, run.id)

        return run
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create run: {e}")
        raise HTTPException(status_code=500, detail="Failed to create run")


@router.get("/v1/threads/{thread_id}/runs/{run_id}", response_model=Run)
async def get_run(thread_id: str, run_id: str):
    """Retrieve a run"""
    logger.info(f"Retrieving run: {run_id}")

    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    return run


@router.post("/v1/threads/{thread_id}/runs/{run_id}", response_model=Run)
async def update_run(thread_id: str, run_id: str, request: ModifyRunRequest):
    """Modify a run"""
    logger.info(f"Updating run: {run_id}")

    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        updates = request.dict(exclude_unset=True)
        run = run_manager.update_run(run_id, updates)
        return run
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update run: {e}")
        raise HTTPException(status_code=500, detail="Failed to update run")


@router.get("/v1/threads/{thread_id}/runs", response_model=RunListResponse)
async def list_runs(
    thread_id: str,
    after: Optional[str] = Query(None, description="A cursor for use in pagination"),
    limit: int = Query(100, ge=1, le=100, description="A limit on the number of objects to be returned"),
    order: str = Query("desc", description="Sort order by created_at timestamp")
):
    """Returns a list of runs for a given thread"""
    logger.info(f"Listing runs for thread: {thread_id}")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")

    return run_manager.list_runs(thread_id, after=after, limit=limit, order=order)


@router.get("/v1/threads/{thread_id}/runs/{run_id}/steps", response_model=RunStepListResponse)
async def list_run_steps(
    thread_id: str,
    run_id: str,
    after: Optional[str] = Query(None, description="A cursor for use in pagination"),
    limit: int = Query(100, ge=1, le=100, description="A limit on the number of objects to be returned"),
    order: str = Query("desc", description="Sort order by created_at timestamp")
):
    """Returns a list of run steps for a given run"""
    logger.info(f"Listing steps for run: {run_id}")

    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")

    return run_manager.list_run_steps(run_id, after=after, limit=limit, order=order)


@router.post("/v1/threads/runs", response_model=Run)
async def create_thread_and_run(request: CreateThreadAndRunRequest, background_tasks: BackgroundTasks):
    """Create a thread and run it in one request"""
    logger.info(f"Creating thread and run for assistant: {request.assistant_id}")

    # Create thread
    from .threads import thread_manager
    thread = thread_manager.create_thread(request.thread or CreateThreadAndRunRequest())

    # Create run request
    run_request = CreateRunRequest(
        assistant_id=request.assistant_id,
        model=request.model,
        instructions=request.instructions,
        tools=request.tools,
        metadata=request.metadata,
        temperature=request.temperature,
        top_p=request.top_p,
        max_prompt_tokens=request.max_prompt_tokens,
        max_completion_tokens=request.max_completion_tokens,
        truncation_strategy=request.truncation_strategy,
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=request.response_format
    )

    # Create run
    run = run_manager.create_run(thread.id, run_request)

    # Start execution in background
    background_tasks.add_task(execute_run, run.id)

    return run