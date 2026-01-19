"""
Fine-tuning API routes following OpenAI specifications
"""
import os
import json
import time
import uuid
import asyncio
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..models import (
    FineTuningJob, FineTuningJobRequest, FineTuningJobList,
    FineTuningJobEvent, FineTuningJobEventsList, FineTuningHyperparameters
)
from ..config import DATA_DIR
from .files import file_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
JOBS_DIR = os.path.join(DATA_DIR, "fine_tuning", "jobs")
EVENTS_DIR = os.path.join(DATA_DIR, "fine_tuning", "events")
JOBS_METADATA_FILE = os.path.join(DATA_DIR, "fine_tuning", "jobs_metadata.json")


class FineTuningJobManager:
    """Manager for fine-tuning jobs with JSON metadata store"""

    def __init__(self, jobs_dir: str, events_dir: str, metadata_file: str):
        self.jobs_dir = jobs_dir
        self.events_dir = events_dir
        self.metadata_file = metadata_file
        self._ensure_dirs()
        self._load_metadata()

    def _ensure_dirs(self):
        """Ensure required directories exist"""
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load jobs metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs metadata: {e}")

    def _generate_job_id(self) -> str:
        """Generate a unique job ID"""
        return f"ftjob-{uuid.uuid4().hex[:16]}"

    def _generate_event_id(self) -> str:
        """Generate a unique event ID"""
        return f"ftevent-{uuid.uuid4().hex[:16]}"

    def create_job(self, request: FineTuningJobRequest) -> FineTuningJob:
        """Create a new fine-tuning job"""
        job_id = self._generate_job_id()
        created_at = int(time.time())

        # Set default hyperparameters if not provided
        hyperparameters = request.hyperparameters or FineTuningHyperparameters()

        # Create job object
        job = FineTuningJob(
            id=job_id,
            model=request.model,
            created_at=created_at,
            status="pending",
            training_file=request.training_file,
            validation_file=request.validation_file,
            hyperparameters=hyperparameters,
        )

        # Store job
        self.metadata[job_id] = job.dict()
        self._save_metadata()

        # Create initial event
        self._add_event(job_id, "info", f"Job created with model {request.model}")

        # Start background job processing
        asyncio.create_task(self._process_job(job_id))

        return job

    def get_job(self, job_id: str) -> Optional[FineTuningJob]:
        """Get job by ID"""
        if job_id not in self.metadata:
            return None
        return FineTuningJob(**self.metadata[job_id])

    def list_jobs(self, after: Optional[str] = None, limit: int = 20) -> FineTuningJobList:
        """List jobs with optional pagination"""
        jobs = []
        for job_data in self.metadata.values():
            job = FineTuningJob(**job_data)
            jobs.append(job)

        # Sort by created_at desc
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        # Pagination
        start_idx = 0
        if after:
            for i, job in enumerate(jobs):
                if job.id == after:
                    start_idx = i + 1
                    break

        end_idx = start_idx + limit
        paginated_jobs = jobs[start_idx:end_idx]

        has_more = end_idx < len(jobs)
        first_id = paginated_jobs[0].id if paginated_jobs else None
        last_id = paginated_jobs[-1].id if paginated_jobs else None

        return FineTuningJobList(
            data=paginated_jobs,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's pending or running"""
        if job_id not in self.metadata:
            return False

        job_data = self.metadata[job_id]
        job = FineTuningJob(**job_data)

        if job.status not in ["pending", "running"]:
            return False

        # Update status
        job.status = "cancelled"
        job.finished_at = int(time.time())
        self.metadata[job_id] = job.dict()
        self._save_metadata()

        # Add event
        self._add_event(job_id, "info", "Job cancelled by user")

        return True

    def _add_event(self, job_id: str, level: str, message: str):
        """Add an event for a job"""
        event_id = self._generate_event_id()
        event = FineTuningJobEvent(
            id=event_id,
            created_at=int(time.time()),
            level=level,
            message=message
        )

        # Store event (in practice, you'd store in a DB or append to a file)
        # For simplicity, we'll just log it
        logger.info(f"Job {job_id} event: {level} - {message}")

    def get_job_events(self, job_id: str) -> FineTuningJobEventsList:
        """Get events for a job (placeholder - would need proper event storage)"""
        # For now, return empty list as we don't have persistent event storage
        return FineTuningJobEventsList(data=[])

    async def _process_job(self, job_id: str):
        """Process a fine-tuning job in the background"""
        try:
            # Update status to running
            job_data = self.metadata[job_id]
            job = FineTuningJob(**job_data)
            job.status = "running"
            self.metadata[job_id] = job.dict()
            self._save_metadata()
            self._add_event(job_id, "info", "Job started processing")

            # Simulate processing time
            await asyncio.sleep(5)  # 5 seconds for demo

            # Check if still running (not cancelled)
            job_data = self.metadata[job_id]
            job = FineTuningJob(**job_data)
            if job.status != "cancelled":
                # Simulate success (in real implementation, would do actual training)
                job.status = "succeeded"
                job.finished_at = int(time.time())
                job.fine_tuned_model = f"{job.model}:ft:{job_id}"
                job.trained_tokens = 1000  # Mock value
                self.metadata[job_id] = job.dict()
                self._save_metadata()
                self._add_event(job_id, "info", "Job completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            # Update status to failed
            job_data = self.metadata[job_id]
            job = FineTuningJob(**job_data)
            job.status = "failed"
            job.finished_at = int(time.time())
            job.error = {"message": str(e)}
            self.metadata[job_id] = job.dict()
            self._save_metadata()
            self._add_event(job_id, "error", f"Job failed: {str(e)}")


# Initialize job manager
job_manager = FineTuningJobManager(JOBS_DIR, EVENTS_DIR, JOBS_METADATA_FILE)


@router.post("/v1/fine_tuning/jobs", response_model=FineTuningJob)
async def create_fine_tuning_job(request: FineTuningJobRequest):
    """Create a fine-tuning job"""
    logger.info(f"Creating fine-tuning job for model {request.model}")

    # Validate training file exists
    training_file = file_manager.get_file(request.training_file)
    if not training_file:
        raise HTTPException(status_code=400, detail="Training file not found")

    if training_file.purpose != "fine-tune":
        raise HTTPException(status_code=400, detail="Training file must have purpose 'fine-tune'")

    # Validate validation file if provided
    if request.validation_file:
        validation_file = file_manager.get_file(request.validation_file)
        if not validation_file:
            raise HTTPException(status_code=400, detail="Validation file not found")
        if validation_file.purpose != "fine-tune":
            raise HTTPException(status_code=400, detail="Validation file must have purpose 'fine-tune'")

    # Validate suffix
    if request.suffix and not request.suffix.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Suffix must contain only alphanumeric characters, hyphens, and underscores")

    try:
        job = job_manager.create_job(request)
        logger.info(f"Fine-tuning job created: {job.id}")
        return job
    except Exception as e:
        logger.error(f"Failed to create fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create fine-tuning job")


@router.get("/v1/fine_tuning/jobs")
async def list_fine_tuning_jobs():
    """List fine-tuning jobs"""
    logger.info("Listing fine-tuning jobs")
    return job_manager.list_jobs()


@router.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}", response_model=FineTuningJob)
async def get_fine_tuning_job(fine_tuning_job_id: str):
    """Retrieve a fine-tuning job"""
    logger.info(f"Retrieving fine-tuning job: {fine_tuning_job_id}")

    job = job_manager.get_job(fine_tuning_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    return job


@router.post("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel", response_model=FineTuningJob)
async def cancel_fine_tuning_job(fine_tuning_job_id: str):
    """Cancel a fine-tuning job"""
    logger.info(f"Cancelling fine-tuning job: {fine_tuning_job_id}")

    job = job_manager.get_job(fine_tuning_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    if job.status not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail=f"Job status is {job.status}, cannot cancel")

    cancelled = job_manager.cancel_job(fine_tuning_job_id)
    if not cancelled:
        raise HTTPException(status_code=500, detail="Failed to cancel job")

    # Return updated job
    job = job_manager.get_job(fine_tuning_job_id)
    return job


@router.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}/events", response_model=FineTuningJobEventsList)
async def list_fine_tuning_job_events(
    fine_tuning_job_id: str,
    after: Optional[str] = Query(None, description="Identifier for the last event from the previous page"),
    limit: int = Query(20, ge=1, le=100, description="Number of events to retrieve")
):
    """List events for a fine-tuning job"""
    logger.info(f"Listing events for fine-tuning job: {fine_tuning_job_id}")

    job = job_manager.get_job(fine_tuning_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    return job_manager.get_job_events(fine_tuning_job_id)