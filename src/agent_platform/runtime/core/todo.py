import json
import logging
import uuid
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(str, Enum):
    TOOL = "tool"
    AGENT = "agent"

class ScopedTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: TaskType = Field(default=TaskType.AGENT)
    title: str
    description: str
    assigned_to: Optional[str] = None # For AGENT type
    payload: Dict[str, Any] = Field(default_factory=dict) # For TOOL type: {name, args}
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None # Persistent result from execution
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TODOManager:
    """
    Manages a persistent TODO list for a session to keep agents scoped.
    Each task is stored as an individual JSON file for atomic updates.
    """

    def __init__(self, session_path: Path):
        self.root = session_path / "todo"
        self.root.mkdir(parents=True, exist_ok=True)

    def add_task(self, task: ScopedTask) -> str:
        path = self.root / f"task_{task.task_id}.json"
        path.write_text(task.model_dump_json(indent=2))
        logger.info(f"Task added to TODO: {task.task_id} - {task.title}")
        return task.task_id

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[ScopedTask]:
        tasks = []
        paths = sorted(
            self.root.glob("task_*.json"),
            key=lambda p: p.stat().st_ctime  # creation time (oldest first)
        )

        for path in paths:
            try:
                task = ScopedTask.model_validate_json(path.read_text())
                if status is None or task.status == status:
                    tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to load task {path}: {e}")

        return tasks

    def update_status(self, task_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None):
        """Updates the status and optionally the result of an existing task."""
        path = self.root / f"task_{task_id}.json"
        if not path.exists():
            # Try to find by finding the file with this ID
            found = list(self.root.glob(f"task_{task_id}.json"))
            if not found:
                raise FileNotFoundError(f"Task {task_id} not found.")
            path = found[0]
        
        task = ScopedTask.model_validate_json(path.read_text())
        task.status = status
        if result is not None:
            task.result = result
        path.write_text(task.model_dump_json(indent=2))
        logger.info(f"Task {task_id} updated to {status}")
