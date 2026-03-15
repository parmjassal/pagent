from typing import List, Dict, Any, Optional
from ..core.todo import TODOManager, ScopedTask, TaskStatus
from pathlib import Path

class TODOTool:
    """
    Tool for agents to interact with the session's TODO list.
    """
    def __init__(self, session_path: Path):
        self.mgr = TODOManager(session_path)

    def add_task(self, title: str, description: str, assigned_to: Optional[str] = None) -> str:
        """Adds a new task to the TODO list."""
        task = ScopedTask(title=title, description=description, assigned_to=assigned_to)
        return self.mgr.add_task(task)

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists tasks, optionally filtered by status."""
        target_status = TaskStatus(status) if status else None
        tasks = self.mgr.list_tasks(status=target_status)
        return [t.model_dump() for t in tasks]

    def update_task_status(self, task_id: str, status: str):
        """Updates the status of an existing task."""
        self.mgr.update_status(task_id, TaskStatus(status))
        return f"Task {task_id} updated to {status}"
