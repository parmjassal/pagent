from typing import List, Dict, Any, Optional
from ..core.todo import TODOManager, ScopedTask, TaskStatus
from pathlib import Path
from ..orch.state import AgentState

class TODOTool:
    """
    Tool for agents to interact with the session's TODO list.
    Supports both AGENT spawning and TOOL execution tasks.
    """
    def __init__(self, session_path: Path):
        self.mgr = TODOManager(session_path)

    def add_task(self, title: str, description: str, type: str = "agent", assigned_to: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, state: Optional[AgentState] = None) -> str:
        """
        Adds a new task to the TODO list.
        type: 'agent' or 'tool'
        payload: {name, args} for tool type.
        """
        task = ScopedTask(
            title=title, 
            description=description, 
            type=TaskType(type),
            assigned_to=assigned_to,
            payload=payload or {}
        )
        return self.mgr.add_task(task)

    def list_tasks(self, status: Optional[str] = None, state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
        """Lists tasks, optionally filtered by status."""
        target_status = TaskStatus(status) if status else None
        tasks = self.mgr.list_tasks(status=target_status)
        return [t.model_dump() for t in tasks]

    def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None, state: Optional[AgentState] = None):
        """Updates the status and result of an existing task."""
        self.mgr.update_status(task_id, TaskStatus(status), result=result)
        return f"Task {task_id} updated to {status}"
