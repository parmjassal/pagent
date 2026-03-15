import pytest
import json
from pathlib import Path
from agent_platform.runtime.core.todo import TODOManager, ScopedTask, TaskStatus

def test_todo_persistence(tmp_path):
    mgr = TODOManager(tmp_path)
    
    # Add Task
    task = ScopedTask(title="Test Task", description="Something to do")
    task_id = mgr.add_task(task)
    
    # Check File exists
    task_file = tmp_path / "todo" / f"task_{task_id}.json"
    assert task_file.exists()
    
    # List pending
    pending = mgr.list_tasks(status=TaskStatus.PENDING)
    assert len(pending) == 1
    assert pending[0].title == "Test Task"
    
    # Update Status
    mgr.update_status(task_id, TaskStatus.COMPLETED)
    
    # Verify update
    completed = mgr.list_tasks(status=TaskStatus.COMPLETED)
    assert len(completed) == 1
    assert completed[0].status == TaskStatus.COMPLETED
