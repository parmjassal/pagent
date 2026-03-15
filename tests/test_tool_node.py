import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent_platform.runtime.orch.state import create_initial_state
from agent_platform.runtime.orch.tool_node import AgentToolNode
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry

@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock(spec=ToolDispatcher)
    return dispatcher

def test_tool_node_success(mock_dispatcher):
    node = AgentToolNode(mock_dispatcher)
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    
    # Setup tool request in metadata
    state["metadata"]["next_tool_call"] = {"name": "test_tool", "args": {"x": 1}}
    
    # Mock successful dispatch
    mock_dispatcher.dispatch.return_value = {"success": True, "output": "result_val", "source": "native"}
    
    result = node(state)
    
    # Verifications
    assert result["metadata"]["next_tool_call"] is None
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["content"] == "result_val"
    assert "successfully" in result["messages"][1]["content"]

def test_tool_node_failure(mock_dispatcher):
    node = AgentToolNode(mock_dispatcher)
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    state["metadata"]["next_tool_call"] = {"name": "fail_tool", "args": {}}
    
    mock_dispatcher.dispatch.return_value = {"success": False, "error": "Internal Error"}
    
    result = node(state)
    
    assert result["messages"][0]["content"] == "Error executing tool 'fail_tool': Internal Error"
    assert "failed" in result["messages"][1]["content"]
