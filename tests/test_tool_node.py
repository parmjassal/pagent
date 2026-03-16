import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from agent_platform.runtime.orch.state import create_initial_state
from agent_platform.runtime.orch.tool_node import AgentToolNode
from agent_platform.runtime.core.dispatcher import ToolDispatcher

@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock(spec=ToolDispatcher)
    dispatcher.dispatch = AsyncMock()
    return dispatcher

@pytest.mark.asyncio
async def test_tool_node_success(mock_dispatcher):
    node = AgentToolNode(mock_dispatcher)
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    
    # Setup tool request in metadata
    state["metadata"]["next_tool_call"] = {"name": "test_tool", "args": {"x": 1}}
    
    # Mock successful dispatch
    mock_dispatcher.dispatch.return_value = {"success": True, "output": "result_val", "source": "native"}
    
    result = await node(state)
    
    # Verifications
    assert result["metadata"]["next_tool_call"] is None
    # If no tool_call_id, it falls back to 'user' role with [Tool Result] prefix
    assert result["messages"][0]["role"] == "user"
    assert "result_val" in result["messages"][0]["content"]
    assert "successfully" in result["messages"][1]["content"]

@pytest.mark.asyncio
async def test_tool_node_failure(mock_dispatcher):
    node = AgentToolNode(mock_dispatcher)
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    state["metadata"]["next_tool_call"] = {"name": "fail_tool", "args": {}}
    
    mock_dispatcher.dispatch.return_value = {"success": False, "error": "Internal Error", "error_code": "TEST_ERROR"}
    
    result = await node(state)
    
    assert "Error executing tool 'fail_tool': Internal Error (Code: TEST_ERROR)" in result["messages"][0]["content"]
    assert "failed (Code: TEST_ERROR)" in result["messages"][1]["content"]

@pytest.mark.asyncio
async def test_tool_node_with_id_compliant(mock_dispatcher):
    """If tool_call_id is present and history is compliant, use 'tool' role."""
    node = AgentToolNode(mock_dispatcher)
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    
    # Pre-populate history with compliant assistant message
    state["messages"] = [
        {"role": "assistant", "content": "Thinking...", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}]}
    ]
    state["metadata"]["next_tool_call"] = {"name": "test_tool", "args": {}, "id": "call_123"}
    
    mock_dispatcher.dispatch.return_value = {"success": True, "output": "ok", "source": "native"}
    
    result = await node(state)
    
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_123"
    assert result["messages"][0]["content"] == "ok"

@pytest.mark.asyncio
async def test_tool_node_with_id_non_compliant(mock_dispatcher):
    """If tool_call_id is present but history is non-compliant, fallback to 'user' role."""
    node = AgentToolNode(mock_dispatcher)
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    
    # Pre-populate history with NON-compliant assistant message (missing tool_calls)
    state["messages"] = [
        {"role": "assistant", "content": "Thinking..."}
    ]
    state["metadata"]["next_tool_call"] = {"name": "test_tool", "args": {}, "id": "call_123"}
    
    mock_dispatcher.dispatch.return_value = {"success": True, "output": "ok", "source": "native"}
    
    result = await node(state)
    
    # Fallback to user role
    assert result["messages"][0]["role"] == "user"
    assert "[Tool Result: test_tool]" in result["messages"][0]["content"]
    assert "ok" in result["messages"][0]["content"]
