import pytest
from pathlib import Path
from agent_platform.runtime.guardrails import GuardrailManager, guardrail_tool_wrapper
from agent_platform.runtime.state import create_initial_state

@pytest.fixture
def manager():
    return GuardrailManager()

@pytest.fixture
def dummy_state():
    return create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))

def test_guardrail_blocking_destructive_tool(manager, dummy_state):
    # Simulated policy blocks tools with 'delete'
    is_allowed, reason = manager.validate_tool_call(dummy_state, "delete_file", {"path": "test.txt"})
    
    assert is_allowed is False
    assert "prohibited" in reason

def test_guardrail_caching_behavior(manager, dummy_state):
    # Call 1: Valid tool
    is_allowed, reason = manager.validate_tool_call(dummy_state, "read_file", {"path": "test.txt"})
    assert is_allowed is True
    
    # Verify it's in the cache
    assert len(manager._policy_cache) == 1
    
    # Call 2: Same context, same tool -> should hit cache
    # (Mocking/Spying could be used to verify hit, but check the count)
    is_allowed2, _ = manager.validate_tool_call(dummy_state, "read_file", {"path": "test.txt"})
    assert is_allowed2 is True
    assert len(manager._policy_cache) == 1 # Cache count stays same

def test_guardrail_context_aware_cache(manager, dummy_state):
    # Call 1: User 1
    manager.validate_tool_call(dummy_state, "read_file", {"path": "test.txt"})
    
    # Call 2: Same tool, Different User -> should be a new cache entry
    dummy_state["user_id"] = "user2"
    manager.validate_tool_call(dummy_state, "read_file", {"path": "test.txt"})
    
    assert len(manager._policy_cache) == 2

def test_tool_wrapper_blocking(manager, dummy_state):
    # Use the decorator on a dummy tool
    @guardrail_tool_wrapper(manager)
    def delete_file(state, path: str):
        return {"status": "deleted"}

    result = delete_file(dummy_state, path="important.txt")
    assert result["status"] == "blocked"
    assert "error" in result
