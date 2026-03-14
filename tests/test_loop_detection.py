import pytest
from agent_platform.runtime.logic import LoopMonitor
from agent_platform.runtime.state import create_initial_state
from pathlib import Path

def test_node_loop_detection():
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    node = "decompose"
    
    # Visit 1
    state["node_counts"] = {node: 1}
    assert LoopMonitor.check_node_loop(state, node, threshold=3) is False
    
    # Visit 3 (Threshold hit)
    state["node_counts"] = {node: 3}
    assert LoopMonitor.check_node_loop(state, node, threshold=3) is True

def test_content_loop_detection():
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    
    # 3 identical messages
    state["messages"] = [
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "Hello"}
    ]
    
    assert LoopMonitor.check_content_loop(state, window=3) is True
    
    # Different messages
    state["messages"] = [
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "World"},
        {"role": "assistant", "content": "Hello"}
    ]
    assert LoopMonitor.check_content_loop(state, window=3) is False
