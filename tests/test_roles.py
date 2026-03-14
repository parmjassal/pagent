import pytest
from agent_platform.runtime.state import create_initial_state, AgentRole
from agent_platform.runtime.supervisor import SupervisorAgent
from pathlib import Path

def test_supervisor_role_threshold(tmp_path):
    # We only need the supervisor instance to test its private routing method
    supervisor = SupervisorAgent(None, None, None, api_key="dummy")
    
    # 1. TEST SUPERVISOR: Threshold 3
    state_sup = create_initial_state("s1", "u1", "sess1", Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    state_sup["node_counts"] = {"decompose": 2} # Under threshold
    assert supervisor._should_continue(state_sup) != "abort"
    
    state_sup["node_counts"] = {"decompose": 3} # Hit threshold
    assert supervisor._should_continue(state_sup) == "abort"

def test_worker_role_threshold(tmp_path):
    supervisor = SupervisorAgent(None, None, None, api_key="dummy")
    
    # 2. TEST WORKER: Threshold 10
    state_work = create_initial_state("w1", "u1", "sess1", Path("/tmp"), Path("/tmp"), role=AgentRole.WORKER)
    state_work["node_counts"] = {"decompose": 3} # Would trigger for supervisor, but not worker
    assert supervisor._should_continue(state_work) != "abort"
    
    state_work["node_counts"] = {"decompose": 10} # Hit worker threshold
    assert supervisor._should_continue(state_work) == "abort"
