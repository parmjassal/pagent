import pytest
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from pathlib import Path

def test_supervisor_role_threshold(tmp_path):
    # We only need the supervisor instance to test its private routing method
    supervisor = SupervisorAgent(None, None, None, llm=None)
    
    # 1. TEST SUPERVISOR: Threshold 3
    state_sup = create_initial_state("s1", "u1", "sess1", Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    state_sup["node_counts"] = {"plan": 2} # Under threshold (plan is the node in supervisor)
    assert supervisor._should_continue(state_sup) != "abort"
    
    state_sup["node_counts"] = {"plan": 101} # Hit threshold (100)
    assert supervisor._should_continue(state_sup) == "abort"

def test_worker_role_threshold(tmp_path):
    supervisor = SupervisorAgent(None, None, None, llm=None)
    
    # 2. TEST WORKER: Threshold is also 100 in current supervisor.py logic
    state_work = create_initial_state("w1", "u1", "sess1", Path("/tmp"), Path("/tmp"), role=AgentRole.WORKER)
    state_work["node_counts"] = {"plan": 50} 
    assert supervisor._should_continue(state_work) != "abort"
    
    state_work["node_counts"] = {"plan": 101} 
    assert supervisor._should_continue(state_work) == "abort"
