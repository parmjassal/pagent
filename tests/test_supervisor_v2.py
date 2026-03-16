import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
from langchain_core.messages import SystemMessage

from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.core.mailbox import Mailbox
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.models import ExecutionStrategy

@pytest.fixture
def mock_env():
    factory = MagicMock(spec=AgentFactory)
    mailbox = MagicMock(spec=Mailbox)
    generator = MagicMock(spec=SystemGeneratorAgent)
    llm = MagicMock()
    
    # Mock workspace in factory
    factory.workspace = MagicMock()
    factory.workspace.get_agent_dir.return_value = Path("/tmp/agent_dir")
    
    return {
        "factory": factory,
        "mailbox": mailbox,
        "generator": generator,
        "llm": llm
    }

@pytest.mark.asyncio
async def test_spawning_node_consumes_next_steps_and_uses_hierarchical_id(mock_env):
    supervisor = SupervisorAgent(
        mock_env["factory"], 
        mock_env["mailbox"], 
        mock_env["generator"], 
        llm=mock_env["llm"]
    )
    
    # Setup state
    state = create_initial_state(
        agent_id="super", 
        user_id="u1", 
        session_id="s1", 
        inbox_path=Path("/tmp/in"), 
        outbox_path=Path("/tmp/out")
    )
    state["next_steps"] = ["worker1", "worker2"]
    state["generated_output"] = "MOCK PROMPT"
    state["metadata"]["sub_tasks_config"] = [
        {"agent_id": "worker1", "role": AgentRole.WORKER, "instructions": "Task 1"},
        {"agent_id": "worker2", "role": AgentRole.WORKER, "instructions": "Task 2"}
    ]
    
    # Mock factory to return a state
    mock_env["factory"].create_agent.return_value = {"dummy": "state"}
    
    # 1. Execute spawning_node for FIRST agent
    result = await supervisor.spawning_node(state)
    
    # VERIFICATIONS for first spawn
    assert result["next_steps"] == ["worker2"]  # Should have consumed 'worker1'
    assert "Spawned super/worker1" in result["messages"][0]["content"]
    
    # Check Mailbox call - should use hierarchical ID
    mock_env["mailbox"].send.assert_called_once()
    args, _ = mock_env["mailbox"].send.call_args
    assert args[0] == "super/worker1"  # Hierarchical ID check
    assert args[1]["system_prompt"] == "MOCK PROMPT"

    # 2. Execute spawning_node for SECOND agent
    # Update state with result from first spawn
    state["next_steps"] = result["next_steps"]
    mock_env["mailbox"].send.reset_mock()
    
    result2 = await supervisor.spawning_node(state)
    
    # VERIFICATIONS for second spawn
    assert result2["next_steps"] == []  # Should have consumed 'worker2'
    assert "Spawned super/worker2" in result2["messages"][0]["content"]
    
    # Check Mailbox call
    mock_env["mailbox"].send.assert_called_once()
    args2, _ = mock_env["mailbox"].send.call_args
    assert args2[0] == "super/worker2"

@pytest.mark.asyncio
async def test_spawning_node_recursive_id(mock_env):
    # Test that it doesn't double-prefix if already prefixed
    mock_compiler = MagicMock()
    mock_compiler.compile_unit.return_value = AsyncMock()
    mock_compiler.compile_unit.return_value.ainvoke = AsyncMock(return_value={"final_result": "Done"})
    
    supervisor = SupervisorAgent(
        mock_env["factory"], 
        mock_env["mailbox"], 
        mock_env["generator"], 
        llm=mock_env["llm"],
        unit_compiler=mock_compiler
    )
    
    state = create_initial_state("parent/super", "u1", "s1", Path("/tmp/in"), Path("/tmp/out"))
    state["next_steps"] = ["child"]
    state["metadata"]["sub_tasks_config"] = [{"agent_id": "child", "role": AgentRole.WORKER, "instructions": "..."}]
    
    mock_env["factory"].create_agent.return_value = {"dummy": "state"}
    
    await supervisor.spawning_node(state)
    
    # Check factory call
    _, kwargs = mock_env["factory"].create_agent.call_args
    assert kwargs["agent_id"] == "parent/super/child"
