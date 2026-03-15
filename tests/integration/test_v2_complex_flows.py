import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.quota import SessionQuota, update_quota
from agent_platform.runtime.orch.state import create_initial_state, update_next_steps, AgentRole
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.validator import SystemValidatorAgent
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry
from agent_platform.runtime.core.guardrails import GuardrailManager
from agent_platform.runtime.core.sandbox import ProcessSandboxRunner
from agent_platform.runtime.orch.models import ValidationResult, PlanningResult, ExecutionStrategy, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v2_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (global_dir := workspace.get_global_dir()).mkdir(parents=True)
    (global_dir / "guidelines.md").write_text("# Safety\n- No destructive actions.")
    (global_prompts := global_dir / "prompts").mkdir()
    (global_prompts / "agent_base.txt").write_text("BASE")

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "v2_user", "sess_v2"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    mock_sup_llm = AsyncMock()
    mock_gen_llm = AsyncMock()
    mock_val_llm = AsyncMock()

    generator = SystemGeneratorAgent(llm=mock_gen_llm, workspace=workspace)
    validator = SystemValidatorAgent(llm=mock_val_llm, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, llm=mock_sup_llm)

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "supervisor": supervisor, "validator": validator, "mailbox": mailbox,
        "mock_val_llm": mock_val_llm, "mock_sup_llm": mock_sup_llm
    }

@pytest.mark.asyncio
async def test_v2_recursive_depth_and_handover(v2_env):
    env = v2_env
    supervisor = env["supervisor"]
    
    env["mock_sup_llm"].ainvoke.return_value = PlanningResult(
        thought_process="Spawning L1",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="agent_l1", role=AgentRole.WORKER, instructions="Task")]
    )

    state = create_initial_state("super", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    
    # 1. Spawn L1
    res1 = await supervisor.planning_node(state)
    state.update(res1)
    res_spawn1 = await supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res_spawn1["quota"])
    
    # 2. Spawn L2 from L1
    env["mock_sup_llm"].ainvoke.return_value = PlanningResult(
        thought_process="Spawning L2",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="agent_l2", role=AgentRole.WORKER, instructions="Task")]
    )
    state["agent_id"] = "super/agent_l1" # Correct hierarchical ID for current agent
    state["current_depth"] = 1
    
    res2 = await supervisor.planning_node(state)
    state.update(res2)
    res_spawn2 = await supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res_spawn2["quota"])
    
    assert state["quota"].agent_count == 2
    # The spawned ID is parent/child -> super/agent_l1/agent_l2
    msg = env["mailbox"].receive("super/agent_l1/agent_l2")
    assert msg["sender"] == "super/agent_l1"

@pytest.mark.asyncio
async def test_v2_validation_positive_negative(v2_env):
    env = v2_env
    validator = env["validator"]
    state = create_initial_state("a1", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))

    state["generated_output"] = "safe code"
    env["mock_val_llm"].ainvoke.return_value = ValidationResult(is_valid=True, reasoning="Safe")
    val_res = await validator.validate_node(state)
    assert val_res["is_valid"] is True

    state["generated_output"] = "destructive code"
    env["mock_val_llm"].ainvoke.return_value = ValidationResult(is_valid=False, reasoning="Violation")
    val_res_fail = await validator.validate_node(state)
    assert val_res_fail["is_valid"] is False

@pytest.mark.asyncio
async def test_v2_session_quota_enforcement(v2_env):
    env = v2_env
    supervisor = env["supervisor"]
    
    state = create_initial_state("super", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    state["quota"].max_agents = 2
    
    # 1
    state["next_steps"] = ["s1"]
    res1 = await supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res1["quota"])
    
    # 2
    state["next_steps"] = ["s2"]
    res2 = await supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res2["quota"])
    
    # 3 (Fail)
    state["next_steps"] = ["s3"]
    res3 = await supervisor.spawning_node(state)
    assert "Failed to spawn super/s3" in res3["messages"][0]["content"]
