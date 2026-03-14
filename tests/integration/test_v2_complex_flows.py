import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.quota import SessionQuota, update_quota
from agent_platform.runtime.state import create_initial_state, update_next_steps
from agent_platform.runtime.generator import SystemGeneratorAgent, TaskType
from agent_platform.runtime.validator import SystemValidatorAgent
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.dispatcher import ToolDispatcher, ToolRegistry, ToolSource
from agent_platform.runtime.guardrails import GuardrailManager
from agent_platform.runtime.sandbox import ProcessSandboxRunner
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

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

    factory = AgentFactory(workspace, max_spawn_depth=5)
    provider = FilesystemMailboxProvider(session_path)
    mailbox = Mailbox(provider)
    sandbox = ProcessSandboxRunner()
    guardrails = GuardrailManager()
    registry = ToolRegistry(session_path)
    dispatcher = ToolDispatcher(registry, sandbox, guardrails)
    
    generator = SystemGeneratorAgent(llm=None, workspace=workspace)
    validator = SystemValidatorAgent(llm=None, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, api_key="dummy")

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "supervisor": supervisor, "validator": validator, "mailbox": mailbox
    }

def test_v2_recursive_depth_and_handover(v2_env):
    env = v2_env
    supervisor = env["supervisor"]
    
    state = create_initial_state("super", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    state["next_steps"] = ["agent_l1"]
    
    # 1. Spawn L1
    res1 = supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res1["quota"])
    state["messages"] += res1["messages"]
    
    # 2. Spawn L2 from L1
    state["agent_id"] = "agent_l1"
    state["current_depth"] = 1
    state["next_steps"] = ["agent_l2"]
    
    res2 = supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res2["quota"])
    
    assert state["quota"].agent_count == 2
    msg = env["mailbox"].receive("agent_l2")
    assert msg["sender"] == "agent_l1"

def test_v2_validation_positive_negative(v2_env):
    env = v2_env
    validator = env["validator"]
    state = create_initial_state("a1", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))

    state["generated_output"] = "def safe(): return 1"
    assert validator.validate_node(state)["is_valid"] is True

    state["generated_output"] = "def unsafe(): import os; os.remove('x')"
    assert validator.validate_node(state)["is_valid"] is False

def test_v2_session_quota_enforcement(v2_env):
    env = v2_env
    supervisor = env["supervisor"]
    
    state = create_initial_state("super", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    state["quota"].max_agents = 2
    
    # Spawn 1
    state["next_steps"] = ["s1"]
    res1 = supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res1.get("quota", SessionQuota(agent_count=0)))
    
    # Spawn 2
    state["next_steps"] = ["s2"]
    res2 = supervisor.spawning_node(state)
    state["quota"] = update_quota(state["quota"], res2.get("quota", SessionQuota(agent_count=0)))
    
    # Spawn 3 (Fail)
    state["next_steps"] = ["s3"]
    res3 = supervisor.spawning_node(state)
    assert "Failed to spawn s3" in res3["messages"][0]["content"]
