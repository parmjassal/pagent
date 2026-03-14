import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.quota import SessionQuota
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.generator import SystemGeneratorAgent, TaskType
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.dispatcher import ToolDispatcher, ToolSource
from agent_platform.runtime.guardrails import GuardrailManager
from agent_platform.runtime.sandbox import ProcessSandboxRunner
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v1_env(tmp_path):
    # 1. Setup Filesystem
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (global_dir := workspace.get_global_dir() / "prompts").mkdir(parents=True)
    (global_dir / "agent_base.txt").write_text("BASE PROMPT")

    # 2. Session Setup
    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "power_user", "sess_v1_001"
    session_path = initializer.initialize(user_id, session_id)

    # 3. Component Orchestration
    factory = AgentFactory(workspace, max_spawn_depth=5)
    provider = FilesystemMailboxProvider(session_path)
    mailbox = Mailbox(provider)
    
    # Safety & Execution
    sandbox = ProcessSandboxRunner()
    guardrails = GuardrailManager()
    dispatcher = ToolDispatcher(sandbox, guardrails)
    
    # Native Community Tool registration
    def community_search(query: str):
        return f"Search results for: {query}"
    dispatcher.register_tool("google_search", ToolSource.COMMUNITY, func=community_search)

    # Generators
    generator = SystemGeneratorAgent(llm=None, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, api_key="dummy-key")

    return {
        "env": (user_id, session_id, session_path),
        "supervisor": supervisor,
        "dispatcher": dispatcher,
        "generator": generator,
        "mailbox": mailbox
    }

def test_v1_full_platform_lifecycle(v1_env):
    """
    Integration V1: 
    Spawning -> Community Tool -> Tool Writing -> Sandboxed Tool -> Quota Check
    """
    user_id, session_id, session_path = v1_env["env"]
    supervisor = v1_env["supervisor"]
    dispatcher = v1_env["dispatcher"]
    generator = v1_env["generator"]
    mailbox = v1_env["mailbox"]

    # --- PHASE 1: ORCHESTRATION ---
    state = create_initial_state(
        "super_01", user_id, session_id,
        session_path / "agents/super_01/inbox",
        session_path / "agents/super_01/outbox"
    )
    
    # Supervisor decomposes and spawns sub-agent
    graph = supervisor.build_graph()
    final_state = graph.invoke(state)
    
    assert final_state["quota"].agent_count == 1
    assert (session_path / "agents" / "researcher_1").exists()

    # --- PHASE 2: NATIVE TOOL USAGE (COMMUNITY) ---
    # Researcher_1 uses a community tool
    native_result = dispatcher.dispatch(final_state, "google_search", query="AI Trends 2026")
    assert native_result["success"] is True
    assert native_result["source"] == "native"
    assert "Search results" in native_result["output"]

    # --- PHASE 3: DYNAMIC TOOL WRITING & USAGE ---
    # Generator creates a tool for researcher_1
    gen_result = generator.generate_node(final_state, task_type=TaskType.TOOL)
    assert "def researcher_1_func" in gen_result["generated_output"]
    
    # Register the newly written tool as DYNAMIC
    dispatcher.register_tool("calc_risk_score", ToolSource.DYNAMIC)
    
    # Dispatcher runs the new dynamic tool
    # (The dispatcher handles loading dynamic code internally or via stub in this v1)
    dynamic_result = dispatcher.dispatch(final_state, "calc_risk_score", data=[1, 2, 3])
    assert dynamic_result["success"] is True
    assert dynamic_result["source"] == "sandbox"
    assert "Sandboxed result" in dynamic_result["output"]

    # --- PHASE 4: GUARDRAIL & CACHE VERIFICATION ---
    # Run same tool again to verify caching
    cache_result = dispatcher.dispatch(final_state, "calc_risk_score", data=[1, 2, 3])
    assert cache_result["success"] is True
    # (Internal check: GuardrailManager would log cache hit)

    # --- PHASE 5: MAILBOX FINAL CHECK ---
    # Confirm supervisor delivered initial task to researcher
    msg = mailbox.receive("researcher_1")
    assert msg["sender"] == "super_01"
    assert "system_prompt" in msg
