import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.quota import SessionQuota
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.prompt_writer import DynamicPromptWriter
from agent_platform.runtime.state import create_initial_state
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def workspace(tmp_path):
    return WorkspaceContext(root=tmp_path)

@pytest.fixture
def mailbox(tmp_path):
    provider = FilesystemMailboxProvider(tmp_path / "sess1")
    return Mailbox(provider)

@pytest.fixture
def prompt_writer(workspace):
    # Pass a dummy llm for testing (not used in current simulated implementation)
    return DynamicPromptWriter(llm=None, workspace=workspace)

def test_supervisor_full_spawning_sequence(workspace, mailbox, prompt_writer):
    # Setup Factory
    factory = AgentFactory(workspace, max_spawn_depth=5)
    supervisor = SupervisorAgent(factory, mailbox, prompt_writer, api_key="dummy-key")
    
    # Initial state
    state = create_initial_state("super", "u1", "s1", Path("/tmp/in"), Path("/tmp/out"))
    
    # 1. Test Decomposition -> Next steps added
    state_after_decomp = supervisor.task_decomposition_node(state)
    assert "researcher_1" in state_after_decomp["next_steps"]

    # 2. Test Prompt Generation
    state.update(state_after_decomp) # Update original state with decomp results
    state_after_prompt = prompt_writer.generate_prompt_node(state)
    assert "generated_prompt" in state_after_prompt
    assert "researcher_1" in state_after_prompt["generated_prompt"]

    # 3. Test Spawning
    state.update(state_after_prompt) # Update with prompt results
    state_after_spawn = supervisor.spawning_node(state)
    assert "Successfully spawned" in state_after_spawn["messages"][0]["content"]
    
    # Check Mailbox
    received = mailbox.receive("researcher_1")
    assert "system_prompt" in received
    assert received["system_prompt"] is not None
