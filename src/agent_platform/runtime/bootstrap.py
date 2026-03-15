import structlog
import uuid
import os
from typing import Optional
from ..logging_config import configure_logging
from .core.workspace import WorkspaceContext
from .core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from .core.agent_factory import AgentFactory
from .core.lifecycle import AgentLifecycleManager

log = structlog.get_logger()

def start_runtime(
    user_id: str, 
    session_id: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    model_name: str = "gpt-4o"
):
    """
    Bootstraps the platform for a specific user and session.
    """
    workspace = WorkspaceContext()
    
    # 1. Resolve IDs
    is_new_session = False
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        is_new_session = True
    else:
        session_path = workspace.get_session_dir(user_id, session_id)
        if not session_path.exists():
            is_new_session = True

    # 2. Initialize / Resolve Path
    if is_new_session:
        res_mgr = SimpleCopyResourceManager()
        initializer = SessionInitializer(workspace, res_mgr)
        session_path = initializer.initialize(user_id, session_id)
    else:
        session_path = workspace.get_session_dir(user_id, session_id)

    # 3. Configure Logging
    log_file = session_path / "platform.log"
    configure_logging(log_file=log_file)
    
    log.info("runtime_bootstrap", 
             user_id=user_id, 
             session_id=session_id, 
             model_name=model_name,
             is_new=is_new_session)

    # 4. Setup Components
    factory = AgentFactory(workspace)
    lifecycle = AgentLifecycleManager(workspace, factory)
    
    # The model_name would be passed to agent initializers in the Scheduler/Service loop
    
    return session_id
