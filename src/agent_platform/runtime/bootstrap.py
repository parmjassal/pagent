import structlog
import uuid
from typing import Optional
from ..logging_config import configure_logging
from .workspace import WorkspaceContext
from .resource_manager import SimpleCopyResourceManager, SessionInitializer
from .agent_factory import AgentFactory
from .lifecycle import AgentLifecycleManager

log = structlog.get_logger()

def start_runtime(user_id: str, session_id: Optional[str] = None):
    """
    Bootstraps the platform for a specific user and session.
    - If session_id is None, a new one is created and initialized.
    - If session_id exists, we resume without re-copying resources (Persistence).
    """
    configure_logging()
    workspace = WorkspaceContext()
    
    # 1. Resolve IDs
    is_new_session = False
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        is_new_session = True
        log.info("creating_new_session", user_id=user_id, session_id=session_id)
    else:
        session_path = workspace.get_session_dir(user_id, session_id)
        if not session_path.exists():
            is_new_session = True
            log.info("session_id_not_found_creating_new", user_id=user_id, session_id=session_id)
        else:
            log.info("resuming_existing_session", user_id=user_id, session_id=session_id)

    # 2. Initialize only if new
    if is_new_session:
        res_mgr = SimpleCopyResourceManager()
        initializer = SessionInitializer(workspace, res_mgr)
        session_path = initializer.initialize(user_id, session_id)
    else:
        session_path = workspace.get_session_dir(user_id, session_id)
    
    log.info("runtime_ready", 
             user_id=user_id, 
             session_id=session_id, 
             path=str(session_path))

    # 3. Setup Core Orchestration
    # The factory will respect the existing max_spawn_depth and quota stored in state.db
    factory = AgentFactory(workspace)
    lifecycle = AgentLifecycleManager(workspace, factory)
    
    log.info("waiting_for_tasks", session_id=session_id)
    
    return session_id
