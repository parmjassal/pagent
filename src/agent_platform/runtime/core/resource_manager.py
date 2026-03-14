import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
from .workspace import WorkspaceContext

class ResourceManager(ABC):
    @abstractmethod
    def copy_resources(self, source_dir: Path, target_dir: Path):
        """Copies resources from source to target."""
        pass

class SimpleCopyResourceManager(ResourceManager):
    """Initial implementation using shutil for straightforward directory copying."""

    def copy_resources(self, source_dir: Path, target_dir: Path):
        if not source_dir.exists():
            return

        # Simple copy logic - ensuring it's not a symlink
        for item in source_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True, symlinks=False)
            else:
                shutil.copy2(item, target_dir / item.name)

class SessionInitializer:
    """Orchestrates the creation and hydration of a session context."""

    def __init__(self, workspace: WorkspaceContext, resource_manager: ResourceManager):
        self.workspace = workspace
        self.resource_manager = resource_manager

    def initialize(self, user_id: str, session_id: str):
        # 1. Create directory structure
        session_path = self.workspace.ensure_session_structure(user_id, session_id)

        # 2. Inherit/Copy Resources (Global -> User -> Session)
        # This order allows user overrides of global, and session is hydrated with the result.
        global_dir = self.workspace.get_global_dir()
        user_dir = self.workspace.get_user_dir(user_id)

        # Copy Global to Session
        self.resource_manager.copy_resources(global_dir / "skills", session_path / "skills")
        self.resource_manager.copy_resources(global_dir / "prompts", session_path / "prompts")
        
        # Explicitly copy guidelines if present
        if (global_dir / "guidelines.md").exists():
            shutil.copy2(global_dir / "guidelines.md", session_path / "guidelines.md")

        # Copy User specific (overriding global if collision)
        self.resource_manager.copy_resources(user_dir / "skills", session_path / "skills")
        self.resource_manager.copy_resources(user_dir / "prompts", session_path / "prompts")
        if (user_dir / "guidelines.md").exists():
            shutil.copy2(user_dir / "guidelines.md", session_path / "guidelines.md")

        return session_path
