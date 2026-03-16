import asyncio
import os
import json
from pathlib import Path
from agent_platform.runtime.core.dispatcher import ToolRegistry
from agent_platform.runtime.core.tools.filesystem import FilesystemTools
from agent_platform.runtime.storage.context_tool import ContextTools
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.core.schema import ToolSource

async def test_registry():
    tmp_path = Path("tmp_test_registry")
    tmp_path.mkdir(exist_ok=True)
    
    registry = ToolRegistry(tmp_path)
    
    # Register some native tools
    fs_tools = FilesystemTools(tmp_path)
    registry.register_native("ls", fs_tools.ls, source=ToolSource.CORE)
    registry.register_native("read_file", fs_tools.read_file, source=ToolSource.CORE)
    
    store = FilesystemContextStore(tmp_path)
    context_tools = ContextTools(store, knowledge_path=tmp_path/"knowledge")
    registry.register_native("update_knowledge", context_tools.update_knowledge, source=ToolSource.CORE)

    # Check manifest
    manifest = registry.get_tool_manifest()
    print("--- Tool Manifest ---")
    print(manifest)
    
    # Check parameters for ls
    ls_meta = registry.metadata["ls"]
    print(f"ls parameters: {ls_meta['parameters']}")
    assert "path" in ls_meta["parameters"]
    
    # Check parameters for update_knowledge
    uk_meta = registry.metadata["update_knowledge"]
    print(f"update_knowledge parameters: {uk_meta['parameters']}")
    assert "name" in uk_meta["parameters"]
    assert "content" in uk_meta["parameters"]

    print("--- Test Passed ---")

if __name__ == "__main__":
    asyncio.run(test_registry())
