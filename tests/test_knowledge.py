import pytest
from pathlib import Path
from agent_platform.runtime.knowledge import FilesystemKnowledgeManager

def test_knowledge_persistence(tmp_path):
    km = FilesystemKnowledgeManager(tmp_path / "knowledge")
    
    # Store
    km.store_fact("user_auth", "Logic for user login using JWT.", "src/auth.py#L10-L50")
    
    # List
    facts = km.list_facts()
    assert "user_auth" in facts
    
    # Retrieve
    content = km.retrieve_fact("user_auth")
    assert "# Fact: user_auth" in content
    assert "JWT" in content
    assert "src/auth.py#L10-L50" in content

def test_knowledge_key_sanitization(tmp_path):
    km = FilesystemKnowledgeManager(tmp_path / "knowledge")
    km.store_fact("complex key with/slash", "content", "ptr")
    
    facts = km.list_facts()
    # Key should be sanitized in filename but remain discoverable
    assert "complex_key_with_slash" in facts
