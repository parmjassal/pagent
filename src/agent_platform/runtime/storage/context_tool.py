import secrets
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.context_store import ContextStore
from ..orch.state import AgentState

logger = logging.getLogger(__name__)

class ContextTools:
    """
    Unified tools for agents to interact with:
    1. Branch Visibility (hierarchical global_context) via update_context (Downward flow).
    2. Global Visibility (session knowledge) via update_knowledge (Reasoning-driven).
    """
    def __init__(self, store: ContextStore, knowledge_path: Optional[Path] = None):
        self.store = store
        self.knowledge_path = knowledge_path

    def list_context(self, state: AgentState) -> List[str]:
        """Lists all fact IDs visible to the current agent in the branch hierarchy."""
        return self.store.list_facts(state["agent_id"])

    def fetch_context(self, state: AgentState, fact_id: str) -> Optional[str]:
        """Reads a specific fact from the branch hierarchy."""
        return self.store.read_fact(state["agent_id"], fact_id)

    def update_context(self, state: AgentState, fact_id: str, content: str) -> str:
        """Allows supervisors to add or update a fact in their own branch context. It is visible to only sub agents."""
        try:
            self.store.update_fact(state["agent_id"], fact_id, content, state["role"])
            return f"Successfully updated fact '{fact_id}' in branch global_context."
        except PermissionError as e:
            return str(e)

    def update_knowledge(self, state: AgentState, name: str, content: str) -> str:
        """
        Promotes a result to Global Knowledge. It is visible to all agents.
        Ensures naming convention: [8-char-hex-prefix]_[contextual_name].json
        """
        if not self.knowledge_path:
            return "Error: Global knowledge path not configured in ContextTools."

        try:
            self.knowledge_path.mkdir(parents=True, exist_ok=True)
            
            # Generate prefix and final path
            prefix = secrets.token_hex(4).upper()
            # Sanitize name
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
            file_name = f"{prefix}_{safe_name}.json"
            target_path = self.knowledge_path / file_name

            # Write content (ensure it's valid JSON if possible, else wrap)
            try:
                if isinstance(content, str):
                    # Try to parse as JSON to validate, then re-dump
                    json_data = json.loads(content)
                else:
                    json_data = content
                target_path.write_text(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                # If not valid JSON, wrap it
                json_data = {"raw_content": content, "source_agent": state["agent_id"]}
                target_path.write_text(json.dumps(json_data, indent=2))

            return f"Successfully promoted to Global Knowledge: knowledge/{file_name}"
        except Exception as e:
            return f"Failed to update global knowledge: {str(e)}"

    def list_knowledge(self, state: AgentState) -> List[str]:
        """
        Lists all Global Knowledge files available in the session knowledge directory.
        Returns the file names so agents can decide which knowledge artifact to fetch.
        """
        if not self.knowledge_path or not self.knowledge_path.exists():
            return []

        return [p.name for p in self.knowledge_path.glob("*.json")]


    def fetch_knowledge(self, state: AgentState, name: str) -> Optional[str]:
        if not self.knowledge_path:
            return None

        try:
            # 1. Absolute path debugging
            abs_path = self.knowledge_path.resolve()
            logger.info(f"[DEBUG] Searching in directory: {abs_path}")
            
            if not abs_path.exists():
                logger.error(f"[DEBUG] Path does not exist: {abs_path}")
                return None

            # 2. Case-insensitive search & flexible extension matching
            # We lowercase everything to ensure a match
            search_pattern = name.lower()
            matches = [
                p for p in abs_path.iterdir() 
                if p.is_file() and search_pattern in p.name.lower()
            ]

            logger.info(f"[DEBUG] Searching for: {name}")
            logger.info(f"[DEBUG] Found: {[p.name for p in matches]}")

            if not matches:
                return None

            combined = []
            for path in matches:
                try:
                    # Specify encoding to avoid Mac/Windows mismatches
                    text = path.read_text(encoding="utf-8")
                    combined.append(f"{path.name}\n\t{text}\n")
                except Exception as e:
                    logger.warning(f"Could not read {path.name}: {e}")
                    continue

            return "".join(combined) if combined else None

        except Exception as e:
            logger.error(f"Error in fetch_knowledge: {e}")
            return None

    def search_context(self, state: AgentState, query: str) -> str:
        """
        Performs a semantic-ish search across visible global_context files.
        (Currently implements simple keyword matching across the hierarchy).
        """
        visible_facts = self.store.list_facts(state["agent_id"])
        results = []
        for fid in visible_facts:
            content = self.store.read_fact(state["agent_id"], fid)
            if query.lower() in content.lower():
                results.append(f"### {fid}\n{content[:200]}...")
        
        if not results:
            return "No matching context found."
        return "\n\n".join(results)

