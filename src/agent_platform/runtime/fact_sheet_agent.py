import logging
from typing import Dict, Any, Optional
from .state import AgentState
from .knowledge import KnowledgeManager, FilesystemKnowledgeManager

logger = logging.getLogger(__name__)

class FactSheetAgent:
    """
    Specialized agent for extracting and aggregating knowledge from file chunks.
    Maintains context without passing around raw large file content.
    """

    def __init__(self, knowledge_manager: KnowledgeManager):
        self.km = knowledge_manager

    def extract_fact_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Extracts a specific fact from a provided chunk and persists it.
        """
        chunk_metadata = state.get("metadata", {}).get("current_chunk")
        if not chunk_metadata:
            return {"messages": [{"role": "system", "content": "FactSheetAgent: No chunk provided."}]}

        # Simulate LLM Extraction logic
        # In production: response = self.llm.invoke(...)
        fact_key = f"logic_{chunk_metadata['path'].split('/')[-1]}_{chunk_metadata['start_line']}"
        extracted_content = f"Summary of lines {chunk_metadata['start_line']}-{chunk_metadata['end_line']}: Implementation of core logic."
        source_ptr = f"{chunk_metadata['path']}#L{chunk_metadata['start_line']}-{chunk_metadata['end_line']}"

        # Persist Fact
        self.km.store_fact(fact_key, extracted_content, source_ptr)

        return {
            "messages": [{"role": "system", "content": f"FactSheetAgent: Extracted and persisted knowledge for {fact_key}"}],
            "metadata": {"last_fact_id": fact_key}
        }

    def list_knowledge_node(self, state: AgentState) -> Dict[str, Any]:
        """Lists all known facts to give the LLM an overview of available context."""
        facts = self.km.list_facts()
        summary = "\n".join([f"- {f}" for f in facts])
        return {
            "messages": [{"role": "assistant", "content": f"Known Knowledge Base:\n{summary}"}]
        }
