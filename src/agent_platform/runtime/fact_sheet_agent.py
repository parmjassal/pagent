import logging
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage
from .state import AgentState
from .knowledge import KnowledgeManager

logger = logging.getLogger(__name__)

class FactSheetAgent:
    """
    Specialized agent for extracting and aggregating knowledge from file chunks.
    Maintains context without passing around raw large file content.
    """

    def __init__(self, knowledge_manager: KnowledgeManager, llm: Optional[Any] = None):
        self.km = knowledge_manager
        self.llm = llm

    def extract_fact_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Extracts a specific fact from a provided chunk using the LLM and persists it.
        """
        chunk_metadata = state.get("metadata", {}).get("current_chunk")
        if not chunk_metadata:
            return {"messages": [{"role": "system", "content": "FactSheetAgent: No chunk provided."}]}

        # 1. Prepare Extraction Prompt
        # In a real scenario, we'd read the actual file lines using the metadata pointers
        source_ptr = f"{chunk_metadata['path']}#L{chunk_metadata['start_line']}-{chunk_metadata['end_line']}"
        instruction = f"Analyze the following file chunk and extract key facts/logic:\nSource: {source_ptr}"
        
        # 2. Invoke LLM (Mocked in tests, real in production)
        # We expect a string or a structured response
        response = self.llm.invoke([SystemMessage(content=instruction)])
        extracted_content = response.content if hasattr(response, "content") else str(response)

        # 3. Resolve Fact Key
        fact_key = f"fact_{chunk_metadata['path'].split('/')[-1]}_{chunk_metadata['start_line']}"

        # 4. Persist Fact
        self.km.store_fact(fact_key, extracted_content, source_ptr)

        log_msg = f"FactSheetAgent: Extracted and persisted knowledge for {fact_key}"
        return {
            "messages": [{"role": "system", "content": log_msg}],
            "metadata": {"last_fact_id": fact_key}
        }

    def list_knowledge_node(self, state: AgentState) -> Dict[str, Any]:
        """Lists all known facts to give the LLM an overview of available context."""
        facts = self.km.list_facts()
        summary = "\n".join([f"- {f}" for f in facts])
        return {
            "messages": [{"role": "assistant", "content": f"Known Knowledge Base:\n{summary}"}]
        }
