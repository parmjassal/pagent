import logging
import json
import secrets
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ResultHook(ABC):
    """
    Interface for processing agent outputs before they are returned to parents.
    Handles data offloading (Summary + File Ref) for large payloads.
    """
    @abstractmethod
    def process_result(self, agent_id: str, result: Any) -> Dict[str, Any]:
        """Returns a result object, potentially offloaded to disk."""
        pass

class OffloadingResultHook(ResultHook):
    """
    Standard implementation that offloads results larger than 'threshold' 
    to the session knowledge directory and returns a reference.
    """
    def __init__(self, knowledge_storage_path: Path, threshold_bytes: int = 10240): # Default 10KB
        self.storage_path = knowledge_storage_path
        self.threshold = threshold_bytes
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def process_result(self, agent_id: str, result: Any) -> Dict[str, Any]:
        # Convert to string/json to check size
        raw_data = json.dumps(result) if not isinstance(result, str) else result
        size = len(raw_data.encode('utf-8'))

        if size <= self.threshold:
            return {"type": "inline", "content": result, "size": size}

        # Offload to knowledge directory with standard prefixing
        prefix = secrets.token_hex(4).upper()
        # Clean agent_id for filename
        safe_agent_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in agent_id)
        file_name = f"offload_{prefix}_{safe_agent_id}.json"
        target_path = self.storage_path / file_name
        target_path.write_text(raw_data)

        logger.info(f"Result for {agent_id} offloaded to {target_path} (Size: {size} bytes)")

        # Return a reference summary
        summary = raw_data[:200] + "..." if len(raw_data) > 200 else raw_data
        return {
            "type": "reference",
            "path": f"knowledge/{file_name}",
            "summary": f"Large result offloaded. Preview: {summary}",
            "size": size
        }
