import re
import hashlib
import logging
from typing import Dict, Any, Tuple
from .state import AgentState

logger = logging.getLogger(__name__)

class ResponseParser:
    """
    Utility to clean and parse LLM responses, handling thinking tags 
    and non-JSON preambles.
    """
    @staticmethod
    def clean_json_response(content: str) -> str:
        """Strips <think> tags and returns the underlying JSON string."""
        # 1. Remove <think>...</think> blocks
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 2. Find the first '{' and last '}' to isolate JSON block
        # This handles cases where LLM includes preamble before or after JSON
        start = content.find('{')
        end = content.rfind('}')

        if start != -1 and end != -1:
            return content[start:end+1]

        return content.strip()

class LoopMonitor:
    """
    Utility to detect repetitive behavior in LLM graphs.
    Checks for:
    1. Node execution thresholds.
    2. Repeated identical messages (Semantic loops).
    """
    @staticmethod
    def check_node_loop(state: AgentState, node_name: str, threshold: int = 3) -> bool:
        """Returns True if a specific node has been visited more than 'threshold' times."""
        count = state.get("node_counts", {}).get(node_name, 0)
        if count >= threshold:
            logger.warning(f"Loop detected: Node '{node_name}' executed {count} times.")
            return True
        return False

    @staticmethod
    def check_content_loop(state: AgentState, window: int = 3) -> bool:
        """Returns True if the last 'window' messages are identical."""
        msgs = state.get("messages", [])
        if len(msgs) < window:
            return False
        
        last_n = [m["content"] for m in msgs[-window:]]
        # If all items in last_n are the same
        if len(set(last_n)) == 1:
            logger.warning("Loop detected: LLM generated identical content multiple times.")
            return True
        return False

    @staticmethod
    def get_update(node_name: str) -> Dict[str, Any]:
        """Convenience to return the state update for node tracking."""
        return {"node_counts": {node_name: 1}}
