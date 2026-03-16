import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def robust_json_parser(text: str) -> Dict[str, Any]:
    """
    Attempts to extract and parse JSON from a string that might contain 
    extra text, reasoning tags (like <think> or <thought>), or markdown code blocks.
    """
    # 1. Strip reasoning tags like <think>...</think> or <thought>...</thought>
    text = re.sub(r'<(think|thought)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Try to find JSON block in markdown code fences
    # Prioritize ```json blocks
    json_block_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    for block in json_block_matches:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
            
    # 3. Try to find the first '{' and last '}'
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            
            # Simple heuristic to handle single quotes if double quotes are missing
            if "'" in json_str and '"' not in json_str:
                # Replace single quotes with double quotes (dangerous but useful for some LLMs)
                json_str = json_str.replace("'", '"')
                # Handle Python booleans/None
                json_str = json_str.replace("True", "true").replace("False", "false").replace("None", "null")

            # Try to fix common issues like trailing commas before closing braces
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 4. Heuristic: If it's a chatty response with no JSON, try to extract key fields manually
    # Looking for "thought_process": "...", "strategy": "..."
    # This is very loose and might not work for all models
    extracted = {}
    
    # Try to find "thought_process": "..." or thought_process is ...
    thought_match = re.search(r'"?thought_process"?\s*(?::|is)\s*"?(.*?)"?(?:\s*and|,|\.|$)', text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        extracted["thought_process"] = thought_match.group(1).strip()
    
    strategy_match = re.search(r'"?strategy"?\s*(?::|is)\s*"?(decompose|tool_use|finish)"?', text, re.IGNORECASE)
    if strategy_match:
        extracted["strategy"] = strategy_match.group(1).lower()
        
    if "thought_process" in extracted and "strategy" in extracted:
        return extracted

    # 5. Last resort: raw parse
    return json.loads(text)
