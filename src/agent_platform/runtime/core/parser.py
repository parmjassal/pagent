import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def robust_json_parser(text: str) -> Dict[str, Any]:
    """
    Attempts to extract and parse JSON from a string that might contain 
    extra text, reasoning tags (like <think>), or markdown code blocks.
    """
    # 1. Strip reasoning tags like <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. Try to find JSON block in markdown code fences
    code_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
            
    # 3. Try generic code block if json one failed
    generic_block_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if generic_block_match:
        try:
            return json.loads(generic_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 4. Try to find the first '{' and last '}'
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 5. Last resort: raw parse
    return json.loads(text)
