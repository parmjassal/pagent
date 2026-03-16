import json
import re
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def robust_json_parser(text: str) -> Dict[str, Any]:
    """
    Attempts to extract and parse JSON from a string that might contain 
    extra text, reasoning tags (like <think> or <thought>), tool outputs,
    or markdown code blocks.
    """

    # ------------------------------------------------------------------
    # 0. Normalize text (ADDITIVE)
    # ------------------------------------------------------------------
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # ------------------------------------------------------------------
    # 1. Strip reasoning tags like <think>...</think> (existing)
    # ------------------------------------------------------------------
    text = re.sub(r'<(think|thought)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # ------------------------------------------------------------------
    # 1.5 Remove hallucinated tool output sections (ADDITIVE)
    # Example: [Tool Result: read_file] {...}
    # ------------------------------------------------------------------
    text = re.sub(r'\[Tool Result:.*?\](?:\n|\r\n)?.*?(?=\n\[|$)', '', text, flags=re.DOTALL)

    # Also remove similar system tool lines
    text = re.sub(r'\[System\].*?(?=\n\[|$)', '', text, flags=re.DOTALL)

    # ------------------------------------------------------------------
    # 2. Try to find JSON block in markdown code fences (existing)
    # ------------------------------------------------------------------
    json_block_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    for block in json_block_matches:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # ------------------------------------------------------------------
    # 2.5 Handle unclosed markdown fences (ADDITIVE)
    # ```json { ... 
    # ------------------------------------------------------------------
    open_fence = re.search(r'```(?:json)?\s*(\{.*)', text, re.DOTALL)
    if open_fence:
        candidate = open_fence.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ------------------------------------------------------------------
    # 3. Balanced brace extraction (ADDITIVE - safer than find/rfind)
    # ------------------------------------------------------------------
    def extract_balanced_json(s: str):
        stack = []
        start = None

        for i, ch in enumerate(s):
            if ch == '{':
                if start is None:
                    start = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        return s[start:i+1]
        return None

    candidate = extract_balanced_json(text)

    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ------------------------------------------------------------------
    # 4. Original heuristic: first '{' to last '}' (existing)
    # ------------------------------------------------------------------
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]

            if "'" in json_str and '"' not in json_str:
                json_str = json_str.replace("'", '"')
                json_str = json_str.replace("True", "true")
                json_str = json_str.replace("False", "false")
                json_str = json_str.replace("None", "null")

            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)

            return json.loads(json_str)

    except json.JSONDecodeError:
        pass

    # ------------------------------------------------------------------
    # 4.5 Handle Python dict-style outputs (ADDITIVE)
    # ------------------------------------------------------------------
    python_dict_match = re.search(r'\{.*\}', text, re.DOTALL)
    if python_dict_match:
        candidate = python_dict_match.group(0)

        try:
            fixed = candidate.replace("'", '"')
            fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
            fixed = re.sub(r',\s*\}', '}', fixed)
            fixed = re.sub(r',\s*\]', ']', fixed)

            return json.loads(fixed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 5. Existing manual extraction heuristic (unchanged)
    # ------------------------------------------------------------------
    extracted = {}

    thought_match = re.search(
        r'"?thought_process"?\s*(?::|is)\s*"?(.*?)"?(?:\s*and|,|\.|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )

    if thought_match:
        extracted["thought_process"] = thought_match.group(1).strip()

    strategy_match = re.search(
        r'"?strategy"?\s*(?::|is)\s*"?(decompose|tool_use|finish)"?',
        text,
        re.IGNORECASE
    )

    if strategy_match:
        extracted["strategy"] = strategy_match.group(1).lower()

    if "thought_process" in extracted and "strategy" in extracted:
        return extracted

    # ------------------------------------------------------------------
    # 6. Final fallback (existing)
    # ------------------------------------------------------------------
    return json.loads(text)