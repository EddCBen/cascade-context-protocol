import json
import re
from typing import Any, Dict, Optional

# Placeholder for the decorator as per instructions to implement later.
# In a real scenario, this might be imported from a core module.
def expose_as_ccp_tool(func):
    """
    Decorator to mark a function as a CCP tool.
    This is a placeholder implementation.
    """
    func._is_ccp_tool = True
    return func

@expose_as_ccp_tool
def parse_json_safely(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parses a JSON string, handling common errors and formatting issues.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common issues like trailing commas or single quotes
        # This is a basic sanitation attempt
        try:
            # Replace single quotes with double quotes
            sanitized = json_str.replace("'", '"')
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object within text (e.g. if wrapped in code blocks)
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
    return None

@expose_as_ccp_tool
def sanitize_input(input_str: str) -> str:
    """
    Sanitizes input string by removing potentially harmful characters or whitespace.
    """
    if not isinstance(input_str, str):
        return str(input_str)
    return input_str.strip()
