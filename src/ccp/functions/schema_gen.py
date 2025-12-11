import inspect
from typing import get_type_hints, Any, Callable, Dict, List

def get_openai_tool_schema(func: Callable) -> Dict[str, Any]:
    """
    Generates an OpenAI tool schema from a Python function's type hints and docstring.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    doc = func.__doc__.strip() if func.__doc__ else "No description available."
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        # Map python types to JSON types
        param_type = type_hints.get(name, Any)
        json_type = "string"
        if param_type == int:
            json_type = "integer"
        elif param_type == float:
            json_type = "number"
        elif param_type == bool:
            json_type = "boolean"
        elif param_type == list or param_type == List:
             json_type = "array"
        elif param_type == dict or param_type == Dict:
             json_type = "object"
             
        parameters["properties"][name] = {
            "type": json_type,
            "description": f"Parameter {name}" # In a real implementation, parse docstring for per-param desc
        }
        
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": parameters
        }
    }
