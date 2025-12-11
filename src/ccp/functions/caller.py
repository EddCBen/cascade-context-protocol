import logging
import torch
from typing import Any, Dict, List, Optional, Callable
from src.ccp.neural.models import SoftmaxRouter
from src.ccp.memory.long_term import QdrantMemory
import src.ccp.functions.utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class FunctionCaller:
    """
    Handles function execution with dual-routing (Neural Intuition vs Memory Search).
    """
    def __init__(self, domain_id: str = "default", llm_service=None, input_dim: int = 768, num_tools: int = 10):
        # Tools Registry
        self.tools: Dict[str, Callable] = {}
        self._register_default_tools()
        
        # Neural Router (The Basal Ganglia)
        self.router = SoftmaxRouter(input_dim=input_dim, num_tools=num_tools)
        self.router.load_weights(domain_id)
        self.router.eval()
        
        # Memory (Fallback)
        self.memory = QdrantMemory(domain_id=domain_id, llm_service=llm_service, input_dim=input_dim)
        
        # ID to Tool Name mapping (mock for now, should align with router training)
        self.idx_to_tool = {i: f"tool_{i}" for i in range(num_tools)}
        # Manually map 0 and 1 to our known utils for demonstration if needed, 
        # or we just rely on string names from memory search.
        self.idx_to_tool[0] = "parse_json_safely"
        self.idx_to_tool[1] = "sanitize_input"

    def _register_default_tools(self):
        """
        Registers tools from src.ccp.functions.utils.
        """
        # In a real scenario, we might inspect for the @expose_as_ccp_tool decorator
        self.tools["parse_json_safely"] = utils.parse_json_safely
        self.tools["sanitize_input"] = utils.sanitize_input

    def call_function(self, context_vector: torch.Tensor, block_id: str, args: Dict[str, Any]) -> Any:
        """
        Executes a function based on context.
        """
        # Ensure context_vector is right shape for router (Batch=1, Dim)
        if context_vector.dim() == 1:
            router_input = context_vector.unsqueeze(0)
        else:
            router_input = context_vector
            
        # 1. Check Neural Router (Fast Path)
        with torch.no_grad():
            probs = self.router(router_input)
            max_prob, tool_idx = torch.max(probs, dim=-1)
            tool_idx = tool_idx.item()
            confidence = max_prob.item()

        selected_tool_name = None
        
        if confidence > 0.9:
            # Fast Path / Intuition
            logger.info(f"[STEP: STAGE_FUNCTION] [ID: {block_id}] [INPUT: {args}] - Fast Path (Conf: {confidence:.2f})")
            if tool_idx in self.idx_to_tool:
                selected_tool_name = self.idx_to_tool[tool_idx]
        else:
            # Slow Path / Deliberation
            logger.info(f"[STEP: STAGE_FUNCTION] [ID: {block_id}] [INPUT: {args}] - Slow Path (Conf: {confidence:.2f})")
            # Creating a dummy query string from args or context for search
            query = str(args) 
            results = self.memory.search_functions(query, top_k=1)
            if results:
                # Use real ID from Qdrant result
                selected_tool_name = str(results[0]["id"]) 

        # Execute
        if selected_tool_name and selected_tool_name in self.tools:
            logger.info(f"Executing tool: {selected_tool_name}")
            try:
                # We assume args map to function kwargs for simplicity 
                # or we just pass the raw input if function expects single arg.
                # Here we blindly attempt to pass **args or single arg.
                func = self.tools[selected_tool_name]
                # Simple logic for the utils we know
                if selected_tool_name == "parse_json_safely":
                     # Expecting 'json_str' in args
                     val = args.get("json_str", "{}")
                     return func(val)
                elif selected_tool_name == "sanitize_input":
                    val = args.get("input_str", "")
                    return func(val)
                else:
                    return f"Executed {selected_tool_name}"
            except Exception as e:
                logger.error(f"Error executing {selected_tool_name}: {e}")
                return str(e)
        else:
            logger.warning(f"No suitable tool found or tool not registered: {selected_tool_name}")
            return None
