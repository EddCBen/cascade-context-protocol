from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from uuid import uuid4
from enum import Enum

class NodeState(str, Enum):
    NEW = "new"
    IN_PROGRESS = "in-progress"
    FINISHED = "finished"

class ExecutionNode(BaseModel):
    id: str
    type: str = "default" # input, llm, output, tool
    label: str
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    data: Dict[str, Any] = Field(default_factory=dict)

class ExecutionEdge(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = None

class ExecutionGraph(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: List[ExecutionNode] = Field(default_factory=list)
    edges: List[ExecutionEdge] = Field(default_factory=list)

class ContextBlock(BaseModel):
    """
    Standard unit of streaming output.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    type: str = "text" # text, tool_call, tool_output, error, graph_update
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution Graph Metadata
    execution_graph_id: Optional[str] = None
    node_id: Optional[str] = None
    connections: List[str] = Field(default_factory=list) # IDs of connected target nodes
    state: NodeState = NodeState.NEW
