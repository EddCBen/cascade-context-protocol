import os
import asyncio
import logging
import torch
from typing import Optional, AsyncGenerator, List, Any
from src.ccp.models.context import ContextBlock, ExecutionGraph, ExecutionNode, ExecutionEdge, NodeState
from src.ccp.storage.mongo import MongoStorage
from src.ccp.neural.models import VectorNormalizer, SoftmaxRouter
from src.ccp.core.settings import settings
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrates the context segmentation, tool retrieval, and execution process.
    Implements the Neuro-Symbolic Pipeline.
    """
    def __init__(self, granularity: float, llm_service=None):
        """
        Args:
            granularity: Float between 0.0 and 1.0. 
            llm_service: Instance of LLMService.
        """
        if not 0.0 <= granularity <= 1.0:
            raise ValueError("Granularity must be between 0.0 and 1.0")
        
        self.granularity = granularity
        self.llm_service = llm_service
        self.mongo_storage = MongoStorage()

        # Neural Components
        self.device = torch.device("cpu") # Keep simple for now
        self.normalizer = VectorNormalizer(input_dim=settings.embedding_dim).to(self.device)
        self.router = SoftmaxRouter(input_dim=settings.embedding_dim, num_tools=20).to(self.device)
        
        # Load weights if available (Mock/Init for now, normally would load from disk)
        # self.normalizer.load_weights("default")
        
        # Retrieval Components
        try:
             self.qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        except:
             self.qdrant = None
             logger.warning("Qdrant not connected in Orchestrator.")

        # Default values for formula
        self.average_token_per_block = 500  # Estimate

    def calculate_segmentation_depth(self) -> int:
        """
        Formula: target_blocks = (granularity * 10) + (context_window_remaining / average_token_per_block * 0.1)
        """
        context_remaining = 0
        if self.llm_service:
            context_remaining = self.llm_service.get_remaining_context()
        
        raw_target = (self.granularity * 10) + (context_remaining / self.average_token_per_block * 0.1)
        return int(max(1, raw_target))

    def get_segmentation_prompt(self) -> str:
        target_blocks = self.calculate_segmentation_depth()
        
        # Safe path resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(base_dir, "prompts")
        segmentation_file = os.path.join(prompts_dir, "segmentation.txt")

        try:
            with open(segmentation_file, "r") as f:
                template = f.read()
                return template.replace("{{target_blocks}}", str(target_blocks))
        except FileNotFoundError:
            return f"Error: Prompt file not found at {segmentation_file}"

    async def retrieve_tools(self, query: str) -> List[Any]:
        """
        Neuro-Symbolic Tool Retrieval:
        1. Embed Query.
        2. Normalize (Neural).
        3. Search (Qdrant).
        4. Return Function Objects.
        """
        if not self.llm_service or not self.qdrant:
            return []

        # 1. Embed
        raw_embedding = self.llm_service.get_embedding(query)
        if not raw_embedding:
            return []
        
        # 2. Normalize via Neural Net
        with torch.no_grad():
            tensor_emb = torch.tensor(raw_embedding, device=self.device).unsqueeze(0)
            normalized_emb = self.normalizer(tensor_emb).squeeze(0).tolist()
            
            # (Optional) Logits from router for future filtering
            # logits = self.router(tensor_emb)
            # logger.info(f"Router Logits: {logits}")

        # 3. Search Qdrant
        try:
            # Check collection exists
            # We assume it exists from LLMService init
            search_result = self.qdrant.search(
                collection_name="function_registry",
                query_vector=normalized_emb,
                limit=3  # Top-3 tools
            )
        except Exception as e:
            logger.error(f"Qdrant Search Error: {e}")
            return []

        # 4. Map back to functions
        # We need the actual function objects. 
        # LLMService -> Registry -> tool_map
        found_tools = []
        if self.llm_service and hasattr(self.llm_service, 'registry'):
             all_tools = self.llm_service.registry.tools # name -> func
             for hit in search_result:
                 tool_name = hit.payload.get("name")
                 if tool_name and tool_name in all_tools:
                     logger.info(f"[Orchestrator] Retrieved Tool: {tool_name} (Score: {hit.score:.4f})")
                     found_tools.append(all_tools[tool_name])
        
        return found_tools

    async def process_message_stream(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        """
        Processes the user message with streaming, Neural Retrieval, and persistence.
        """
        logger.info(f"[{session_id}] Processing new message: {message[:50]}...")

        # --- RETRIEVAL PHASE ---
        logger.info(f"[{session_id}] Running Neuro-Symbolic Retrieval...")
        retrieved_tools = await self.retrieve_tools(message)
        tool_names = [t.__name__ for t in retrieved_tools]
        
        # --- GRAPH INITIALIZATION ---
        graph = ExecutionGraph()
        
        node_start = ExecutionNode(id="start", label="Start", type="input", position={"x": 250, "y": 0})
        
        # Retrieval Node (New)
        node_retrieval = ExecutionNode(
            id="retrieval", 
            label="Neural Retrieval", 
            type="tool", # Using 'tool' type for visualization of internal step
            position={"x": 250, "y": 150},
            data={"normalized": True, "tools_found": tool_names}
        )

        # LLM Node
        provider_info = {
            "provider": self.llm_service.provider_type if self.llm_service else "unknown",
            "model": self.llm_service.model_name if self.llm_service else "unknown"
        }
        node_llm = ExecutionNode(
            id="llm_process", 
            label="LLM Processing", 
            type="llm", 
            position={"x": 250, "y": 300},
            data=provider_info
        )
        
        graph.nodes = [node_start, node_retrieval, node_llm]
        graph.edges = [
            ExecutionEdge(id="e1", source="start", target="retrieval"),
            ExecutionEdge(id="e2", source="retrieval", target="llm_process")
        ]
        
        # Emit Initial Graph
        yield ContextBlock(
            content="Initializing Neuro-Symbolic Graph...", 
            type="graph_update",
            metadata={"graph": graph.model_dump()},
            execution_graph_id=graph.id
        ).model_dump_json() + "\n"

        if not self.llm_service:
            yield ContextBlock(content="Error: LLM Service not initialized.", type="error").model_dump_json()
            return

        # 1. Save User Message
        try:
            await self.mongo_storage.save_message(session_id, "user", message)
        except Exception:
            pass

        # 2. Prepare Prompt
        system_instruction = self.get_segmentation_prompt()
        full_prompt = f"{system_instruction}\\n\\nUSER MESSAGE:\\n{message}"

        # 3. Initialize Semantic Chunker
        from src.ccp.core.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(
            chunk_mode="semantic",
            max_chunk_size=512,
            min_chunk_size=50
        )
        
        # 4. Stream LLM Response with Semantic Chunking
        stream = self.llm_service.generate_content_stream(full_prompt)
        full_response_text = ""
        reasoning_nodes = []  # Track dynamically created reasoning nodes
        
        try:
            for token in stream:
                if token:
                    full_response_text += token
                    
                    # Add token to chunker
                    block_data = chunker.add_token(token)
                    
                    if block_data:
                        # Create reasoning step node
                        step_num = block_data['metadata']['step']
                        node_id = f"reasoning_step_{step_num}"
                        
                        reasoning_node = ExecutionNode(
                            id=node_id,
                            label=f"Step {step_num}: {block_data['type'].replace('_', ' ').title()}",
                            type=block_data['type'],
                            position={"x": 250, "y": 450 + (step_num * 100)},
                            data=block_data['metadata']
                        )
                        reasoning_nodes.append(reasoning_node)
                        graph.nodes.append(reasoning_node)
                        
                        # Add edge from previous node
                        if len(reasoning_nodes) == 1:
                            # First reasoning step connects to llm_process
                            edge = ExecutionEdge(
                                id="e_llm_to_step1",
                                source="llm_process",
                                target=node_id,
                                label="generates"
                            )
                        else:
                            # Subsequent steps connect to previous step
                            prev_node = reasoning_nodes[-2]
                            edge = ExecutionEdge(
                                id=f"e_step_{step_num-1}_to_{step_num}",
                                source=prev_node.id,
                                target=node_id,
                                label="leads to"
                            )
                        graph.edges.append(edge)
                        
                        # Emit semantic block
                        block = ContextBlock(
                            content=block_data['content'],
                            type=block_data['type'],
                            metadata=block_data['metadata'],
                            execution_graph_id=graph.id,
                            node_id=node_id,
                            connections=[edge.target] if len(reasoning_nodes) > 1 else [],
                            state=NodeState.FINISHED
                        )
                        yield block.model_dump_json() + "\\n"
                        
                        # Emit graph update
                        yield ContextBlock(
                            content="",
                            type="graph_update",
                            metadata={"graph": graph.model_dump()},
                            execution_graph_id=graph.id
                        ).model_dump_json() + "\\n"
            
            # Flush any remaining content in buffer
            final_block_data = chunker.flush()
            if final_block_data:
                step_num = final_block_data['metadata']['step']
                node_id = f"reasoning_step_{step_num}"
                
                reasoning_node = ExecutionNode(
                    id=node_id,
                    label=f"Step {step_num}: {final_block_data['type'].replace('_', ' ').title()}",
                    type=final_block_data['type'],
                    position={"x": 250, "y": 450 + (step_num * 100)},
                    data=final_block_data['metadata']
                )
                reasoning_nodes.append(reasoning_node)
                graph.nodes.append(reasoning_node)
                
                if len(reasoning_nodes) > 1:
                    prev_node = reasoning_nodes[-2]
                    edge = ExecutionEdge(
                        id=f"e_step_{step_num-1}_to_{step_num}",
                        source=prev_node.id,
                        target=node_id,
                        label="leads to"
                    )
                    graph.edges.append(edge)
                
                block = ContextBlock(
                    content=final_block_data['content'],
                    type=final_block_data['type'],
                    metadata=final_block_data['metadata'],
                    execution_graph_id=graph.id,
                    node_id=node_id,
                    state=NodeState.FINISHED
                )
                yield block.model_dump_json() + "\\n"
        
        except Exception as e:
            logger.error(f"[{session_id}] Streaming error: {e}")
            yield ContextBlock(content=f"Error: {e}", type="error").model_dump_json() + "\\n"
        
        # Final completion block
        yield ContextBlock(
            content="", 
            type="completion",
            execution_graph_id=graph.id,
            node_id="llm_process",
            state=NodeState.FINISHED
        ).model_dump_json() + "\\n"

        # 5. Save AI Response
        try:
            await self.mongo_storage.save_message(session_id, "assistant", full_response_text)
        except Exception:
            pass
