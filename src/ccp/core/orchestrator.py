import os
import asyncio
import logging

from typing import Optional, AsyncGenerator, List, Any
from src.ccp.models.context import ContextBlock, ExecutionGraph, ExecutionNode, ExecutionEdge, NodeState
from src.ccp.storage.mongo import MongoStorage
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
        
        # 2. Use Raw Embedding directly (Standard Vector Search)
        normalized_emb = raw_embedding


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

    async def retrieve_context(self, query: str, limit: int = 3) -> str:
        """
        Retrieve relevant past context blocks from Qdrant.
        """
        if not self.llm_service or not self.qdrant:
            return ""

        try:
            # 1. Embed Query
            embedding = self.llm_service.get_embedding(query)
            if not embedding:
                return ""

            # 2. Search Qdrant (domain_contexts or appropriate collection)
            # Assuming 'domain_contexts' stores general knowledge/history
            search_result = self.qdrant.search(
                collection_name="domain_contexts",
                query_vector=embedding,
                limit=limit
            )
            
            # 3. Format Context
            context_str = ""
            for hit in search_result:
                payload = hit.payload
                if payload and "content" in payload:
                    context_str += f"- {payload['content']}\n"
            
            if context_str:
                return f"RELEVANT PAST CONTEXT:\n{context_str}\n"
            return ""

        except Exception as e:
            logger.error(f"Context Retrieval Error: {e}")
            return ""

    async def process_message_stream(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        """
        Processes the user message with streaming, Neural Retrieval, and persistence.
        Handles Infinite Context via Input Segmentation.
        """
        logger.info(f"[{session_id}] Processing new message: {message[:50]}...")

        # --- GRAPH INITIALIZATION ---
        graph = ExecutionGraph()
        
        # Start Node
        node_start = ExecutionNode(id="start", label="Start", type="input", position={"x": 250, "y": 0})
        graph.nodes.append(node_start)

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

        # 1. Save User Message (Full)
        try:
            await self.mongo_storage.save_message(session_id, "user", message)
        except Exception:
            pass

        # 2. Input Segmentation (The "Infinite Input" Logic)
        from src.ccp.core.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(chunk_mode="semantic", max_chunk_size=512)
        
        # Breakdown user input into manageable blocks
        input_blocks = chunker.chunk_text(message)
        logger.info(f"[{session_id}] Input Segmented into {len(input_blocks)} blocks")

        # Track previous node for linking
        previous_node_id = "start"

        # 3. Sequential Processing Loop
        for i, input_block_text in enumerate(input_blocks):
            logger.info(f"[{session_id}] Processing Input Block {i+1}/{len(input_blocks)}")

            # --- A. Create Input Block Node ---
            input_node_id = f"input_block_{i+1}"
            input_node = ExecutionNode(
                id=input_node_id,
                label=f"Input Chunk {i+1}",
                type="input_segment",
                position={"x": 250, "y": 100 + (i * 300)}, # Offset y
                data={"content": input_block_text}
            )
            graph.nodes.append(input_node)
            graph.edges.append(ExecutionEdge(
                id=f"e_{previous_node_id}_to_{input_node_id}",
                source=previous_node_id,
                target=input_node_id,
                label="sequence"
            ))
            previous_node_id = input_node_id

            # Emit Graph Update for Input Node
            yield ContextBlock(
                content="", type="graph_update", metadata={"graph": graph.model_dump()}, execution_graph_id=graph.id
            ).model_dump_json() + "\n"

            # --- B. Context Retrieval (Per Block) ---
            retrieved_context = await self.retrieve_context(input_block_text)
            
            # --- C. Tool Retrieval (Per Block) ---
            retrieved_tools = await self.retrieve_tools(input_block_text)
            tool_names = [t.__name__ for t in retrieved_tools]
            
            # Retrieval Node Visualization
            retrieval_node_id = f"retrieval_{i+1}"
            retrieval_node = ExecutionNode(
                id=retrieval_node_id,
                label=f"Neural Retrieval {i+1}",
                type="tool",
                position={"x": 450, "y": 100 + (i * 300)}, # Offset x
                data={"tools": tool_names, "context_found": len(retrieved_context) > 0}
            )
            graph.nodes.append(retrieval_node)
            graph.edges.append(ExecutionEdge(
                id=f"e_{input_node_id}_to_{retrieval_node_id}",
                source=input_node_id,
                target=retrieval_node_id,
                label="augments"
            ))
            
            # --- D. Dynamic Prompt Construction ---
            system_instruction = self.get_segmentation_prompt()
            full_prompt = f"{system_instruction}\n\n"
            if retrieved_context:
                full_prompt += f"{retrieved_context}\n"
            
            full_prompt += f"CURRENT INPUT SEGMENT:\n{input_block_text}\n"

            # --- E. Stream LLM Response ---
            stream = self.llm_service.generate_content_stream(full_prompt)
            
            # Output Chunker (Reset for each block response)
            output_chunker = SemanticChunker(chunk_mode="semantic", max_chunk_size=512)
            block_response_text = ""
            
            try:
                reasoning_step_count = 0
                
                for token in stream:
                    if token:
                        block_response_text += token
                        
                        # Chunk Output
                        out_block_data = output_chunker.add_token(token)
                        
                        if out_block_data:
                            reasoning_step_count += 1
                            out_node_id = f"output_{i+1}_step_{reasoning_step_count}"
                            
                            # Create Output Node
                            out_node = ExecutionNode(
                                id=out_node_id,
                                label=f"Reasoning {i+1}.{reasoning_step_count}",
                                type=out_block_data['type'],
                                position={"x": 250, "y": 200 + (i * 300) + (reasoning_step_count * 50)},
                                data=out_block_data['metadata']
                            )
                            graph.nodes.append(out_node)
                            
                            # Link from previous (either input node or previous output step)
                            source_id = retrieval_node_id if reasoning_step_count == 1 else f"output_{i+1}_step_{reasoning_step_count-1}"
                            # Actually, normally connects from retrieval or input. Implementation plan says Input->Output. 
                            # Let's chain them: Input -> Retrieval -> Output1 -> Output2...
                            # But wait, visually Input and Retrieval are parallel "sources". 
                            # Let's link Retrieval -> Output1 to show flow.
                            
                            graph.edges.append(ExecutionEdge(
                                id=f"e_{source_id}_to_{out_node_id}",
                                source=source_id,
                                target=out_node_id,
                                label="generated"
                            ))
                            
                            # Update previous_node_id for the NEXT input block to chain after this output
                            previous_node_id = out_node_id

                            # Emit Content Block
                            yield ContextBlock(
                                content=out_block_data['content'],
                                type=out_block_data['type'],
                                metadata=out_block_data['metadata'],
                                execution_graph_id=graph.id,
                                node_id=out_node_id,
                                state=NodeState.FINISHED
                            ).model_dump_json() + "\n"

                            # Emit Graph Update
                            yield ContextBlock(content="", type="graph_update", metadata={"graph": graph.model_dump()}, execution_graph_id=graph.id).model_dump_json() + "\n"

                # Flush Output Chunker
                final_block = output_chunker.flush()
                if final_block:
                    reasoning_step_count += 1
                    out_node_id = f"output_{i+1}_step_{reasoning_step_count}"
                    
                    out_node = ExecutionNode(
                        id=out_node_id,
                        label=f"Reasoning {i+1}.{reasoning_step_count}",
                        type=final_block['type'],
                        position={"x": 250, "y": 200 + (i * 300) + (reasoning_step_count * 50)},
                        data=final_block['metadata']
                    )
                    graph.nodes.append(out_node)
                    
                    source_id = retrieval_node_id if reasoning_step_count == 1 else f"output_{i+1}_step_{reasoning_step_count-1}"
                    graph.edges.append(ExecutionEdge(id=f"e_{source_id}_to_{out_node_id}", source=source_id, target=out_node_id, label="generated"))
                    
                    previous_node_id = out_node_id

                    yield ContextBlock(
                        content=final_block['content'], type=final_block['type'], metadata=final_block['metadata'], execution_graph_id=graph.id, node_id=out_node_id, state=NodeState.FINISHED
                    ).model_dump_json() + "\n"
                
            except Exception as e:
                logger.error(f"[{session_id}] Streaming error in block {i}: {e}")
                yield ContextBlock(content=f"Error processing block {i}: {e}", type="error").model_dump_json() + "\n"

        # 4. Save AI Response (Full aggregated)
        # Note: We aren't aggregating the full response text variable here to save memory, 
        # but could if needed. For now, Mongo saves are handled by the caller or we assumes steps are saved individually (future).
        # Re-adding aggregation for compatibility with existing save_message
        # (Simply logging completion here)
        logger.info(f"[{session_id}] Finished processing all input blocks.")
        
        # Final completion block
        yield ContextBlock(
            content="", 
            type="completion",
            execution_graph_id=graph.id,
            node_id="end",
            state=NodeState.FINISHED
        ).model_dump_json() + "\n"
