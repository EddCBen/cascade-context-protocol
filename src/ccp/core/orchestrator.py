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

    def get_system_prompt(self) -> str:
        """
        Reads the standard system prompt.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(base_dir, "prompts")
        system_file = os.path.join(prompts_dir, "system.txt")

        try:
            with open(system_file, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "SYSTEM: You are a helpful assistant."

    async def retrieve_tools(self, query: str, threshold: float = 0.75) -> List[Any]:
        """
        Neuro-Symbolic Tool Retrieval with Threshold:
        1. Embed Query.
        2. Normalize (Neural).
        3. Search (Qdrant).
        4. Filter by Threshold.
        5. Return Function Objects.
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
                limit=3,  # Top-3 tools
                score_threshold=threshold # Filter by similarity
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
                     logger.info(f"[Orchestrator] Vector Hit: {tool_name} (Score: {hit.score:.4f})")
                     found_tools.append(all_tools[tool_name])
        
    async def retrieve_tools(self, query: str, threshold: float = 0.60) -> List[Any]:
        """
        Neuro-Symbolic Tool Retrieval with Regex Reinforcement:
        1. Embed Query.
        2. Normalize (Neural).
        3. Search (Qdrant) with Lower Threshold.
        4. Regex Pattern Matching (Reinforcement).
        5. Return Union of Tools.
        """
        import re
        
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
                limit=3,  # Top-3 tools
                score_threshold=threshold # Filter by similarity
            )
        except Exception as e:
            logger.error(f"Qdrant Search Error: {e}")
            return []

        # 4. Map back to functions
        found_tools = []
        if self.llm_service and hasattr(self.llm_service, 'registry'):
             all_tools = self.llm_service.registry.tools # name -> func
             
             # Vector Hits
             for hit in search_result:
                 tool_name = hit.payload.get("name")
                 if tool_name and tool_name in all_tools:
                     logger.info(f"[Orchestrator] Vector Hit: {tool_name} (Score: {hit.score:.4f})")
                     found_tools.append(all_tools[tool_name])
             
             # 5. Regex Reinforcement
             # Explicit intent detection to guarantee tool selection for specific commands
             regex_patterns = {
                 "search_web_advanced": [
                     r"(?i)\b(search|find|look\sup)\b.*\b(web|online|google|internet)\b",
                     r"(?i)\b(search|query)\b.*\b(web|internet)\b"
                 ],
                 "search_news": [
                     r"(?i)\b(news|headline|latest)\b",
                     r"(?i)\b(what|match)\b.*\b(happened|occurring)\b"
                 ],
                 "cross_check_fact": [
                     r"(?i)\b(verify|fact[\s-]?check|true|false|validate)\b"
                 ],
                 "ccp_search": [
                     r"(?i)\b(search|find|query)\b" # Broad catch-all for search if not specific to web
                 ]
             }
             
             for tool_key, patterns in regex_patterns.items():
                 # Only check if tool exists
                 if tool_key in all_tools:
                     for pattern in patterns:
                         if re.search(pattern, query):
                             tool_func = all_tools[tool_key]
                             if tool_func not in found_tools:
                                 logger.info(f"[Orchestrator] Regex Reinforcement: {tool_key} (Matched: '{pattern}')")
                                 found_tools.append(tool_func)
                             break # Matched this tool, move to next tool

        return found_tools

    async def select_tool_and_args(self, candidate_tools: List[Any], context_text: str) -> tuple[Optional[str], dict]:
        """
        Presented with a list of candidate tools (from Search), the LLM selects the 
        most appropriate one (or None) and generates its arguments.
        """
        import inspect
        import json

        tools_desc = []
        for tool in candidate_tools:
            tool_name = tool.__name__
            docstring = inspect.getdoc(tool)
            sig_obj = inspect.signature(tool)
            
            params = []
            for name, param in sig_obj.parameters.items():
                type_name = str(param.annotation).replace("typing.", "") if param.annotation != inspect.Parameter.empty else "Any"
                default_val = f"(default: {param.default})" if param.default != inspect.Parameter.empty else "(REQUIRED)"
                params.append(f"- {name}: {type_name} {default_val}")
            
            tool_desc = f"""
--- TOOL OPTION: {tool_name} ---
SIGNATURE: {tool_name}({', '.join([k for k in sig_obj.parameters.keys()])})
PARAMS:
{chr(10).join(params)}
DOCS: {docstring}
"""
            tools_desc.append(tool_desc)

        all_tools_str = "\n".join(tools_desc)

        prompt = f"""SYSTEM: You are an autonomous Tool Selector & Argument Generator.
Task: Analyze the Context Segment and the available Candidate Tools.
1. Select the SINGLE best tool to solve the problem in the context. If none are relevant, return "None".
2. If a tool is selected, extract precise arguments from the context.

CANDIDATE TOOLS:
{all_tools_str}

CONTEXT SEGMENT:
"{context_text}"

INSTRUCTIONS:
- If a relevant tool exists, output JSON with "tool_name" and "arguments".
- If NO tool is relevant, output JSON using "tool_name": "None".
- Do not hallucinate tools not in the list.

Example Output:
{{
  "tool_name": "ccp_search",
  "arguments": {{ "query": "latest AI trends", "sources": "web" }}
}}

JSON OUTPUT:"""

        try:
            # Call LLM
            resp_stream = self.llm_service.generate_content_stream(prompt)
            full_resp = ""
            for token in resp_stream:
                full_resp += token
            
            clean_json = full_resp.strip()
            if clean_json.startswith("```json"): clean_json = clean_json[7:]
            if clean_json.startswith("```"): clean_json = clean_json[3:]
            if clean_json.endswith("```"): clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            if not clean_json: return None, {}
            
            data = json.loads(clean_json)
            tool_name = data.get("tool_name")
            args = data.get("arguments", {})
            
            logger.info(f"[Orchestrator] Selected Tool: {tool_name}, Args: {args}")
            return tool_name, args

        except Exception as e:
            logger.warning(f"[Orchestrator] Tool Selection Failed: {e}")
            return "None", {}

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

        # 2. Input Segmentation (Context-Aware)
        from src.ccp.core.semantic_chunker import SemanticChunker
        from src.ccp.core.settings import settings
        
        # Determine if segmentation is needed based on Context Window
        # We leave ~25% buffer for system prompts and reasoning
        context_limit = int(settings.context_window_size * 0.75)
        input_tokens = len(message.split()) # Rough estimate
        
        chunker = SemanticChunker(chunk_mode="semantic", max_chunk_size=context_limit)
        
        if input_tokens < context_limit:
            input_blocks = [message]
            logger.info(f"[{session_id}] Message ({input_tokens} tokens) fits in context ({context_limit}). Skipping segmentation.")
        else:
            input_blocks = chunker.chunk_text(message)
            logger.info(f"[{session_id}] Message ({input_tokens} tokens) exceeds context ({context_limit}). Segmented into {len(input_blocks)} blocks.")

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
                # position={"x": 250, "y": 100 + (i * 300)}, # REMOVED COORDS
                data={"content": input_block_text[:50]+"..."}
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

            # --- B. Context/Tool Retrieval (Per Block) ---
            # 1. Retrieve Optimal Tool (Neural Decision with Threshold)
            retrieved_tools = await self.retrieve_tools(input_block_text, threshold=0.75)
            tool_names = [t.__name__ for t in retrieved_tools]
            
            retrieved_context = ""
            active_tool_name = "None"
            tool_args_used = {}
            
            # 2. Execute Tool (if any)
            # 2. Execute Tool (if any)
            if retrieved_tools:
                # Dynamic Selection: Let LLM choose from retrieved tools + Generate Args
                try:
                    active_tool_name, tool_args_used = await self.select_tool_and_args(retrieved_tools, input_block_text)
                    
                    if active_tool_name and active_tool_name != "None":
                        # Find function object
                        active_tool = next((t for t in retrieved_tools if t.__name__ == active_tool_name), None)
                        
                        if active_tool:
                            logger.info(f"[{session_id}] Executing Selected Tool: {active_tool_name} with args: {tool_args_used}")
                            
                            # Execute
                            if tool_args_used:
                                tool_result = active_tool(**tool_args_used)
                            else:
                                # Fallback: simple call
                                import inspect
                                sig = inspect.signature(active_tool)
                                if "query" in sig.parameters:
                                    tool_result = active_tool(query=input_block_text)
                                else:
                                    tool_result = active_tool(input_block_text)
                            
                            retrieved_context = str(tool_result)
                        else:
                             logger.warning(f"[{session_id}] Selected tool '{active_tool_name}' not found in candidates.")
                    else:
                        logger.info(f"[{session_id}] LLM decided NOT to use any tool.")
                        
                except Exception as e:
                    logger.error(f"Tool Selection/Execution Failed: {e}")
                    retrieved_context = f"Error executing tool: {e}"

            # Tool Execution Node Visualization
            tool_node_id = f"tool_exec_{i+1}"
            tool_node = ExecutionNode(
                id=tool_node_id,
                label=f"Tool: {active_tool_name}",
                type="tool_execution", 
                data={
                    "tool_name": active_tool_name,
                    "input": input_block_text[:100],
                    "args": tool_args_used,
                    "output_preview": retrieved_context[:200] if retrieved_context else "No Output/Context",
                    "output_preview": retrieved_context[:100] if retrieved_context else "No Output/Context",
                    "tools_available": tool_names,
                    "context_found": bool(retrieved_context)
                }
            )
            graph.nodes.append(tool_node)
            graph.edges.append(ExecutionEdge(
                id=f"e_{input_node_id}_to_{tool_node_id}",
                source=input_node_id,
                target=tool_node_id,
                label="uses_tool"
            ))

            # Emit Graph Update for Tool Node (Crucial for visualization)
            yield ContextBlock(
                content="", type="graph_update", metadata={"graph": graph.model_dump()}, execution_graph_id=graph.id
            ).model_dump_json() + "\n"
            
            # --- D. Dynamic Prompt Construction ---
            # Corrected: Use System Prompt (Assistant Persona) instead of Segmentation Prompt
            system_instruction = self.get_system_prompt()
            full_prompt = f"{system_instruction}\n\n"
            if retrieved_context:
                full_prompt += f"TOOL OUTPUT ({active_tool_name}):\n{retrieved_context}\n\n"
            
            full_prompt += f"USER INPUT:\n{input_block_text}\n"

            # --- E. Stream LLM Response ---
            stream = self.llm_service.generate_content_stream(full_prompt)
            
            # Output Chunker (Reset for each block response)
            output_chunker = SemanticChunker(chunk_mode="semantic", max_chunk_size=512)
            block_response_text = ""
            
            try:
                reasoning_step_count = 0
                in_think_block = False
                
                for token in stream:
                    if not token:
                        continue
                    
                    # --- THINK TAG FILTERING ---
                    # Simple state machine to suppress <think>...</think> content
                    token_to_process = token
                    
                    if "<think>" in token:
                        in_think_block = True
                        # If token has content before <think>, we keep it? 
                        # Assuming token is usually granular, but if it contains partial text:
                        # "Hello <think>" -> process "Hello "
                        parts = token.split("<think>")
                        token_to_process = parts[0]
                    
                    if "</think>" in token:
                        in_think_block = False
                        # If content after, process it
                        parts = token.split("</think>")
                        if len(parts) > 1:
                             token_to_process = parts[1]
                        else:
                             token_to_process = ""

                    if in_think_block:
                        continue
                    
                    if not token_to_process:
                        continue
                    
                    # --- END FILTERING ---

                    block_response_text += token_to_process
                    
                    # Chunk Output
                    out_block_data = output_chunker.add_token(token_to_process)
                    
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
                        source_id = tool_node_id if reasoning_step_count == 1 else f"output_{i+1}_step_{reasoning_step_count-1}"
                        
                        graph.edges.append(ExecutionEdge(
                            id=f"e_{source_id}_to_{out_node_id}",
                            source=source_id,
                            target=out_node_id,
                            label="generated"
                        ))
                        
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
                        self._persist_graph(graph)
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
                    
                    source_id = tool_node_id if reasoning_step_count == 1 else f"output_{i+1}_step_{reasoning_step_count-1}"
                    graph.edges.append(ExecutionEdge(id=f"e_{source_id}_to_{out_node_id}", source=source_id, target=out_node_id, label="generated"))
                    
                    previous_node_id = out_node_id
                    
                    self._persist_graph(graph)
                    yield ContextBlock(
                        content=final_block['content'], type=final_block['type'], metadata=final_block['metadata'], execution_graph_id=graph.id, node_id=out_node_id, state=NodeState.FINISHED
                    ).model_dump_json() + "\n"
                
            except Exception as e:
                logger.error(f"[{session_id}] Streaming error in block {i}: {e}")
                yield ContextBlock(content=f"Error processing block {i}: {e}", type="error").model_dump_json() + "\n"

        # 4. Save AI Response (Full aggregated)
        logger.info(f"[{session_id}] Finished processing all input blocks.")
        
        # Final completion block
        yield ContextBlock(
            content="", 
            type="completion",
            execution_graph_id=graph.id,
            node_id="end",
            state=NodeState.FINISHED
        ).model_dump_json() + "\n"

    def _persist_graph(self, graph: ExecutionGraph):
        """Saves current graph state to src/web/execution_graph.json"""
        import os
        import json
        try:
            web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
            os.makedirs(web_dir, exist_ok=True)
            path = os.path.join(web_dir, "execution_graph.json")
            with open(path, "w") as f:
                f.write(graph.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to persist graph: {e}")
