import inspect
import importlib
import pkgutil
import hashlib
import logging
from typing import List, Dict, Any, Callable
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pymongo import MongoClient

import src.ccp.functions.library
from src.ccp.functions.utils import expose_as_ccp_tool
from src.ccp.core.settings import settings

logger = logging.getLogger(__name__)

class FunctionRegistry:
    """
    Manages the synchronization of function tools between code, persistent metadata (Mongo),
    and semantic vector store (Qdrant).
    """
    def __init__(self, use_ml_tools: bool = True):
        self.tools: Dict[str, Callable] = {}
        self.use_ml_tools = use_ml_tools
        self._discover_tools()

    def _discover_tools(self):
        """
        Dynamically scans for tools decorated with @expose_as_ccp_tool.
        """
        self.tools = {}
        
        # 1. Utils (Legacy/Core)
        try:
            from src.ccp.functions import utils
            self._scan_module(utils)
        except Exception as e:
            logger.error(f"Error scanning utils: {e}")

        # 2. Library (Data, ML, etc.)
        for _, name, _ in pkgutil.iter_modules(src.ccp.functions.library.__path__):
            try:
                module = importlib.import_module(f"src.ccp.functions.library.{name}")
                self._scan_module(module)
            except Exception as e:
                logger.error(f"Error scanning library module {name}: {e}")

        logger.info(f"[REGISTRY] Discovered {len(self.tools)} tools: {list(self.tools.keys())}")

    def _scan_module(self, module):
        """Helper to scan a single module for decorated functions."""
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                # Check for our custom attribute set by the decorator
                # Note: Our simple decorator just returns the func, but usually adds an attribute.
                # However, our current `expose_as_ccp_tool` decorator in `utils.py` does NOTHING but return func.
                # To efficiently detect them without modifying `utils.py` (if restricted), we'd need a marker.
                # But the prompt implies "Scan all functions decorated with...".
                # Standard python decorators don't leave a trace unless they wrap or set attr.
                # If `expose_as_ccp_tool` is just a pass-through identity function, we technically can't detect it easily
                # UNLESS we rely on the function module or name conventions.
                # OR, we should modify `utils.py` to add a marker.
                # Given instructions: "Code Level: Scan all functions decorated..."
                # I WILL modify `utils.py` to add a marker if I haven't already. (Checked: it doesn't).
                # But I am implementing Registry now. I can check for `_is_ccp_tool` attribute.
                if getattr(obj, "_is_ccp_tool", False):
                    self.tools[name] = obj

    def get_tool_list(self) -> List[Callable]:
        return list(self.tools.values())

    def synchronize_stores(self, mongo_client: MongoClient, qdrant_client: QdrantClient, list_embedding_func: Callable[[str], List[float]]):
        """
        Syncs code state with persistence.
        """
        db = mongo_client["ccp_storage"]
        collection_meta = db["function_metadata"]
        
        # Ensure Qdrant Collection exists with correct dimensions
        try:
             from qdrant_client.models import VectorParams, Distance
             
             # Check if collection exists
             collections = qdrant_client.get_collections()
             exists = any(c.name == "function_registry" for c in collections.collections)
             
             if exists:
                 # Verify dimension matches (with error handling for Pydantic issues)
                 try:
                     collection_info = qdrant_client.get_collection("function_registry")
                     current_dim = collection_info.config.params.vectors.size
                     
                     if current_dim != settings.embedding_dim:
                         logger.warning(f"Dimension mismatch detected! Collection has {current_dim}d, but settings require {settings.embedding_dim}d. Recreating collection...")
                         qdrant_client.delete_collection("function_registry")
                         exists = False
                 except Exception as dim_check_error:
                     # If we can't check dimensions (e.g., Pydantic validation error), 
                     # assume mismatch and recreate to be safe
                     logger.warning(f"Could not verify collection dimensions ({dim_check_error}). Recreating collection to ensure {settings.embedding_dim}d...")
                     try:
                         qdrant_client.delete_collection("function_registry")
                     except:
                         pass  # Collection might not exist
                     exists = False
             
             if not exists:
                 logger.info(f"Creating 'function_registry' collection in Qdrant with {settings.embedding_dim} dimensions.")
                 qdrant_client.create_collection(
                     collection_name="function_registry",
                     vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
                 )
        except Exception as e:
             logger.error(f"Error checking/creating Qdrant collection: {e}")
             # We might continue if it already exists, but log it.

        for name, func in self.tools.items():
            docstring = inspect.getdoc(func) or ""
            source = inspect.getsource(func)
            
            # Compute Hash
            content_hash = hashlib.sha256((source + docstring).encode("utf-8")).hexdigest()
            
            # Check Mongo
            stored_meta = collection_meta.find_one({"name": name})
            
            needs_update = False
            if not stored_meta:
                logger.info(f"[REGISTRY] New tool detected: {name}")
                needs_update = True
            elif stored_meta.get("hash") != content_hash:
                logger.info(f"[REGISTRY] Tool modified: {name}")
                needs_update = True
                
            if needs_update:
                # 1. Update Mongo
                schema = str(inspect.signature(func)) # Simplified schema for now
                collection_meta.update_one(
                    {"name": name},
                    {"$set": {
                        "name": name,
                        "hash": content_hash,
                        "docstring": docstring,
                        "schema": schema,
                        "updated_at": "now" # TODO real timestamp
                    }},
                    upsert=True
                )
                
                # 2. Update Qdrant
                # We need an embedding for the docstring/function description
                description_for_embedding = f"{name}: {docstring}"
                embedding = list_embedding_func(description_for_embedding)
                
                qdrant_client.upsert(
                    collection_name="function_registry",
                    points=[
                        PointStruct(
                            id=hashlib.md5(name.encode()).hexdigest(), # unstable ID based on name hash? UUID preferred but deterministic needed
                            vector=embedding,
                            payload={"name": name, "docstring": docstring}
                        )
                    ]
                )
                logger.info(f"[REGISTRY] Synced {name} to persistent stores.")
