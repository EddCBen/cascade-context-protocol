import logging
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
import httpx
from src.ccp.core.settings import settings

load_dotenv()

load_dotenv()

from app.routes import router as api_router, llm_service

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Cascade Context Protocol API")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

app.include_router(api_router)

# --- Mount Static Files (Dashboard) ---
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/web")
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

@app.get("/dashboard")
async def dashboard_redirect():
    return RedirectResponse(url="/web/dashboard.html")

@app.on_event("startup")
async def startup_event():
    logger.info(">>> Kicking off Startup Hooks <<<")
    
    # 1. Verify Function Registry Population
    if hasattr(llm_service, 'registry'):
        logger.info("[STARTUP] ========================================")
        logger.info("[STARTUP] FUNCTION REGISTRY VERIFICATION")
        logger.info("[STARTUP] ========================================")
        
        registry = llm_service.registry
        tool_count = len(registry.tools)
        
        if tool_count > 0:
            logger.info(f"[STARTUP] ✅ Function Registry POPULATED: {tool_count} tools discovered")
            logger.info(f"[STARTUP] 📋 Registered Tools:")
            for i, tool_name in enumerate(sorted(registry.tools.keys()), 1):
                logger.info(f"[STARTUP]    {i}. {tool_name}")
            
            # Verify Qdrant sync
            try:
                from qdrant_client import QdrantClient
                from pydantic import ValidationError
                qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
                
                # Check if function_registry collection exists
                try:
                    collections = qdrant.get_collections()
                    registry_exists = any(c.name == "function_registry" for c in collections.collections)
                except Exception as e:
                    if "validation error" in str(e).lower():
                        logger.warning(f"[STARTUP] ⚠️ Qdrant Validation Warning (get_collections) - ignoring extra fields.")
                        registry_exists = True 
                    else:
                        raise e
                
                if registry_exists:
                    try:
                        collection_info = qdrant.get_collection("function_registry")
                        point_count = collection_info.points_count
                        logger.info(f"[STARTUP] ✅ Qdrant Sync: {point_count} functions in vector store")
                        
                        if point_count != tool_count:
                            logger.warning(f"[STARTUP] ⚠️  Mismatch: {tool_count} tools in code, {point_count} in Qdrant")
                            logger.info(f"[STARTUP] 🔄 Re-syncing registry...")
                            llm_service._synchronize_registry()
                    except Exception as e:
                        if "validation error" in str(e).lower():
                            logger.warning(f"[STARTUP] ⚠️ Qdrant Validation Warning (get_collection) - skipping count check.")
                        else:
                            raise e
                else:
                    logger.warning(f"[STARTUP] ⚠️  Qdrant collection 'function_registry' not found")
                    logger.info(f"[STARTUP] 🔄 Creating and syncing registry...")
                    llm_service._synchronize_registry()
                    
            except Exception as e:
                logger.error(f"[STARTUP] ❌ Qdrant verification failed: {e}")
        else:
            logger.error(f"[STARTUP] ❌ Function Registry is EMPTY!")
            logger.error(f"[STARTUP] 🔍 Check that functions are decorated with @expose_as_ccp_tool")
            logger.error(f"[STARTUP] 🔍 Check that src/ccp/functions/library modules are importable")
        
        logger.info("[STARTUP] ========================================")
    else:
        logger.error("[STARTUP] ❌ LLM Service has no registry attribute!")

    # 2. Ping Local LLM
    if settings.llm_provider == "local":
        base_url = settings.local_llm_base_url
        logger.info(f"[STARTUP] Checking Local LLM Connectivity at {base_url}...")
        try:
            async with httpx.AsyncClient() as client:
                url = f"{base_url}/models"
                resp = await client.get(url, timeout=5.0)
                if resp.status_code == 200:
                    logger.info(f"[STARTUP] ✅ Local LLM Connected! Models available: {len(resp.json().get('data', []))}")
                else:
                    logger.warning(f"[STARTUP] Local LLM responded with {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"[STARTUP] Failed to connect to Local LLM: {e}")
            logger.warning(">>> Ensure 'local-models' container is running <<<")
    
    logger.info("[STARTUP] ✅ Startup Complete")
@app.get("/")
async def root():
    return {"message": "CCP API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
