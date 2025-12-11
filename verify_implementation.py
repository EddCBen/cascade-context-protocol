import os
import sys
import logging
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFICATION")

try:
    from src.ccp.core.llm_service import LLMService
    from src.ccp.core.orchestrator import Orchestrator
    from src.ccp.memory.long_term import QdrantMemory
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify():
    print("--- Verification Started ---")
    
    # 1. Check Env
    if "GOOGLE_API_KEY" not in os.environ:
        logger.error("GOOGLE_API_KEY not found. Please set it before running.")
        return

    # 2. Initialize Service
    try:
        llm_service = LLMService()
        logger.info(f"LLMService initialized. Model: {llm_service.model_name}")
    except Exception as e:
        logger.error(f"LLMService init failed: {e}")
        return

    # 3. Initialize Orchestrator
    try:
        orchestrator = Orchestrator(granularity=0.5, llm_service=llm_service)
        logger.info("Orchestrator initialized.")
    except Exception as e:
        logger.error(f"Orchestrator init failed: {e}")
        return

    # 4. Initialize Memory (Will try to connect to localhost)
    try:
        # We expect this might fail if Qdrant isn't running, which is expected during Dev
        memory = QdrantMemory(domain_id="test", llm_service=llm_service)
        logger.info("QdrantMemory initialized.")
        # Try a quick search to verify connection
        res = memory.search_knowledge("test")
        logger.info(f"Qdrant Search Result: {res}")
    except Exception as e:
        logger.warning(f"Qdrant Connection/Search failed (Expected if DB not running): {e}")

    # 5. Run a test flow
    try:
        logger.info("Testing process_message (Calls Gemini)...")
        response = orchestrator.process_message("Test message for verification.")
        logger.info("--- LLM Response Start ---")
        print(response)
        logger.info("--- LLM Response End ---")
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    verify()
