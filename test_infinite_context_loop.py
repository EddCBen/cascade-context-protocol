
import asyncio
import sys
import logging
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.ccp.core.orchestrator import Orchestrator

# Configure Logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_script")
# Reduce noise from connection libraries if real connection succeeds
logging.getLogger("httpx").setLevel(logging.WARNING)

async def test_infinite_loop():
    logger.info("Initializing Orchestrator...")
    
    try:
        from src.ccp.core.llm_service import LLMService
        llm_service = LLMService()
        orchestrator = Orchestrator(granularity=0.5, llm_service=llm_service)
        logger.info("✅ Real LLMService initialized")
    except Exception as e:
        logger.warning(f"⚠️ Failed to init real LLMService (Docker might be down): {e}")
        logger.info("Using MOCK LLMService for logic verification...")
        
        class MockLLMService:
            def __init__(self):
                self.provider_type = "mock"
                self.model_name = "mock-model"
                self.context_window_size = 100000
                self.registry = type("Registry", (), {"tools": {}})()
            
            def get_remaining_context(self): return 90000
            def get_embedding(self, text): return [0.1] * 384
            def generate_content_stream(self, prompt):
                yield "This "
                yield "is "
                yield "a "
                yield "mock "
                yield "response "
                yield "for "
                yield "infinite "
                yield "logic "
                yield "verification.\n"
                yield "Therefore, "
                yield "it "
                yield "works."

        orchestrator = Orchestrator(granularity=0.5, llm_service=MockLLMService())
        # We also need to mock Qdrant on orchestrator if real init failed
        if not orchestrator.qdrant:
            orchestrator.qdrant = type("MockQdrant", (), {
                "search": lambda **kwargs: []
            })()

    # Create a MASSIVE input to force segmentation
    # SemanticChunker max is 512 tokens (~2000 chars)
    # We will generate ~10000 chars
    paragraph = "This is a sentence about the history of computing. " * 20 + "\n\n"
    massive_input = paragraph * 10
    
    logger.info(f"Generated Massive Input: {len(massive_input)} chars")
    
    logger.info(">>> STARTING PROCESS MESSAGE STREAM <<<")
    # Using 'test_session'
    async for block in orchestrator.process_message_stream("test_session", massive_input):
        pass
    logger.info(">>> FINISHED STREAM <<<")

if __name__ == "__main__":
    asyncio.run(test_infinite_loop())
