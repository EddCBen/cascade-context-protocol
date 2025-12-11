import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pymongo import MongoClient

# Import CCP modules
from src.ccp.core.orchestrator import Orchestrator
from src.ccp.core.llm_service import LLMService
from src.ccp.core.settings import settings
from src.ccp.distillation.engine import DistillationEngine
from src.ccp.distillation.domain_manager import DomainManager
from src.ccp.distillation.models import (
    DomainDistillationRequest,
    DomainMasteryResponse,
    TaskType
)

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(..., description="Unique session identifier for persistence")
    granularity_level: float = 0.5
    task_domain: Optional[str] = None
    domains: Optional[List[str]] = None  # NEW: Domain filtering

class TrainRequest(BaseModel):
    topic: str
    iterations: int = 1

class DomainToggleRequest(BaseModel):
    domain: str
    enabled: bool

# --- Helpers ---
# Initialize services
mongo_client = MongoClient(settings.mongo_uri)
llm_service = LLMService()
distillation_engine = DistillationEngine(mongo_client, llm_service)
domain_manager = DomainManager(mongo_client)

# --- Endpoints ---

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with Streaming, Persistence, Granularity Control, and Domain Filtering.
    Returns a stream of ContextBlock objects (NDJSON).
    """
    domain_id = request.task_domain or "default"
    
    # Logging Neural Weights status
    logger.info(f"[NEURAL STATE] Active Weights Domain: {domain_id}. Session: {request.session_id}")
    
    # Log domain filtering
    if request.domains:
        logger.info(f"[DOMAIN FILTER] Restricting to domains: {request.domains}")
    
    try:
        # Instantiating orchestrator with requested granularity
        orchestrator = Orchestrator(
            granularity=request.granularity_level,
            llm_service=llm_service,
            active_domains=request.domains  # Pass domain filter
        )
        
        # We rely on StreamingResponse to iterate over the async generator
        return StreamingResponse(
            orchestrator.process_message_stream(request.session_id, request.message),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/distill/domain")
async def distill_domain(request: DomainDistillationRequest, background_tasks: BackgroundTasks):
    """
    Trigger autonomous domain learning.
    
    The DistillationEngine will:
    1. Decompose domain into subdomains
    2. Scrape web for domain content
    3. Create training datasets (classification, QA, etc.)
    4. Store in MongoDB
    5. Train neural router (placeholder)
    6. Register domain
    
    Example:
    {
        "domain": "machine_learning",
        "task_description": "Learn ML concepts for classification tasks",
        "max_samples": 1000,
        "task_types": ["classification", "qa"]
    }
    """
    job_id = str(uuid.uuid4())
    logger.info(f"[DISTILLATION] Starting domain distillation: {request.domain} (Job: {job_id})")
    
    try:
        # Run distillation in background
        background_tasks.add_task(
            distillation_engine.distill_domain,
            request
        )
        
        return {
            "domain_name": request.domain,
            "status": "distilling",
            "job_id": job_id,
            "estimated_time": "5-15 minutes",
            "max_samples": request.max_samples
        }
    
    except Exception as e:
        logger.error(f"[DISTILLATION] Error starting distillation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains/mastery")
async def get_domain_mastery():
    """
    Get all domains the chatbot has mastered.
    
    Returns domain profiles with mastery scores, sample counts, and training status.
    
    Example Response:
    {
        "domains": [
            {
                "name": "machine_learning",
                "mastery_score": 0.92,
                "subdomains": ["supervised", "unsupervised", "deep_learning"],
                "sample_count": 1000,
                "enabled": true,
                "last_trained": "2025-12-11T06:00:00Z"
            }
        ],
        "total_domains": 3,
        "active_domains": 2
    }
    """
    try:
        all_domains = await domain_manager.get_all_domains()
        active_count = sum(1 for d in all_domains if d.enabled)
        
        return DomainMasteryResponse(
            domains=all_domains,
            total_domains=len(all_domains),
            active_domains=active_count
        )
    
    except Exception as e:
        logger.error(f"[DOMAINS] Error getting mastery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/domains/toggle")
async def toggle_domain(request: DomainToggleRequest):
    """
    Enable or disable a specific domain.
    
    When disabled, the domain will not be used for routing or retrieval.
    
    Example:
    {
        "domain": "finance",
        "enabled": false
    }
    """
    try:
        success = await domain_manager.toggle_domain(request.domain, request.enabled)
        
        if success:
            return {
                "domain": request.domain,
                "enabled": request.enabled,
                "status": "updated"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Domain '{request.domain}' not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DOMAINS] Error toggling domain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train_task")
async def train_task(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Triggers background distillation task (legacy endpoint).
    Use /distill/domain for new domain learning.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received training request for topic: {request.topic} (Task ID: {task_id})")
    
    # Convert to domain distillation request
    distill_request = DomainDistillationRequest(
        domain=request.topic,
        max_samples=500,
        task_types=[TaskType.CLASSIFICATION, TaskType.QA]
    )
    
    # Add to background tasks
    background_tasks.add_task(distillation_engine.distill_domain, distill_request)
    
    return {
        "task_id": task_id,
        "status": "Training Started",
        "topic": request.topic
    }

