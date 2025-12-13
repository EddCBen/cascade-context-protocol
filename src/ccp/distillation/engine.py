"""
Distillation Engine - Core orchestrator for autonomous domain learning.
Coordinates scraping, dataset creation, training, and domain registration.
"""
import logging
import asyncio
from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient
from src.ccp.distillation.models import (
    DomainProfile,
    TrainingDataset,
    TaskType,
    DomainDistillationRequest
)
from src.ccp.distillation.scraper import DomainScraper
from src.ccp.distillation.dataset_creator import DatasetCreator
from src.ccp.distillation.domain_manager import DomainManager
from src.ccp.core.embedding_service import EmbeddingService
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.ccp.core.settings import settings
import uuid

logger = logging.getLogger(__name__)


class DistillationEngine:
    """
    Autonomous domain learning engine.
    Orchestrates the full pipeline from scraping to neural training.
    """
    
    def __init__(self, mongo_client: MongoClient, llm_service=None):
        """
        Initialize distillation engine.
        
        Args:
            mongo_client: MongoDB client
            llm_service: Optional LLM service for domain decomposition
        """
        self.mongo_client = mongo_client
        self.llm_service = llm_service
        
        # Initialize scraper with enhanced config
        from src.ccp.distillation.scraper import ScraperConfig, DomainScraper
        scraper_config = ScraperConfig(
            min_samples=200,
            max_samples=500,
            parallel=True,
            max_workers=5,
            max_retries=3
        )
        self.scraper = DomainScraper(scraper_config)
        self.dataset_creator = DatasetCreator(llm_service)
        self.domain_manager = DomainManager(mongo_client)
        
        self.db = mongo_client["ccp_storage"]
    
    async def distill_domain(self, request: DomainDistillationRequest) -> DomainProfile:
        """
        Full distillation pipeline for a domain.
        
        Steps:
        1. Decompose domain into subdomains
        2. Scrape domain-specific data
        3. Create training datasets
        4. Store datasets in MongoDB
        5. Train neural router (placeholder for now)
        6. Register domain
        
        Args:
            request: Domain distillation request
        
        Returns:
            DomainProfile with mastery metrics
        """
        logger.info(f"[DistillationEngine] Starting distillation for domain: {request.domain}")
        
        # 1. Decompose domain
        subdomains = await self.decompose_domain(request.domain, request.task_description)
        logger.info(f"[DistillationEngine] Decomposed into {len(subdomains)} subdomains: {subdomains}")
        
        # 2. Scrape data
        scraped_data = await self.scraper.scrape_domain(
            request.domain,
            subdomains,
            max_samples=request.max_samples
        )
        logger.info(f"[DistillationEngine] Scraped {len(scraped_data)} samples")
        
        # 3. Create datasets for each task type
        datasets = []
        for task_type in request.task_types:
            dataset = self.dataset_creator.create_dataset(
                scraped_data,
                request.domain,
                task_type
            )
            datasets.append(dataset)
        
        # 4. Store datasets in MongoDB
        collection_name = f"{request.domain}_training_data"
        await self.store_datasets(datasets, collection_name)

        # 4.1 Index Scraped Data into Qdrant for Hybrid Search
        logger.info(f"[DistillationEngine] Indexing {len(scraped_data)} scraped segments into Qdrant...")
        try:
            # Initialize services
            embedding_service = EmbeddingService()
            qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            
            points = []
            for item in scraped_data:
                # Generate embedding
                vector = embedding_service.embed(item.content)
                
                # Deterministic ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item.url + item.content[:100]))
                
                payload = {
                    "domain": request.domain,
                    "url": item.url,
                    "title": item.title,
                    "content": item.content,  # Store full content for retrieval
                    "source": item.source,
                    "type": "raw_scraped_data"
                }
                
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            
            if points:
                # Upsert to 'domain_contexts' collection
                qdrant_client.upsert(
                    collection_name="domain_contexts",
                    points=points
                )
                logger.info(f"[DistillationEngine] ✅ Indexed {len(points)} segments in Qdrant (domain_contexts)")
                
        except Exception as e:
            logger.error(f"[DistillationEngine] Failed to index scraped data in Qdrant: {e}")
        
        # 4.5 [NEW] Extract and Index Structured Knowledge
        logger.info(f"[DistillationEngine] Extracting structured knowledge for {request.domain}...")
        try:
            await self.extract_and_index_knowledge(request.domain, scraped_data)
        except Exception as e:
            logger.error(f"[DistillationEngine] Failed to extract structured knowledge: {e}")

        # 5. Create semantic clusters and train neural networks
        logger.info(f"[DistillationEngine] Neural training skipped (Architecture 2.0). Using Hybrid Search instead.")
        
        # Placeholder for compatibility
        mastery_score = 1.0
        weights_path = "weights/none"
        
        # 6. Create and register domain profile
        profile = DomainProfile(
            name=request.domain,
            subdomains=subdomains,
            task_types=request.task_types,
            dataset_collection=collection_name,
            sample_count=sum(len(d.samples) for d in datasets),
            mastery_score=mastery_score,
            neural_weights_path=weights_path,
            enabled=True,
            last_trained=datetime.utcnow()
        )
        
        await self.domain_manager.register_domain(profile)
        
        logger.info(f"[DistillationEngine] Completed distillation for {request.domain}")
        return profile
    
    async def extract_and_index_knowledge(self, domain: str, scraped_data: List):
        """
        Extract structured knowledge (concepts, definitions) from scraped data and index in Qdrant.
        This provides holistic domain coverage beyond raw chunks.
        """
        if not self.llm_service:
            logger.warning("[DistillationEngine] No LLM service available for knowledge extraction")
            return

        from src.ccp.core.embedding_service import EmbeddingService
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        from src.ccp.core.settings import settings
        import hashlib
        import uuid
        
        embedding_service = EmbeddingService()
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        # Prepare context from scraped data (titles and snippets)
        context_text = "\n".join([f"- {d.title}: {d.content[:200]}" for d in scraped_data[:50]])
        
        prompt = f"""Extract the core structured knowledge for the domain "{domain}".
Based on the following content samples:
{context_text[:3000]}

Identify 10-15 key concepts, definitions, or fundamental facts.
Return ONLY a valid JSON list of objects.
Format:
[
  {{
    "concept": "Name of concept",
    "definition": "Clear, concise definition",
    "category": "Core Concept/Tool/Methodology/History"
  }}
]
"""
        try:
            response = self.llm_service.generate_content(prompt, max_tokens=1500, temperature=0.2)
            # Basic cleanup for JSON parsing
            import json
            import re
            
            # Find JSON/list structure
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                concepts = json.loads(json_str)
                
                logger.info(f"[DistillationEngine] Extracted {len(concepts)} structured concepts for {domain}")
                
                points = []
                for concept in concepts:
                    # Create rich text representation for embedding
                    text_rep = f"Domain: {domain}\nConcept: {concept['concept']}\nDefinition: {concept['definition']}\nCategory: {concept['category']}"
                    
                    # Generate embedding
                    vector = embedding_service.embed(text_rep)
                    
                    # Generate deterministic UUID
                    concept_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{domain}_{concept['concept']}"))
                    
                    points.append(PointStruct(
                        id=concept_id,
                        vector=vector,
                        payload={
                            "domain": domain,
                            "type": "structured_knowledge",  # Special tag
                            "concept": concept['concept'],
                            "definition": concept['definition'],
                            "category": concept['category'],
                            "content_preview": text_rep,
                            "source": "llm_synthesis"
                        }
                    ))
                
                if points:
                    qdrant_client.upsert(
                        collection_name="domain_contexts",  # Store in same collection but distinguishable by type
                        points=points
                    )
                    logger.info(f"[DistillationEngine] ✅ Indexed {len(points)} structured knowledge concepts in Qdrant")
                    
        except Exception as e:
            logger.error(f"[DistillationEngine] Error in knowledge extraction: {e}")
    
    async def decompose_domain(self, domain: str, task_description: Optional[str]) -> List[str]:
        """
        Decompose domain into subdomains using LLM.
        
        Args:
            domain: Main domain name
            task_description: Optional task description
        
        Returns:
            List of subdomain names
        """
        if self.llm_service:
            # Use LLM to decompose domain
            prompt = f"""List 3-5 key subdomains for "{domain}".

Task: {task_description or 'General domain knowledge'}

Return ONLY a comma-separated list of subdomain names. No explanations, no thinking process.

Format: subdomain1, subdomain2, subdomain3

Example for "machine_learning": supervised_learning, unsupervised_learning, deep_learning, reinforcement_learning

Subdomains for "{domain}":\n"""
            
            try:
                response = self.llm_service.generate_content(prompt, max_tokens=100, temperature=0.3)
                # Clean response - remove any thinking tags or extra text
                response = response.replace('<think>', '').replace('</think>', '')
                response = response.split('\n')[0]  # Take first line only
                subdomains = [s.strip() for s in response.split(',') if s.strip()]
                # Filter out any non-subdomain text
                subdomains = [s for s in subdomains if len(s) < 50 and '_' in s or ' ' in s]
                if subdomains:
                    return subdomains[:5]  # Limit to 5
            except Exception as e:
                logger.warning(f"[DistillationEngine] LLM decomposition failed: {e}")
        
        # Fallback: simple heuristic
        return [f"{domain}_basics", f"{domain}_intermediate", f"{domain}_advanced"]
    
    async def store_datasets(self, datasets: List[TrainingDataset], collection_name: str):
        """
        Store training datasets in MongoDB.
        
        Args:
            datasets: List of training datasets
            collection_name: MongoDB collection name
        """
        try:
            collection = self.db[collection_name]
            
            # Clear existing data
            collection.delete_many({})
            
            # Insert all samples from all datasets
            all_samples = []
            for dataset in datasets:
                for sample in dataset.samples:
                    sample_doc = {
                        "domain": dataset.domain,
                        "task_type": dataset.task_type.value,
                        "input": sample["input"],
                        "label": sample["label"],
                        "metadata": sample.get("metadata", {})
                    }
                    all_samples.append(sample_doc)
            
            if all_samples:
                collection.insert_many(all_samples)
                logger.info(f"[DistillationEngine] Stored {len(all_samples)} samples in {collection_name}")
        
        except Exception as e:
            logger.error(f"[DistillationEngine] Error storing datasets: {e}")
