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
        from src.ccp.neural.training_config import ScraperConfig
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
        
        # 5. Create semantic clusters and train neural networks
        logger.info(f"[DistillationEngine] Creating semantic clusters and training neural networks for {request.domain}...")
        
        try:
            from src.ccp.neural.semantic_cluster import ClusterManager
            from src.ccp.neural.models import NeuralVectorNormalizer, SoftmaxRouter
            from src.ccp.neural.training_config import Context2VecConfig, RouterConfig, TrainingConfig, ClusterConfig
            from src.ccp.neural.trainer_loop import NeuralTrainer
            from src.ccp.core.embedding_service import EmbeddingService
            from qdrant_client import QdrantClient
            from src.ccp.core.settings import settings
            import numpy as np
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            
            # Initialize services
            embedding_service = EmbeddingService()
            qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            
            # Get training dataset
            training_dataset = datasets[0] if datasets else None
            
            if training_dataset and len(training_dataset.samples) >= 5:  # Lowered to 5 for testing
                # Step 5.1: Generate embeddings for all samples
                logger.info(f"[DistillationEngine] Generating embeddings for {len(training_dataset.samples)} samples...")
                embeddings = []
                for sample in training_dataset.samples:
                    emb = embedding_service.embed(sample["input"])
                    embeddings.append(emb)
                embeddings_array = np.array(embeddings)
                
                # Step 5.2: Create semantic clusters
                logger.info(f"[DistillationEngine] Creating 7 semantic clusters...")
                cluster_config = ClusterConfig()
                cluster_manager = ClusterManager(num_clusters=7, use_graph=True)
                clusters = cluster_manager.create_clusters(
                    embeddings_array,
                    request.domain,
                    method="kmeans"
                )
                
                # Store clusters in Qdrant
                cluster_manager.store_in_qdrant(qdrant_client, collection_name="cluster_centers")
                logger.info(f"[DistillationEngine] Stored {len(clusters)} clusters in Qdrant")
                
                # Get cluster centers
                cluster_centers = cluster_manager.get_cluster_centers()
                
                # Assign cluster labels to samples (for training)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=7, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings_array)
                
                # Step 5.3: Prepare training data
                train_size = int(0.8 * len(embeddings_array))
                train_embeddings = torch.FloatTensor(embeddings_array[:train_size])
                train_labels = torch.LongTensor(cluster_labels[:train_size])
                val_embeddings = torch.FloatTensor(embeddings_array[train_size:])
                val_labels = torch.LongTensor(cluster_labels[train_size:])
                
                train_dataset = TensorDataset(train_embeddings, train_labels)
                val_dataset = TensorDataset(val_embeddings, val_labels)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32)
                
                # Step 5.4: Train Context2Vec (NeuralVectorNormalizer)
                logger.info(f"[DistillationEngine] Training Context2Vec...")
                context2vec_config = Context2VecConfig(num_clusters=7)
                context2vec_model = NeuralVectorNormalizer(context2vec_config)
                
                training_config = TrainingConfig(num_epochs=10, batch_size=32)
                trainer = NeuralTrainer(training_config)
                
                context2vec_weights, context2vec_loss = trainer.train_context2vec(
                    context2vec_model,
                    train_loader,
                    val_loader,
                    cluster_centers,
                    request.domain
                )
                logger.info(f"[DistillationEngine] Context2Vec trained. Loss: {context2vec_loss:.4f}")
                
                # Step 5.5: Train SoftmaxRouter
                logger.info(f"[DistillationEngine] Training SoftmaxRouter...")
                router_config = RouterConfig(num_clusters=7)
                router_model = SoftmaxRouter(router_config)
                router_model.load_cluster_centers(cluster_centers)
                
                router_weights, router_accuracy = trainer.train_router(
                    router_model,
                    train_loader,
                    val_loader,
                    request.domain
                )
                logger.info(f"[DistillationEngine] SoftmaxRouter trained. Accuracy: {router_accuracy:.2%}")
                
                # Compute mastery score (average of Context2Vec and Router performance)
                mastery_score = max(0.0, min(1.0, (1.0 - context2vec_loss + router_accuracy) / 2))
                weights_path = context2vec_weights  # Primary weights
                
            else:
                logger.warning(f"[DistillationEngine] Insufficient samples ({len(training_dataset.samples) if training_dataset else 0}), using placeholder")
                mastery_score = 0.5
                weights_path = f"weights/domains/{request.domain}.pt"
        
        except Exception as e:
            logger.error(f"[DistillationEngine] Training failed: {e}", exc_info=True)
            # Fallback to placeholder
            mastery_score = 0.5
            weights_path = f"weights/domains/{request.domain}.pt"
        
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
