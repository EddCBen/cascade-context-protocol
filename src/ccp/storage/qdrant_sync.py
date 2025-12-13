"""
Qdrant Sync Service - Syncs MongoDB collections to Qdrant with vector representations.
Handles batch embedding generation and cluster assignment.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import uuid

from src.ccp.core.embedding_service import EmbeddingService
from src.ccp.storage.mongo import MongoStorage
from src.ccp.neural.semantic_cluster import ClusterManager

logger = logging.getLogger(__name__)


class QdrantSyncService:
    """
    Syncs MongoDB scraped data to Qdrant with embeddings and cluster assignments.
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        mongo_storage: MongoStorage,
        embedding_service: EmbeddingService
    ):
        self.qdrant = qdrant_client
        self.mongo = mongo_storage
        self.embedder = embedding_service
        self.collection_name = "domain_contexts"
        self.cluster_collection = "cluster_centers"
    
    async def initialize_collections(self):
        """Create Qdrant collections if they don't exist."""
        try:
            self.qdrant.get_collection(self.collection_name)
            logger.info(f"[QdrantSync] Collection {self.collection_name} already exists")
        except:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"[QdrantSync] Created collection {self.collection_name}")
        
        try:
            self.qdrant.get_collection(self.cluster_collection)
            logger.info(f"[QdrantSync] Collection {self.cluster_collection} already exists")
        except:
            self.qdrant.create_collection(
                collection_name=self.cluster_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"[QdrantSync] Created collection {self.cluster_collection}")
    
    async def sync_collection(
        self,
        domain: str,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Sync MongoDB collection to Qdrant for a specific domain.
        
        Args:
            domain: Domain name to sync
            batch_size: Batch size for embedding generation
        
        Returns:
            Statistics: synced, skipped, errors
        """
        logger.info(f"[QdrantSync] Starting sync for domain: {domain}")
        
        # Get all data for domain from MongoDB
        mongo_data = await self.mongo.get_all_domain_data([domain])
        
        if not mongo_data:
            logger.warning(f"[QdrantSync] No data found for domain: {domain}")
            return {"synced": 0, "skipped": 0, "errors": 0}
        
        # Get cluster centers for domain
        cluster_centers = await self.get_cluster_centers(domain)
        
        stats = {"synced": 0, "skipped": 0, "errors": 0}
        
        # Process in batches
        for i in range(0, len(mongo_data), batch_size):
            batch = mongo_data[i:i + batch_size]
            batch_stats = await self._sync_batch(batch, domain, cluster_centers)
            
            stats["synced"] += batch_stats["synced"]
            stats["skipped"] += batch_stats["skipped"]
            stats["errors"] += batch_stats["errors"]
        
        logger.info(f"[QdrantSync] Sync complete for {domain}: {stats}")
        return stats
    
    async def _sync_batch(
        self,
        batch: List[Dict[str, Any]],
        domain: str,
        cluster_centers: Optional[np.ndarray]
    ) -> Dict[str, int]:
        """Sync a batch of documents."""
        stats = {"synced": 0, "skipped": 0, "errors": 0}
        
        # Extract texts
        texts = [doc["content"] for doc in batch]
        
        # Generate embeddings
        try:
            embeddings = self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error(f"[QdrantSync] Error generating embeddings: {e}")
            stats["errors"] = len(batch)
            return stats
        
        # Assign to clusters if available
        cluster_ids = None
        if cluster_centers is not None and len(cluster_centers) > 0:
            cluster_ids = self._assign_to_clusters(embeddings, cluster_centers)
        
        # Create Qdrant points
        points = []
        for idx, (doc, embedding) in enumerate(zip(batch, embeddings)):
            try:
                # Use MongoDB _id as Qdrant point ID
                point_id = str(doc["_id"])
                
                # Prepare payload
                payload = {
                    "domain": domain,
                    "source": doc["source"],
                    "title": doc["title"],
                    "content": doc["content"][:500],  # Truncate for payload
                    "url": doc.get("url", ""),
                    "content_hash": doc.get("content_hash", ""),
                    "scraped_at": doc.get("scraped_at").isoformat() if doc.get("scraped_at") else None
                }
                
                # Add cluster assignment if available
                if cluster_ids is not None:
                    payload["cluster_id"] = int(cluster_ids[idx])
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))
                
            except Exception as e:
                logger.error(f"[QdrantSync] Error creating point: {e}")
                stats["errors"] += 1
                continue
        
        # Upsert to Qdrant
        if points:
            try:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                stats["synced"] = len(points)
            except Exception as e:
                logger.error(f"[QdrantSync] Error upserting to Qdrant: {e}")
                stats["errors"] = len(points)
        
        return stats
    
    def _assign_to_clusters(
        self,
        embeddings: np.ndarray,
        cluster_centers: np.ndarray
    ) -> np.ndarray:
        """
        Assign embeddings to nearest cluster centers.
        
        Args:
            embeddings: Array of shape (n_samples, 384)
            cluster_centers: Array of shape (n_clusters, 384)
        
        Returns:
            Array of cluster IDs (n_samples,)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute similarities
        similarities = cosine_similarity(embeddings, cluster_centers)
        
        # Get nearest cluster for each embedding
        cluster_ids = np.argmax(similarities, axis=1)
        
        return cluster_ids
    
    async def get_cluster_centers(self, domain: str) -> Optional[np.ndarray]:
        """
        Get cluster centers for a domain from Qdrant.
        
        Args:
            domain: Domain name
        
        Returns:
            Array of cluster centers or None if not found
        """
        try:
            # Scroll through cluster centers for domain
            points, _ = self.qdrant.scroll(
                collection_name=self.cluster_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
                ),
                limit=100,
                with_vectors=True
            )
            
            if not points:
                logger.warning(f"[QdrantSync] No cluster centers found for domain: {domain}")
                return None
            
            # Extract vectors
            centers = np.array([point.vector for point in points])
            logger.info(f"[QdrantSync] Loaded {len(centers)} cluster centers for {domain}")
            return centers
            
        except Exception as e:
            logger.error(f"[QdrantSync] Error loading cluster centers: {e}")
            return None
    
    async def batch_embed_and_upsert(
        self,
        texts: List[str],
        domain: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Batch embed texts and upsert to Qdrant.
        
        Args:
            texts: List of text strings
            domain: Domain name
            metadata_list: Optional list of metadata dicts (same length as texts)
            batch_size: Batch size for processing
        
        Returns:
            List of Qdrant point IDs
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        point_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.embedder.embed_batch(batch_texts)
            
            # Create points
            points = []
            for text, embedding, metadata in zip(batch_texts, embeddings, batch_metadata):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                payload = {
                    "domain": domain,
                    "content": text[:500],  # Truncate
                    **metadata
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))
            
            # Upsert
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(f"[QdrantSync] Upserted {len(point_ids)} points for domain: {domain}")
        return point_ids
    
    async def get_cluster_assignments(
        self,
        domain: str
    ) -> Dict[int, List[str]]:
        """
        Get cluster assignments for a domain.
        
        Args:
            domain: Domain name
        
        Returns:
            Dict mapping cluster_id to list of point IDs
        """
        # Scroll through all points for domain
        points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
            ),
            limit=10000,
            with_payload=True
        )
        
        # Group by cluster
        cluster_assignments = {}
        for point in points:
            cluster_id = point.payload.get("cluster_id")
            if cluster_id is not None:
                if cluster_id not in cluster_assignments:
                    cluster_assignments[cluster_id] = []
                cluster_assignments[cluster_id].append(point.id)
        
        logger.info(f"[QdrantSync] Retrieved cluster assignments for {domain}: {len(cluster_assignments)} clusters")
        return cluster_assignments
