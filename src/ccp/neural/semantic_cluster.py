"""
Semantic clustering system for latent-semantic-cluster discovery.
Uses K-means for efficient clustering with Qdrant graph features.
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SemanticCluster:
    """
    Represents a semantic cluster in latent space.
    
    Can represent:
    - Aggregation of context vectors
    - Cluster center from K-means
    - Domain/concept/event/personality/semantic-field
    """
    cluster_id: str  # UUID string for Qdrant
    cluster_name: str  # Human-readable name
    center_vector: np.ndarray  # 384d cluster center
    domain: str
    concept: str
    member_count: int
    member_vectors: Optional[List[np.ndarray]] = None
    graph_neighbors: List[str] = None  # Qdrant point IDs of nearby clusters
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "center_vector": self.center_vector.tolist(),
            "domain": self.domain,
            "concept": self.concept,
            "member_count": self.member_count,
            "graph_neighbors": self.graph_neighbors or []
        }


class ClusterManager:
    """
    Manages semantic clusters and cluster centers.
    Optimized for speed and low RAM usage.
    """
    
    def __init__(self, num_clusters: int = 7, use_graph: bool = True):
        """
        Initialize cluster manager.
        
        Args:
            num_clusters: Number of clusters (default: 7, Miller's Law)
            use_graph: Use Qdrant graph features
        """
        self.num_clusters = num_clusters
        self.use_graph = use_graph
        self.clusters: List[SemanticCluster] = []
    
    def create_clusters(
        self,
        vectors: np.ndarray,
        domain: str,
        method: str = "kmeans"
    ) -> List[SemanticCluster]:
        """
        Create semantic clusters from vectors.
        
        Args:
            vectors: Array of shape (n_samples, 384)
            domain: Domain name
            method: Clustering method (kmeans for speed)
        
        Returns:
            List of SemanticCluster objects
        """
        logger.info(f"[ClusterManager] Creating {self.num_clusters} clusters for domain: {domain}")
        
        if method == "kmeans":
            # K-means: fast, low RAM
            # Adjust number of clusters for small datasets
            n_clusters = min(self.num_clusters, len(vectors))
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=100
            )
            labels = kmeans.fit_predict(vectors)
            centers = kmeans.cluster_centers_
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Create SemanticCluster objects
        clusters = []
        for i, center in enumerate(centers):
            cluster_vectors = vectors[labels == i]
            
            cluster = SemanticCluster(
                cluster_id=str(uuid.uuid4()),  # UUID for Qdrant
                cluster_name=f"{domain}_cluster_{i}",  # Human-readable name
                center_vector=center,
                domain=domain,
                concept=f"{domain}_concept_{i}",
                member_count=len(cluster_vectors),
                member_vectors=cluster_vectors.tolist() if len(cluster_vectors) < 100 else None
            )
            clusters.append(cluster)
        
        self.clusters = clusters
        logger.info(f"[ClusterManager] Created {len(clusters)} clusters")
        return clusters
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get all cluster center vectors."""
        return np.array([c.center_vector for c in self.clusters])
    
    def store_in_qdrant(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "cluster_centers"
    ):
        """
        Store cluster centers in Qdrant with graph features.
        
        Args:
            qdrant_client: Qdrant client
            collection_name: Collection name
        """
        logger.info(f"[ClusterManager] Storing {len(self.clusters)} clusters in Qdrant")
        
        # Create collection if not exists
        try:
            qdrant_client.get_collection(collection_name)
        except:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        # Build graph connections first
        if self.use_graph:
            self._build_cluster_graph()
        
        # Store clusters
        points = []
        for cluster in self.clusters:
            points.append(PointStruct(
                id=cluster.cluster_id,
                vector=cluster.center_vector.tolist(),
                payload={
                    "domain": cluster.domain,
                    "concept": cluster.concept,
                    "member_count": cluster.member_count,
                    "graph_neighbors": cluster.graph_neighbors or []
                }
            ))
        
        qdrant_client.upsert(collection_name=collection_name, points=points)
        logger.info(f"[ClusterManager] Stored clusters in {collection_name}")
    
    def _build_cluster_graph(self, k_neighbors: int = 5):
        """
        Build graph connections between clusters.
        Connects each cluster to its k nearest neighbors.
        
        Args:
            k_neighbors: Number of neighbors to connect
        """
        logger.info(f"[ClusterManager] Building cluster graph with k={k_neighbors}")
        
        centers = self.get_cluster_centers()
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(centers)
        
        # For each cluster, find k nearest neighbors
        for i, cluster in enumerate(self.clusters):
            # Get similarities to other clusters
            sims = similarities[i]
            # Exclude self
            sims[i] = -1
            # Get top-k
            neighbor_indices = np.argsort(sims)[-k_neighbors:]
            
            # Store neighbor IDs
            cluster.graph_neighbors = [
                self.clusters[idx].cluster_id
                for idx in neighbor_indices
            ]
        
        logger.info(f"[ClusterManager] Built graph connections")
    
    def load_from_qdrant(
        self,
        qdrant_client: QdrantClient,
        domain: str,
        collection_name: str = "cluster_centers"
    ) -> List[SemanticCluster]:
        """
        Load clusters for a domain from Qdrant.
        
        Args:
            qdrant_client: Qdrant client
            domain: Domain name
            collection_name: Collection name
        
        Returns:
            List of SemanticCluster objects
        """
        try:
            # Scroll through collection to find domain clusters
            points, _ = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={"must": [{"key": "domain", "match": {"value": domain}}]},
                limit=100,
                with_payload=True,
                with_vectors=True
            )
            
            clusters = []
            for point in points:
                cluster = SemanticCluster(
                    cluster_id=point.id,
                    center_vector=np.array(point.vector),
                    domain=point.payload["domain"],
                    concept=point.payload["concept"],
                    member_count=point.payload["member_count"],
                    graph_neighbors=point.payload.get("graph_neighbors", [])
                )
                clusters.append(cluster)
            
            self.clusters = clusters
            logger.info(f"[ClusterManager] Loaded {len(clusters)} clusters for domain: {domain}")
            return clusters
        
        except Exception as e:
            logger.error(f"[ClusterManager] Error loading clusters: {e}")
            return []
    
    def retrieve_with_graph(
        self,
        qdrant_client: QdrantClient,
        query_vector: np.ndarray,
        collection_name: str = "cluster_centers",
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve using graph-enhanced search.
        
        1. Find closest cluster center
        2. Navigate graph to nearby clusters
        3. Return semantically relevant results
        
        Args:
            qdrant_client: Qdrant client
            query_vector: Query vector (384d)
            collection_name: Collection name
            top_k: Number of results
        
        Returns:
            List of search results
        """
        # Search for closest cluster
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=1
        )
        
        if not results:
            return []
        
        # Get closest cluster
        closest_cluster = results[0]
        
        # Navigate graph to neighbors
        neighbor_ids = closest_cluster.payload.get("graph_neighbors", [])
        
        # Retrieve from neighbors
        all_results = [closest_cluster]
        for neighbor_id in neighbor_ids[:top_k-1]:
            try:
                neighbor = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=[neighbor_id]
                )
                if neighbor:
                    all_results.extend(neighbor)
            except:
                continue
        
        return all_results[:top_k]
