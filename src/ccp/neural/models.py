"""
Advanced neural models for CCP routing system.
- NeuralVectorNormalizer (Context2Vec): Transforms embeddings to perfect cluster-routing vectors
- SoftmaxRouter: Routes vectors to semantic cluster centers
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from src.ccp.neural.training_config import Context2VecConfig, RouterConfig


class NeuralVectorNormalizer(nn.Module):
    """
    Context2Vec network for semantic cluster routing.
    
    Transforms context block embeddings into perfect vectors that score high
    on similarity measures with cluster centers.
    
    Architecture:
    - Transformer encoder for context understanding
    - Projection layer for dimensionality
    - Layer normalization for stability
    
    Training:
    - Word2Vec-style contrastive learning
    - Learns to route to semantic cluster centers
    - Improves with each domain learned
    """
    
    def __init__(self, config: Context2VecConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Transformer encoder for context understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output projection back to original dimension
        self.output_proj = nn.Linear(config.hidden_dim, config.input_dim)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(config.input_dim)
        
    def forward(self, x: torch.Tensor, mode: str = "inference") -> torch.Tensor:
        """
        Transform embedding to perfect cluster-routing vector.
        
        Args:
            x: Input embeddings [Batch, 384] or [Batch, Seq, 384]
            mode: "train" or "inference"
        
        Returns:
            Normalized vector [Batch, 384]
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [Batch, 1, 384]
        
        # Project to hidden dimension
        x = self.input_proj(x)  # [Batch, Seq, 512]
        
        # Transformer encoding
        x = self.transformer(x)  # [Batch, Seq, 512]
        
        # Project back to original dimension
        x = self.output_proj(x)  # [Batch, Seq, 384]
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [Batch, 384]
        
        # Normalize
        x = self.layer_norm(x)
        
        # L2 normalization for cosine similarity
        if self.config.normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        
        return x
    
    def load_weights(self, domain_id: str, weights_dir: str = "weights/domains"):
        """
        Load domain-specific weights.
        
        Args:
            domain_id: Domain identifier
            weights_dir: Directory containing weights
        """
        path = os.path.join(weights_dir, f"{domain_id}_context2vec.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"[NeuralVectorNormalizer] Loaded weights for domain: {domain_id}")
        else:
            print(f"[NeuralVectorNormalizer] No weights found for {domain_id}, using base weights")


class SoftmaxRouter(nn.Module):
    """
    Mixture-of-Experts router for cluster center routing.
    
    Routes normalized vectors to predefined cluster indices.
    Trained as classification task: high similarity → correct cluster index.
    
    Architecture:
    - MLP with gating mechanism
    - Temperature-scaled softmax
    - Top-k routing
    
    Inspired by:
    - Switch Transformers (Fedus et al., 2021)
    - Mixture-of-Experts (Shazeer et al., 2017)
    """
    
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Router network
        self.router_network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_clusters)
        )
        
        # Gating mechanism (optional)
        if config.use_gating:
            self.gate = nn.Linear(config.input_dim, config.num_clusters)
        
        # Cluster centers (loaded from Qdrant)
        self.register_buffer('cluster_centers', torch.zeros(config.num_clusters, config.input_dim))
        self.cluster_centers_loaded = False
    
    def load_cluster_centers(self, centers: np.ndarray):
        """
        Load cluster centers from Qdrant.
        
        Args:
            centers: Cluster center vectors [num_clusters, 384]
        """
        self.cluster_centers = torch.from_numpy(centers).float()
        self.cluster_centers_loaded = True
        print(f"[SoftmaxRouter] Loaded {len(centers)} cluster centers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Route vector to cluster centers.
        
        Args:
            x: Normalized context vector [Batch, 384]
        
        Returns:
            Cluster probabilities [Batch, num_clusters]
        """
        # Router logits
        logits = self.router_network(x)  # [Batch, num_clusters]
        
        # Optional gating
        if self.config.use_gating:
            gate_logits = self.gate(x)
            logits = logits + gate_logits
        
        # Temperature-scaled softmax
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        
        return probs
    
    def route(self, x: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route to top-k clusters.
        
        Args:
            x: Normalized context vector [Batch, 384]
            top_k: Number of clusters to route to (default: config.top_k)
        
        Returns:
            Tuple of (cluster_indices, cluster_probs)
        """
        top_k = top_k or self.config.top_k
        
        # Get probabilities
        probs = self.forward(x)  # [Batch, num_clusters]
        
        # Top-k routing
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        
        return top_indices, top_probs
    
    def get_cluster_vectors(self, cluster_indices: torch.Tensor) -> torch.Tensor:
        """
        Get cluster center vectors for given indices.
        
        Args:
            cluster_indices: Cluster indices [Batch, top_k]
        
        Returns:
            Cluster vectors [Batch, top_k, 384]
        """
        if not self.cluster_centers_loaded:
            raise RuntimeError("Cluster centers not loaded. Call load_cluster_centers() first.")
        
        return self.cluster_centers[cluster_indices]
    
    def load_weights(self, domain_id: str, weights_dir: str = "weights/domains"):
        """
        Load domain-specific weights.
        
        Args:
            domain_id: Domain identifier
            weights_dir: Directory containing weights
        """
        path = os.path.join(weights_dir, f"{domain_id}_router.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"[SoftmaxRouter] Loaded weights for domain: {domain_id}")
        else:
            print(f"[SoftmaxRouter] No weights found for {domain_id}, using base weights")


# Legacy compatibility
VectorNormalizer = NeuralVectorNormalizer
