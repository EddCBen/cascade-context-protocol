"""
Training configuration models for neural networks.
Inspired by latest research on Context2Vec and Mixture-of-Experts architectures.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class Context2VecConfig:
    """
    Configuration for NeuralVectorNormalizer (Context2Vec).
    
    Inspired by:
    - word2vec Parameter Learning Explained (Rong, 2014)
    - Transformer architecture (Vaswani et al., 2017)
    """
    # Architecture
    input_dim: int = 384  # SentenceTransformer dimension
    hidden_dim: int = 512
    num_layers: int = 4  # Transformer encoder layers
    num_heads: int = 8
    dropout: float = 0.1
    
    # Word2Vec-inspired training
    window_size: int = 5
    negative_samples: int = 5
    
    # Cluster routing
    num_clusters: int = 7  # Miller's Law: 7±2
    cluster_temperature: float = 0.1
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Output normalization
    normalize_output: bool = True
    norm_type: str = "l2"  # l2, layer_norm


@dataclass
class RouterConfig:
    """
    Configuration for SoftmaxRouter (Mixture-of-Experts).
    
    Inspired by:
    - Switch Transformers (Fedus et al., 2021)
    - Mixture-of-Experts (Shazeer et al., 2017)
    """
    # Architecture
    input_dim: int = 384
    hidden_dim: int = 256
    num_clusters: int = 7  # Match Context2Vec
    temperature: float = 1.0
    dropout: float = 0.2
    
    # Mixture-of-Experts
    top_k: int = 3  # Route to top-3 clusters
    load_balancing_loss_weight: float = 0.01
    
    # Gating
    use_gating: bool = True
    gating_type: str = "learned"  # learned, hash, random
    
    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    """General training configuration."""
    # General
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_samples: int = 200  # Minimum samples per domain
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "weights/checkpoints"
    best_model_path: str = "weights/best_model.pt"
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision: bool = True
    num_workers: int = 4  # DataLoader workers
    
    # Logging
    log_every: int = 10
    validate_every: int = 1


@dataclass
class ClusterConfig:
    """Configuration for semantic clustering."""
    # Clustering
    method: str = "kmeans"  # kmeans (faster, less RAM)
    num_clusters: int = 7  # Miller's Law
    max_iterations: int = 100
    random_state: int = 42
    
    # Qdrant storage
    collection_name: str = "cluster_centers"
    use_graph_features: bool = True
    graph_k_neighbors: int = 5  # Connect to 5 nearest clusters
    
    # Optimization
    parallel: bool = True
    n_jobs: int = -1  # Use all CPU cores


@dataclass
class ScraperConfig:
    """Configuration for enhanced data scraping."""
    # Minimum samples
    min_samples: int = 200
    max_samples: int = 500
    
    # Sources
    sources: List[str] = field(default_factory=lambda: [
        "wikipedia",
        "stackoverflow",
        "web_search"
    ])
    
    # Parallelization
    parallel: bool = True
    max_workers: int = 5
    
    # Retry logic
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 10
