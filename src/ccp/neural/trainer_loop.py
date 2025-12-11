"""
PyTorch training loops with context managers for Context2Vec and SoftmaxRouter.
Implements sequential training: Context2Vec first, then SoftmaxRouter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import contextmanager
import logging
from typing import Dict, Tuple
from pathlib import Path
from src.ccp.neural.models import NeuralVectorNormalizer, SoftmaxRouter
from src.ccp.neural.training_config import TrainingConfig, Context2VecConfig, RouterConfig
from src.ccp.neural.semantic_cluster import ClusterManager
import numpy as np

logger = logging.getLogger(__name__)


class NeuralTrainer:
    """
    Unified training loop for both neural networks.
    Implements sequential training with context managers.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"[NeuralTrainer] Using device: {self.device}")
    
    @contextmanager
    def train_mode(self, model):
        """Context manager for training mode."""
        model.train()
        try:
            yield model
        finally:
            model.eval()
    
    @contextmanager
    def inference_mode(self, model):
        """Context manager for inference mode."""
        model.eval()
        with torch.no_grad():
            try:
                yield model
            finally:
                pass
    
    def train_context2vec(
        self,
        model: NeuralVectorNormalizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cluster_centers: np.ndarray,
        domain_name: str
    ) -> Tuple[str, float]:
        """
        Train Context2Vec network.
        
        Learns to transform embeddings into perfect vectors for cluster routing.
        
        Args:
            model: NeuralVectorNormalizer model
            train_loader: Training data loader
            val_loader: Validation data loader
            cluster_centers: Cluster center vectors [num_clusters, 384]
            domain_name: Domain name for saving weights
        
        Returns:
            Tuple of (weights_path, best_val_loss)
        """
        logger.info(f"[NeuralTrainer] Training Context2Vec for domain: {domain_name}")
        
        model = model.to(self.device)
        cluster_centers_tensor = torch.from_numpy(cluster_centers).float().to(self.device)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs
        )
        
        # Loss function (contrastive)
        criterion = Context2VecLoss(cluster_centers_tensor)
        
        best_val_loss = float('inf')
        weights_dir = Path(self.config.checkpoint_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            # Training
            with self.train_mode(model):
                train_loss = self._train_epoch_context2vec(
                    model, train_loader, optimizer, criterion
                )
            
            # Validation
            with self.inference_mode(model):
                val_loss = self._validate_context2vec(
                    model, val_loader, criterion
                )
            
            scheduler.step()
            
            logger.info(
                f"[Context2Vec] Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                weights_path = weights_dir / f"{domain_name}_context2vec.pt"
                torch.save(model.state_dict(), weights_path)
                logger.info(f"[Context2Vec] Saved best model: {weights_path}")
        
        return str(weights_path), best_val_loss
    
    def _train_epoch_context2vec(
        self,
        model: NeuralVectorNormalizer,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train one epoch for Context2Vec."""
        total_loss = 0.0
        
        for batch_idx, (embeddings, cluster_labels) in enumerate(train_loader):
            embeddings = embeddings.to(self.device)
            cluster_labels = cluster_labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            normalized = model(embeddings, mode="train")
            
            # Compute loss
            loss = criterion(normalized, cluster_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config.log_every == 0:
                logger.debug(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def _validate_context2vec(
        self,
        model: NeuralVectorNormalizer,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validate Context2Vec."""
        total_loss = 0.0
        
        for embeddings, cluster_labels in val_loader:
            embeddings = embeddings.to(self.device)
            cluster_labels = cluster_labels.to(self.device)
            
            # Forward pass
            normalized = model(embeddings, mode="inference")
            
            # Compute loss
            loss = criterion(normalized, cluster_labels)
            
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train_router(
        self,
        model: SoftmaxRouter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        domain_name: str
    ) -> Tuple[str, float]:
        """
        Train SoftmaxRouter.
        
        Learns to route normalized vectors to correct cluster indices.
        
        Args:
            model: SoftmaxRouter model
            train_loader: Training data loader
            val_loader: Validation data loader
            domain_name: Domain name for saving weights
        
        Returns:
            Tuple of (weights_path, best_val_accuracy)
        """
        logger.info(f"[NeuralTrainer] Training SoftmaxRouter for domain: {domain_name}")
        
        model = model.to(self.device)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        weights_dir = Path(self.config.checkpoint_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            # Training
            with self.train_mode(model):
                train_loss, train_acc = self._train_epoch_router(
                    model, train_loader, optimizer, criterion
                )
            
            # Validation
            with self.inference_mode(model):
                val_loss, val_acc = self._validate_router(
                    model, val_loader, criterion
                )
            
            scheduler.step()
            
            logger.info(
                f"[SoftmaxRouter] Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weights_path = weights_dir / f"{domain_name}_router.pt"
                torch.save(model.state_dict(), weights_path)
                logger.info(f"[SoftmaxRouter] Saved best model: {weights_path}")
        
        return str(weights_path), best_val_acc
    
    def _train_epoch_router(
        self,
        model: SoftmaxRouter,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train one epoch for SoftmaxRouter."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (embeddings, cluster_labels) in enumerate(train_loader):
            embeddings = embeddings.to(self.device)
            cluster_labels = cluster_labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            probs = model(embeddings)
            
            # Compute loss
            loss = criterion(probs, cluster_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(probs, 1)
            total += cluster_labels.size(0)
            correct += (predicted == cluster_labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_router(
        self,
        model: SoftmaxRouter,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate SoftmaxRouter."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for embeddings, cluster_labels in val_loader:
            embeddings = embeddings.to(self.device)
            cluster_labels = cluster_labels.to(self.device)
            
            # Forward pass
            probs = model(embeddings)
            
            # Compute loss
            loss = criterion(probs, cluster_labels)
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(probs, 1)
            total += cluster_labels.size(0)
            correct += (predicted == cluster_labels).sum().item()
        
        return total_loss / len(val_loader), correct / total


class Context2VecLoss(nn.Module):
    """
    Contrastive loss for Context2Vec.
    Pulls vectors towards target cluster centers.
    """
    
    def __init__(self, cluster_centers: torch.Tensor):
        super().__init__()
        self.cluster_centers = cluster_centers  # [num_clusters, 384]
    
    def forward(self, pred: torch.Tensor, target_clusters: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            pred: Predicted vectors [Batch, 384]
            target_clusters: Target cluster indices [Batch]
        
        Returns:
            Loss value
        """
        # Get target cluster centers
        target_centers = self.cluster_centers[target_clusters]  # [Batch, 384]
        
        # Positive loss: maximize similarity to target cluster
        pos_sim = F.cosine_similarity(pred, target_centers, dim=-1)
        pos_loss = 1 - pos_sim
        
        # Negative loss: minimize similarity to other clusters
        # Compute similarity to all clusters
        all_sims = torch.matmul(pred, self.cluster_centers.T)  # [Batch, num_clusters]
        
        # Mask out target clusters
        mask = torch.zeros_like(all_sims)
        mask.scatter_(1, target_clusters.unsqueeze(1), 1)
        neg_sims = all_sims * (1 - mask)
        
        # Negative loss: penalize high similarity to wrong clusters
        neg_loss = neg_sims.max(dim=1)[0]
        
        # Combined loss
        loss = pos_loss.mean() + 0.5 * neg_loss.mean()
        
        return loss
