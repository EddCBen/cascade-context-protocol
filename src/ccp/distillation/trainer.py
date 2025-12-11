"""
Domain Trainer - Trains neural networks for domain-specific routing.
Fine-tunes VectorNormalizer on domain datasets for improved routing accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from src.ccp.neural.models import VectorNormalizer
from src.ccp.distillation.models import TrainingDataset, DomainProfile
from src.ccp.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class DomainDataset(Dataset):
    """PyTorch dataset for domain training."""
    
    def __init__(self, training_dataset: TrainingDataset, embedding_service: EmbeddingService):
        """
        Initialize domain dataset.
        
        Args:
            training_dataset: Training dataset with samples
            embedding_service: Service for generating embeddings
        """
        self.samples = training_dataset.samples
        self.embedding_service = embedding_service
        self.embeddings = []
        self.labels = []
        
        # Pre-compute embeddings
        logger.info(f"[DomainDataset] Computing embeddings for {len(self.samples)} samples...")
        for sample in self.samples:
            embedding = self.embedding_service.embed(sample["input"])
            self.embeddings.append(torch.tensor(embedding, dtype=torch.float32))
            # For classification, convert label to index
            self.labels.append(sample["label"])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class DomainTrainer:
    """Trains neural networks for domain-specific routing."""
    
    def __init__(self, embedding_service: EmbeddingService, device: str = "cpu"):
        """
        Initialize domain trainer.
        
        Args:
            embedding_service: Service for generating embeddings
            device: Device to train on (cpu/cuda)
        """
        self.embedding_service = embedding_service
        self.device = torch.device(device)
    
    async def train_domain_router(
        self,
        dataset: TrainingDataset,
        domain_name: str,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Tuple[str, float]:
        """
        Train VectorNormalizer for domain routing.
        
        Args:
            dataset: Training dataset
            domain_name: Name of domain
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        
        Returns:
            Tuple of (weights_path, mastery_score)
        """
        logger.info(f"[DomainTrainer] Starting training for domain: {domain_name}")
        
        # Create PyTorch dataset
        domain_dataset = DomainDataset(dataset, self.embedding_service)
        
        # Split into train/val
        train_size = int(0.8 * len(domain_dataset))
        val_size = len(domain_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            domain_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = 384  # SentenceTransformer dimension
        hidden_dim = 256
        output_dim = 384
        
        model = VectorNormalizer(input_dim, hidden_dim, output_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()  # For embedding normalization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for embeddings, _ in train_loader:
                embeddings = embeddings.to(self.device)
                
                optimizer.zero_grad()
                normalized = model(embeddings)
                
                # Loss: encourage unit norm
                target_norm = torch.ones(normalized.size(0), 1).to(self.device)
                actual_norm = torch.norm(normalized, dim=1, keepdim=True)
                loss = criterion(actual_norm, target_norm)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for embeddings, _ in val_loader:
                    embeddings = embeddings.to(self.device)
                    normalized = model(embeddings)
                    
                    target_norm = torch.ones(normalized.size(0), 1).to(self.device)
                    actual_norm = torch.norm(normalized, dim=1, keepdim=True)
                    loss = criterion(actual_norm, target_norm)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            logger.info(f"[DomainTrainer] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Save weights
        weights_dir = Path("weights/domains")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(weights_dir / f"{domain_name}.pt")
        
        torch.save(model.state_dict(), weights_path)
        logger.info(f"[DomainTrainer] Saved weights to {weights_path}")
        
        # Compute mastery score (inverse of validation loss, normalized to 0-1)
        mastery_score = max(0.0, min(1.0, 1.0 - best_val_loss))
        logger.info(f"[DomainTrainer] Mastery score: {mastery_score:.3f}")
        
        return weights_path, mastery_score
    
    def compute_mastery_score(self, val_results: Dict) -> float:
        """
        Compute mastery score from validation results.
        
        Args:
            val_results: Validation metrics
        
        Returns:
            Mastery score (0.0 to 1.0)
        """
        # Simple heuristic: inverse of loss
        val_loss = val_results.get("loss", 1.0)
        return max(0.0, min(1.0, 1.0 - val_loss))
