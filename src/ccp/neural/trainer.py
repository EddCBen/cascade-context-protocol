import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import os
from src.ccp.neural.models import VectorNormalizer

class NeuralTrainer:
    """
    Trainer for CCP Neural Models.
    Specializes in fine-tuning VectorNormalizer on synthetic QA pairs.
    """
    def __init__(self, input_dim: int = 768):
        self.input_dim = input_dim

    def _get_embedding(self, text: str) -> torch.Tensor:
        # Mock embedding generation
        return torch.randn(self.input_dim)

    def train_domain_adapter(self, dataset_id: str, epochs: int = 5):
        """
        Loads data from MongoDB and trains the VectorNormalizer.
        """
        print(f"Starting training for domain/dataset: {dataset_id}")
        
        # 1. Load QA pairs from MongoDB (Mock)
        # In reality: db.training_datasets.find({"dataset_id": dataset_id})
        dataset = [
            {"query": "User Query 1", "answer": "Ideal Answer 1"},
            {"query": "User Query 2", "answer": "Ideal Answer 2"},
            # ...
        ]
        
        # 2. Prepare Model
        model = VectorNormalizer(input_dim=self.input_dim)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CosineEmbeddingLoss()
        
        # 3. Training Loop
        for epoch in range(epochs):
            total_loss = 0
            for item in dataset:
                optimizer.zero_grad()
                
                # Convert to vectors
                input_vec = self._get_embedding(item["query"]).unsqueeze(0) # Batch dim
                target_vec = self._get_embedding(item["answer"]).unsqueeze(0)
                
                # Forward
                output_vec = model(input_vec)
                
                # Loss (Maximize similarity -> target 1)
                # target for CosineEmbeddingLoss is a tensor of 1s or -1s.
                target_label = torch.ones(1) 
                
                loss = criterion(output_vec, target_vec, target_label)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}")
            
        # 4. Save Weights
        os.makedirs("weights", exist_ok=True)
        save_path = f"weights/normalizer_{dataset_id}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")

