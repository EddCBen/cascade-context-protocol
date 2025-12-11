"""
Domain models for the Distillation Engine.
Defines data structures for domain profiles, training datasets, and mastery tracking.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    """Supported training task types."""
    CLASSIFICATION = "classification"
    QA = "qa"
    SUMMARIZATION = "summarization"
    DOMAIN_ADAPTATION = "domain_adaptation"


class DomainProfile(BaseModel):
    """Profile of a learned domain with mastery metrics."""
    name: str
    subdomains: List[str] = Field(default_factory=list)
    task_types: List[TaskType] = Field(default_factory=list)
    dataset_collection: str  # MongoDB collection name
    sample_count: int = 0
    mastery_score: float = 0.0  # 0.0 to 1.0
    neural_weights_path: Optional[str] = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_trained: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingDataset(BaseModel):
    """Training dataset for a specific domain and task type."""
    domain: str
    task_type: TaskType
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    split: Dict[str, int] = Field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    
    def add_sample(self, input_text: str, label: Any, metadata: Optional[Dict] = None):
        """Add a training sample."""
        self.samples.append({
            "input": input_text,
            "label": label,
            "metadata": metadata or {}
        })
        
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split dataset into train/val/test."""
        total = len(self.samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        self.split = {
            "train": train_size,
            "val": val_size,
            "test": total - train_size - val_size
        }


class ScrapedData(BaseModel):
    """Raw scraped data from web sources."""
    source: str  # wikipedia, arxiv, stackoverflow, etc.
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class DomainDistillationRequest(BaseModel):
    """Request to distill a new domain."""
    domain: str
    task_description: Optional[str] = None
    max_samples: int = 1000
    task_types: List[TaskType] = Field(default_factory=lambda: [TaskType.CLASSIFICATION, TaskType.QA])
    scraping_sources: List[str] = Field(default_factory=lambda: ["wikipedia", "stackoverflow"])


class DomainMasteryResponse(BaseModel):
    """Response containing domain mastery information."""
    domains: List[DomainProfile]
    total_domains: int
    active_domains: int
