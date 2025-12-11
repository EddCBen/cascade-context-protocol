"""
Dataset creator - Creates training datasets from scraped data.
Generates sample/label pairs for different task types.
"""
import json
import logging
from typing import List, Dict, Any
from src.ccp.distillation.models import ScrapedData, TrainingDataset, TaskType

logger = logging.getLogger(__name__)


class DatasetCreator:
    """Creates training datasets from scraped data."""
    
    def __init__(self, llm_service=None):
        """
        Initialize dataset creator.
        
        Args:
            llm_service: Optional LLM service for generating labels/summaries
        """
        self.llm_service = llm_service
    
    def create_dataset(
        self,
        scraped_data: List[ScrapedData],
        domain: str,
        task_type: TaskType
    ) -> TrainingDataset:
        """
        Create training dataset from scraped data.
        
        Args:
            scraped_data: List of scraped content
            domain: Domain name
            task_type: Type of training task
        
        Returns:
            TrainingDataset with sample/label pairs
        """
        dataset = TrainingDataset(domain=domain, task_type=task_type)
        
        if task_type == TaskType.CLASSIFICATION:
            dataset = self.create_classification_dataset(scraped_data, domain)
        elif task_type == TaskType.QA:
            dataset = self.create_qa_dataset(scraped_data, domain)
        elif task_type == TaskType.SUMMARIZATION:
            dataset = self.create_summarization_dataset(scraped_data, domain)
        elif task_type == TaskType.DOMAIN_ADAPTATION:
            dataset = self.create_domain_adaptation_dataset(scraped_data, domain)
        
        # Split into train/val/test
        dataset.split_data(train_ratio=0.8, val_ratio=0.1)
        
        logger.info(f"[DatasetCreator] Created {task_type} dataset with {len(dataset.samples)} samples")
        return dataset
    
    def create_classification_dataset(self, data: List[ScrapedData], domain: str) -> TrainingDataset:
        """
        Create classification dataset (text, category).
        
        Uses source as category label.
        """
        dataset = TrainingDataset(domain=domain, task_type=TaskType.CLASSIFICATION)
        
        for item in data:
            # Use source as category
            category = item.source
            
            # Create sample
            dataset.add_sample(
                input_text=item.content,
                label=category,
                metadata={
                    "title": item.title,
                    "url": item.url,
                    "domain": domain
                }
            )
        
        return dataset
    
    def create_qa_dataset(self, data: List[ScrapedData], domain: str) -> TrainingDataset:
        """
        Create Q&A dataset (question, answer).
        
        For StackOverflow data, uses title as question and content as answer.
        For other sources, generates questions from content.
        """
        dataset = TrainingDataset(domain=domain, task_type=TaskType.QA)
        
        for item in data:
            if item.source == "stackoverflow":
                # Title is the question, content is the answer
                dataset.add_sample(
                    input_text=item.title,
                    label=item.content,
                    metadata={
                        "url": item.url,
                        "domain": domain,
                        "question_type": "stackoverflow"
                    }
                )
            else:
                # Generate question from title/content
                # Simple heuristic: "What is {title}?"
                question = f"What is {item.title}?"
                answer = item.content[:500]  # Limit answer length
                
                dataset.add_sample(
                    input_text=question,
                    label=answer,
                    metadata={
                        "url": item.url,
                        "domain": domain,
                        "question_type": "generated"
                    }
                )
        
        return dataset
    
    def create_summarization_dataset(self, data: List[ScrapedData], domain: str) -> TrainingDataset:
        """
        Create summarization dataset (document, summary).
        
        Uses title as summary and content as document.
        """
        dataset = TrainingDataset(domain=domain, task_type=TaskType.SUMMARIZATION)
        
        for item in data:
            # Content is the document, title is the summary
            if len(item.content) > 100:  # Only use substantial content
                dataset.add_sample(
                    input_text=item.content,
                    label=item.title,
                    metadata={
                        "url": item.url,
                        "domain": domain,
                        "source": item.source
                    }
                )
        
        return dataset
    
    def create_domain_adaptation_dataset(self, data: List[ScrapedData], domain: str) -> TrainingDataset:
        """
        Create domain adaptation dataset (source_text, target_text).
        
        Pairs content from different sources for domain adaptation.
        """
        dataset = TrainingDataset(domain=domain, task_type=TaskType.DOMAIN_ADAPTATION)
        
        # Group by source
        by_source = {}
        for item in data:
            if item.source not in by_source:
                by_source[item.source] = []
            by_source[item.source].append(item)
        
        # Create pairs between different sources
        sources = list(by_source.keys())
        if len(sources) >= 2:
            source1, source2 = sources[0], sources[1]
            
            for item1, item2 in zip(by_source[source1], by_source[source2]):
                dataset.add_sample(
                    input_text=item1.content,
                    label=item2.content,
                    metadata={
                        "source_domain": source1,
                        "target_domain": source2,
                        "domain": domain
                    }
                )
        
        return dataset
    
    def augment_with_llm(self, dataset: TrainingDataset) -> TrainingDataset:
        """
        Use LLM to augment dataset with additional samples.
        
        Args:
            dataset: Existing dataset to augment
        
        Returns:
            Augmented dataset
        """
        if not self.llm_service:
            logger.warning("[DatasetCreator] No LLM service available for augmentation")
            return dataset
        
        # TODO: Implement LLM-based data augmentation
        # - Generate paraphrases
        # - Generate additional Q&A pairs
        # - Generate synthetic examples
        
        return dataset
