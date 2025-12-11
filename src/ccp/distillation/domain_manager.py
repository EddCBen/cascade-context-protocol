"""
Domain Manager - Manages domain states, weights, and activation.
Handles domain switching and weight loading for neural routing.
"""
import logging
from typing import Set, Dict, List, Optional
from pymongo import MongoClient
from src.ccp.distillation.models import DomainProfile
from src.ccp.core.settings import settings

logger = logging.getLogger(__name__)


class DomainManager:
    """Manages domain states and neural weights."""
    
    def __init__(self, mongo_client: MongoClient):
        """
        Initialize domain manager.
        
        Args:
            mongo_client: MongoDB client for domain registry
        """
        self.mongo_client = mongo_client
        self.db = mongo_client["ccp_storage"]
        self.domain_collection = self.db["domain_registry"]
        
        self.active_domains: Set[str] = set()
        self.domain_weights: Dict[str, str] = {}  # domain -> weights_path
        
        # Load active domains on init
        self._load_active_domains()
    
    def _load_active_domains(self):
        """Load all enabled domains from database."""
        try:
            domains = self.domain_collection.find({"enabled": True})
            for domain_doc in domains:
                domain_name = domain_doc["name"]
                self.active_domains.add(domain_name)
                if domain_doc.get("neural_weights_path"):
                    self.domain_weights[domain_name] = domain_doc["neural_weights_path"]
            
            logger.info(f"[DomainManager] Loaded {len(self.active_domains)} active domains")
        except Exception as e:
            logger.error(f"[DomainManager] Error loading active domains: {e}")
    
    async def register_domain(self, profile: DomainProfile) -> bool:
        """
        Register a new domain in the registry.
        
        Args:
            profile: Domain profile to register
        
        Returns:
            True if successful
        """
        try:
            self.domain_collection.update_one(
                {"name": profile.name},
                {"$set": profile.model_dump()},
                upsert=True
            )
            
            if profile.enabled:
                self.active_domains.add(profile.name)
                if profile.neural_weights_path:
                    self.domain_weights[profile.name] = profile.neural_weights_path
            
            logger.info(f"[DomainManager] Registered domain: {profile.name}")
            return True
        except Exception as e:
            logger.error(f"[DomainManager] Error registering domain {profile.name}: {e}")
            return False
    
    async def get_domain(self, domain_name: str) -> Optional[DomainProfile]:
        """Get domain profile by name."""
        try:
            domain_doc = self.domain_collection.find_one({"name": domain_name})
            if domain_doc:
                domain_doc.pop("_id", None)
                return DomainProfile(**domain_doc)
            return None
        except Exception as e:
            logger.error(f"[DomainManager] Error getting domain {domain_name}: {e}")
            return None
    
    async def get_all_domains(self) -> List[DomainProfile]:
        """Get all registered domains."""
        try:
            domains = []
            for domain_doc in self.domain_collection.find():
                domain_doc.pop("_id", None)
                domains.append(DomainProfile(**domain_doc))
            return domains
        except Exception as e:
            logger.error(f"[DomainManager] Error getting all domains: {e}")
            return []
    
    async def get_active_domains(self) -> List[DomainProfile]:
        """Get all enabled domains."""
        try:
            domains = []
            for domain_doc in self.domain_collection.find({"enabled": True}):
                domain_doc.pop("_id", None)
                domains.append(DomainProfile(**domain_doc))
            return domains
        except Exception as e:
            logger.error(f"[DomainManager] Error getting active domains: {e}")
            return []
    
    async def toggle_domain(self, domain_name: str, enabled: bool) -> bool:
        """
        Enable or disable a domain.
        
        Args:
            domain_name: Name of domain to toggle
            enabled: True to enable, False to disable
        
        Returns:
            True if successful
        """
        try:
            result = self.domain_collection.update_one(
                {"name": domain_name},
                {"$set": {"enabled": enabled}}
            )
            
            if result.modified_count > 0:
                if enabled:
                    self.active_domains.add(domain_name)
                else:
                    self.active_domains.discard(domain_name)
                
                logger.info(f"[DomainManager] Toggled domain {domain_name}: enabled={enabled}")
                return True
            return False
        except Exception as e:
            logger.error(f"[DomainManager] Error toggling domain {domain_name}: {e}")
            return False
    
    async def load_domain_weights(self, domain_name: str) -> Optional[str]:
        """
        Load neural weights for a domain.
        
        Args:
            domain_name: Name of domain
        
        Returns:
            Path to weights file if available
        """
        if domain_name in self.domain_weights:
            return self.domain_weights[domain_name]
        
        # Try to load from database
        domain = await self.get_domain(domain_name)
        if domain and domain.neural_weights_path:
            self.domain_weights[domain_name] = domain.neural_weights_path
            return domain.neural_weights_path
        
        return None
    
    async def update_mastery_score(self, domain_name: str, score: float) -> bool:
        """Update mastery score for a domain."""
        try:
            result = self.domain_collection.update_one(
                {"name": domain_name},
                {"$set": {"mastery_score": score}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"[DomainManager] Error updating mastery score: {e}")
            return False
