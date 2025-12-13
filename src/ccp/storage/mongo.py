import motor.motor_asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.ccp.core.settings import settings

logger = logging.getLogger(__name__)

class MongoStorage:
    """
    Enhanced MongoDB storage with upsert, deduplication, and continual learning support.
    Handles both conversation history and scraped domain data.
    """
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongo_uri)
        self.db = self.client.ccp_db
        self.conversations = self.db.conversations
        self.scraped_data = self.db.scraped_data
        self.domain_profiles = self.db.domain_profiles
        
    async def initialize_indexes(self):
        """Create indexes for efficient queries."""
        # Scraped data indexes
        await self.scraped_data.create_index("content_hash", unique=True)
        await self.scraped_data.create_index([("domain", 1), ("scraped_at", -1)])
        await self.scraped_data.create_index("source")
        
        # Domain profiles index
        await self.domain_profiles.create_index("domain", unique=True)
        
        # Conversations index
        await self.conversations.create_index([("session_id", 1), ("timestamp", 1)])
        
        logger.info("[MongoStorage] Indexes created successfully")

    # ========== Conversation Methods (Legacy) ==========
    
    async def save_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        Saves a single message to the conversation history.
        """
        document = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        await self.conversations.insert_one(document)

    async def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves recent conversation history.
        """
        cursor = self.conversations.find({"session_id": session_id}).sort("timestamp", 1).limit(limit)
        history = await cursor.to_list(length=limit)
        return history
    
    # ========== Scraped Data Methods (New) ==========
    
    def _compute_content_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of content for deduplication.
        
        Args:
            content: Text content
        
        Returns:
            Hex digest of content hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def upsert_scraped_data(
        self,
        domain: str,
        source: str,
        url: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upsert scraped data with content hash deduplication.
        
        Args:
            domain: Domain name
            source: Source type (wikipedia, stackoverflow, web_search)
            url: Source URL
            title: Content title
            content: Text content
            metadata: Additional metadata
        
        Returns:
            Upserted document with _id and upserted flag
        """
        content_hash = self._compute_content_hash(content)
        
        document = {
            "domain": domain,
            "source": source,
            "url": url,
            "title": title,
            "content": content,
            "content_hash": content_hash,
            "metadata": metadata or {},
            "scraped_at": datetime.utcnow()
        }
        
        # Upsert based on content_hash
        result = await self.scraped_data.update_one(
            {"content_hash": content_hash},
            {"$set": document},
            upsert=True
        )
        
        # Get the document
        doc = await self.scraped_data.find_one({"content_hash": content_hash})
        
        return {
            "_id": str(doc["_id"]),
            "upserted": result.upserted_id is not None,
            "content_hash": content_hash
        }
    
    async def bulk_upsert_scraped_data(
        self,
        data_list: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Bulk upsert scraped data for efficiency.
        
        Args:
            data_list: List of dicts with keys: domain, source, url, title, content, metadata
        
        Returns:
            Statistics: inserted, updated, duplicates
        """
        stats = {"inserted": 0, "updated": 0, "duplicates": 0}
        
        for data in data_list:
            result = await self.upsert_scraped_data(
                domain=data["domain"],
                source=data["source"],
                url=data["url"],
                title=data["title"],
                content=data["content"],
                metadata=data.get("metadata", {})
            )
            
            if result["upserted"]:
                stats["inserted"] += 1
            else:
                stats["duplicates"] += 1
        
        logger.info(f"[MongoStorage] Bulk upsert: {stats['inserted']} inserted, {stats['duplicates']} duplicates")
        return stats
    
    async def get_all_domain_data(
        self,
        domains: List[str],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all scraped data for multiple domains (for continual learning).
        
        Args:
            domains: List of domain names
            limit: Optional limit on total documents
        
        Returns:
            List of scraped data documents
        """
        query = {"domain": {"$in": domains}}
        cursor = self.scraped_data.find(query).sort("scraped_at", -1)
        
        if limit:
            cursor = cursor.limit(limit)
        
        data = await cursor.to_list(length=limit or 10000)
        logger.info(f"[MongoStorage] Retrieved {len(data)} documents for domains: {domains}")
        return data
    
    async def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a domain (for cluster count determination).
        
        Args:
            domain: Domain name
        
        Returns:
            Statistics dict with counts by source, total, etc.
        """
        pipeline = [
            {"$match": {"domain": domain}},
            {"$group": {
                "_id": "$source",
                "count": {"$sum": 1}
            }}
        ]
        
        source_counts = {}
        async for doc in self.scraped_data.aggregate(pipeline):
            source_counts[doc["_id"]] = doc["count"]
        
        total = sum(source_counts.values())
        
        stats = {
            "domain": domain,
            "total_documents": total,
            "by_source": source_counts,
            "avg_content_length": 0  # Could compute if needed
        }
        
        logger.info(f"[MongoStorage] Domain stats for {domain}: {total} documents")
        return stats
    
    async def save_domain_profile(
        self,
        domain: str,
        num_samples: int,
        num_clusters: int,
        mastery_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save or update domain profile after training.
        
        Args:
            domain: Domain name
            num_samples: Number of training samples
            num_clusters: Number of clusters created
            mastery_score: Training mastery score
            metadata: Additional metadata
        """
        profile = {
            "domain": domain,
            "num_samples": num_samples,
            "num_clusters": num_clusters,
            "mastery_score": mastery_score,
            "metadata": metadata or {},
            "last_trained": datetime.utcnow()
        }
        
        await self.domain_profiles.update_one(
            {"domain": domain},
            {"$set": profile},
            upsert=True
        )
        
        logger.info(f"[MongoStorage] Saved domain profile for {domain}")
    
    async def get_domain_profile(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain profile if exists."""
        return await self.domain_profiles.find_one({"domain": domain})
    
    async def list_all_domains(self) -> List[str]:
        """Get list of all trained domains."""
        domains = await self.domain_profiles.distinct("domain")
        return domains
