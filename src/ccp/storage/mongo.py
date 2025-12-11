import motor.motor_asyncio
import logging
from typing import List, Dict, Any
from src.ccp.core.settings import settings

logger = logging.getLogger(__name__)

class MongoStorage:
    """
    Handles long-term persistence of conversation history using MongoDB (Async).
    """
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongo_uri)
        self.db = self.client.ccp_db
        self.collection = self.db.conversations

    async def save_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        Saves a single message to the conversation history.
        """
        document = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": "TODO" # Should add timestamp
        }
        await self.collection.insert_one(document)

    async def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves recent conversation history.
        """
        cursor = self.collection.find({"session_id": session_id}).sort("_id", 1).limit(limit) # -1 desc, 1 asc
        # We likely want chronological order for context reconstruction
        # Logic might need adjustment if we want *last* N messages but ordered chronologically.
        # usually: sort desc by time, limit N, then reverse.
        
        # simpler approach for now:
        history = await cursor.to_list(length=limit)
        return history
