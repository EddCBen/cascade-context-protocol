"""
CCP Search Engine - Unified search across Redis, MongoDB, Qdrant, and Web.
Provides sophisticated result aggregation and ranking.
"""
import requests
import json
import redis
from pymongo import MongoClient
from qdrant_client import QdrantClient
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from src.ccp.functions.utils import expose_as_ccp_tool
from src.ccp.core.settings import settings


class CCPSearchEngine:
    """Unified search engine across multiple data sources."""
    
    def __init__(self):
        self.redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
        self.mongo_client = MongoClient(settings.mongo_uri)
        self.qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    def search_redis(self, query: str, pattern: str = "*") -> List[Dict]:
        """Search Redis keys matching pattern."""
        try:
            keys = self.redis_client.keys(f"*{query}*{pattern}")
            results = []
            for key in keys[:10]:  # Limit to 10 results
                value = self.redis_client.get(key)
                results.append({
                    "source": "redis",
                    "key": key,
                    "value": value,
                    "score": 1.0  # Simple relevance score
                })
            return results
        except Exception as e:
            return [{"source": "redis", "error": str(e)}]
    
    def search_mongodb(self, query: str, collection: str = "ccp_db", limit: int = 10) -> List[Dict]:
        """Search MongoDB collections with text search."""
        try:
            db = self.mongo_client["ccp_db"]
            
            # Try text search first
            results = []
            for coll_name in db.list_collection_names():
                if collection != "ccp_db" and coll_name != collection:
                    continue
                
                coll = db[coll_name]
                # Simple field search (in production, use text indexes)
                docs = list(coll.find(
                    {"$or": [
                        {"name": {"$regex": query, "$options": "i"}},
                        {"description": {"$regex": query, "$options": "i"}},
                        {"content": {"$regex": query, "$options": "i"}}
                    ]},
                    {"_id": 0}
                ).limit(limit))
                
                for doc in docs:
                    results.append({
                        "source": "mongodb",
                        "collection": coll_name,
                        "document": doc,
                        "score": 0.8  # Relevance score
                    })
            
            return results
        except Exception as e:
            return [{"source": "mongodb", "error": str(e)}]
    
    def search_qdrant(self, query: str, collection: str = "function_registry", limit: int = 10) -> List[Dict]:
        """Search Qdrant vector store."""
        try:
            # This would use embeddings in production
            # For now, we'll use payload filtering
            points, _ = self.qdrant_client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in points:
                payload = point.payload
                # Simple relevance check
                if query.lower() in str(payload).lower():
                    results.append({
                        "source": "qdrant",
                        "collection": collection,
                        "payload": payload,
                        "score": point.score if hasattr(point, 'score') else 0.9
                    })
            
            return results
        except Exception as e:
            return [{"source": "qdrant", "error": str(e)}]
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search web and scrape content from top results."""
        try:
            # Use DuckDuckGo HTML search (no API key needed)
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            
            response = requests.get(search_url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; CCPBot/1.0)'
            }, timeout=10)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.select('.result')[:num_results]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                url_elem = result.select_one('.result__url')
                
                if title_elem and url_elem:
                    results.append({
                        "source": "web",
                        "title": title_elem.get_text(strip=True),
                        "url": url_elem.get_text(strip=True),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                        "score": 0.7  # Web results get lower score
                    })
            
            return results
        except Exception as e:
            return [{"source": "web", "error": str(e)}]
    
    def aggregate_results(self, all_results: List[Dict], max_results: int = 20) -> List[Dict]:
        """Aggregate and rank results from multiple sources."""
        # Remove error results
        valid_results = [r for r in all_results if "error" not in r]
        
        # Sort by score (descending)
        sorted_results = sorted(valid_results, key=lambda x: x.get("score", 0), reverse=True)
        
        # Deduplicate based on content similarity (simple version)
        unique_results = []
        seen_content = set()
        
        for result in sorted_results:
            content_key = str(result.get("document", result.get("payload", result.get("title", ""))))[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results[:max_results]


@expose_as_ccp_tool
def ccp_search(
    query: str,
    sources: str = "all",
    max_results: int = 20,
    mongodb_collection: str = "ccp_db",
    qdrant_collection: str = "function_registry"
) -> str:
    """
    Unified search across Redis, MongoDB, Qdrant, and Web.
    
    Args:
        query: Search query
        sources: Comma-separated list of sources ("redis,mongodb,qdrant,web" or "all")
        max_results: Maximum number of results to return
        mongodb_collection: MongoDB collection to search (default: all in ccp_db)
        qdrant_collection: Qdrant collection to search
    
    Returns:
        Aggregated search results as JSON string
    """
    try:
        engine = CCPSearchEngine()
        
        # Parse sources
        if sources == "all":
            source_list = ["redis", "mongodb", "qdrant", "web"]
        else:
            source_list = [s.strip() for s in sources.split(",")]
        
        # Collect results from each source
        all_results = []
        
        if "redis" in source_list:
            all_results.extend(engine.search_redis(query))
        
        if "mongodb" in source_list:
            all_results.extend(engine.search_mongodb(query, mongodb_collection))
        
        if "qdrant" in source_list:
            all_results.extend(engine.search_qdrant(query, qdrant_collection))
        
        if "web" in source_list:
            all_results.extend(engine.search_web(query))
        
        # Aggregate and rank results
        final_results = engine.aggregate_results(all_results, max_results)
        
        return json.dumps({
            "query": query,
            "sources_searched": source_list,
            "total_results": len(final_results),
            "results": final_results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"query": query, "error": str(e)})


@expose_as_ccp_tool
def search_with_expansion(query: str, sources: str = "all", max_results: int = 20) -> str:
    """
    Search with automatic query expansion.
    
    Args:
        query: Original search query
        sources: Data sources to search
        max_results: Maximum results
    
    Returns:
        Search results with expanded query as JSON string
    """
    try:
        # Expand query first
        from src.ccp.functions.library.query_tools import expand_query
        expansion_result = json.loads(expand_query(query, "synonyms"))
        
        expanded_terms = expansion_result.get("expanded_terms", [])
        expanded_query = " ".join(expanded_terms)
        
        # Search with expanded query
        search_result = json.loads(ccp_search(expanded_query, sources, max_results))
        
        # Add expansion info
        search_result["query_expansion"] = {
            "original": query,
            "expanded": expanded_query,
            "terms_added": len(expanded_terms) - len(expansion_result.get("original_terms", []))
        }
        
        return json.dumps(search_result, indent=2)
    
    except Exception as e:
        return json.dumps({"query": query, "error": str(e)})
