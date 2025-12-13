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
        # Initialize embedding model for vector search
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

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
        """Search MongoDB collections with text search (requires text index) or regex fallback."""
        try:
            db = self.mongo_client["ccp_db"]
            results = []
            
            for coll_name in db.list_collection_names():
                if collection != "ccp_db" and coll_name != collection:
                    continue
                
                coll = db[coll_name]
                
                # specific exposed collections (Data-Clusters, Function-store)
                # User asked to always expose them.
                
                # Hybrid: Try text search first (if index exists)
                try:
                    docs = list(coll.find(
                        {"$text": {"$search": query}},
                        {"score": {"$meta": "textScore"}}
                    ).sort([("score", {"$meta": "textScore"})]).limit(limit))
                    
                    for doc in docs:
                        results.append({
                            "source": "mongodb",
                            "collection": coll_name,
                            "document": doc,
                            "score": doc.get('score', 0.5) / 10.0 # Normalize roughly
                        })
                except Exception:
                    # Fallback to regex if no text index
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
                            "score": 0.5
                        })
            
            return results
        except Exception as e:
            return [{"source": "mongodb", "error": str(e)}]
    
    def search_qdrant(self, query: str, collection: str = "function_registry", limit: int = 10) -> List[Dict]:
        """Search Qdrant vector store using real embeddings."""
        try:
            vector = self.encoder.encode(query).tolist()
            
            points = self.qdrant_client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit
            )
            
            results = []
            for point in points:
                results.append({
                    "source": "qdrant",
                    "collection": collection,
                    "payload": point.payload,
                    "score": point.score
                })
            
            return results
        except Exception as e:
            # Check if collection exists, if not maybe return empty or error
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
        """Aggregate and rank results from multiple sources (Hybrid Fusion)."""
        # Remove error results
        valid_results = [r for r in all_results if "error" not in r]
        
        # Reciprocal Rank Fusion implementation or simple score sorting
        # Since scores are different scales (Qdrant: cosine 0-1, Mongo: textScore arbitrary), 
        # we should normalize or just rank by source priority if naive.
        # But RRF is better.
        
        # Simple Deduplication
        seen_content = set()
        unique_results = []
        
        # Sort by raw score first to get best candidates
        sorted_raw = sorted(valid_results, key=lambda x: float(x.get("score", 0)), reverse=True)
        
        for result in sorted_raw:
            # content key based on payload/doc
            content = str(result.get("document") or result.get("payload") or result.get("snippet"))
            if content not in seen_content:
                seen_content.add(content)
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
