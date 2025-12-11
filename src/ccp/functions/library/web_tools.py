"""
Web scraping and MongoDB dataset collection tools for CCP.
Provides web content extraction and dataset management.
"""
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from typing import List, Dict, Optional
import json
from src.ccp.functions.utils import expose_as_ccp_tool
from src.ccp.core.settings import settings


@expose_as_ccp_tool
def scrape_webpage(url: str, selector: Optional[str] = None) -> str:
    """
    Scrape content from a webpage.
    
    Args:
        url: URL to scrape
        selector: Optional CSS selector to extract specific content
    
    Returns:
        Scraped content as JSON string
    """
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; CCPBot/1.0)'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if selector:
            elements = soup.select(selector)
            content = [elem.get_text(strip=True) for elem in elements]
        else:
            # Extract main text content
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator=' ', strip=True)
        
        return json.dumps({
            "url": url,
            "title": soup.title.string if soup.title else None,
            "content": content,
            "status": "success"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"url": url, "error": str(e), "status": "failed"})


@expose_as_ccp_tool
def scrape_links(url: str, pattern: Optional[str] = None) -> str:
    """
    Extract links from a webpage.
    
    Args:
        url: URL to scrape
        pattern: Optional pattern to filter links (substring match)
    
    Returns:
        List of links as JSON string
    """
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; CCPBot/1.0)'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if pattern is None or pattern in href:
                links.append({
                    "url": href,
                    "text": a_tag.get_text(strip=True)
                })
        
        return json.dumps({"source_url": url, "links": links, "count": len(links)}, indent=2)
    
    except Exception as e:
        return json.dumps({"source_url": url, "error": str(e)})


@expose_as_ccp_tool
def create_mongodb_collection(collection_name: str, documents: str) -> str:
    """
    Create a MongoDB collection and insert documents.
    
    Args:
        collection_name: Name of the collection to create
        documents: JSON string of documents to insert (list of dicts)
    
    Returns:
        Success message with inserted count
    """
    try:
        docs = json.loads(documents)
        if not isinstance(docs, list):
            docs = [docs]
        
        client = MongoClient(settings.mongo_uri)
        db = client["ccp_db"]
        collection = db[collection_name]
        
        result = collection.insert_many(docs)
        
        return json.dumps({
            "collection": collection_name,
            "inserted_count": len(result.inserted_ids),
            "status": "success"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"collection": collection_name, "error": str(e), "status": "failed"})


@expose_as_ccp_tool
def query_mongodb_collection(collection_name: str, query: str, limit: int = 10) -> str:
    """
    Query a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: JSON string of MongoDB query filter
        limit: Maximum number of results
    
    Returns:
        Query results as JSON string
    """
    try:
        query_filter = json.loads(query) if query else {}
        
        client = MongoClient(settings.mongo_uri)
        db = client["ccp_db"]
        collection = db[collection_name]
        
        results = list(collection.find(query_filter, {"_id": 0}).limit(limit))
        
        return json.dumps({
            "collection": collection_name,
            "results": results,
            "count": len(results)
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"collection": collection_name, "error": str(e)})


@expose_as_ccp_tool
def update_mongodb_document(collection_name: str, query: str, update: str) -> str:
    """
    Update documents in a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: JSON string of query filter
        update: JSON string of update operations
    
    Returns:
        Update result as JSON string
    """
    try:
        query_filter = json.loads(query)
        update_ops = json.loads(update)
        
        client = MongoClient(settings.mongo_uri)
        db = client["ccp_db"]
        collection = db[collection_name]
        
        result = collection.update_many(query_filter, update_ops)
        
        return json.dumps({
            "collection": collection_name,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "status": "success"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"collection": collection_name, "error": str(e), "status": "failed"})


@expose_as_ccp_tool
def delete_mongodb_documents(collection_name: str, query: str) -> str:
    """
    Delete documents from a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: JSON string of query filter
    
    Returns:
        Delete result as JSON string
    """
    try:
        query_filter = json.loads(query)
        
        client = MongoClient(settings.mongo_uri)
        db = client["ccp_db"]
        collection = db[collection_name]
        
        result = collection.delete_many(query_filter)
        
        return json.dumps({
            "collection": collection_name,
            "deleted_count": result.deleted_count,
            "status": "success"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"collection": collection_name, "error": str(e), "status": "failed"})
