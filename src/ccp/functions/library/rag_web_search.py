"""
Advanced RAG (Retrieval-Augmented Generation) Web Search Tools.
Provides specialized search capabilities for News, Academic papers, and deep content extraction.
"""
import json
import logging
import requests
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
from src.ccp.functions.utils import expose_as_ccp_tool

logger = logging.getLogger(__name__)

# Try importing duckduckgo_search, else fallback
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logger.warning("duckduckgo_search not installed. Advanced search features will be limited.")

@expose_as_ccp_tool
def search_web_advanced(query: str, region: str = "wt-wt", time_limit: str = "y", max_results: int = 10) -> str:
    """
    Advanced Web Search with filters.
    
    Args:
        query: Search query
        region: Region code (e.g., 'us-en', 'uk-en', 'wt-wt' for world)
        time_limit: Time limit ('d' for day, 'w' for week, 'm' for month, 'y' for year)
        max_results: Max results
    
    Returns:
        JSON string of results
    """
    try:
        results = []
        if HAS_DDGS:
            with DDGS() as ddgs:
                # DDGS support text search with region and time
                ddgs_gen = ddgs.text(query, region=region, timelimit=time_limit, max_results=max_results)
                if ddgs_gen:
                    for r in ddgs_gen:
                        results.append({
                            "title": r.get('title'),
                            "href": r.get('href'),
                            "body": r.get('body'),
                            "source": "duckduckgo_advanced"
                        })
        else:
            # Fallback to simple scraping hack (less reliable)
            # For robustness in this demo environment, we return a warning if dependencies missing
            return json.dumps({"error": "duckduckgo_search library required for advanced features."})
            
        return json.dumps({
            "query": query,
            "filters": {"region": region, "time": time_limit},
            "results": results
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})

@expose_as_ccp_tool
def search_news(query: str, max_results: int = 10) -> str:
    """
    Search specifically for News articles.
    
    Args:
        query: News topic
        max_results: Max articles
        
    Returns:
        JSON string of news results
    """
    try:
        results = []
        if HAS_DDGS:
            with DDGS() as ddgs:
                ddgs_gen = ddgs.news(query, max_results=max_results)
                if ddgs_gen:
                    for r in ddgs_gen:
                        results.append({
                            "title": r.get('title'),
                            "href": r.get('href'),
                            "snippet": r.get('body'),
                            "date": r.get('date'),
                            "source": r.get('source'),
                            "image": r.get('image')
                        })
        else:
            return json.dumps({"error": "duckduckgo_search library required."})
            
        return json.dumps({
            "query": query,
            "vertical": "news",
            "results": results
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@expose_as_ccp_tool
def extract_article_content(url: str) -> str:
    """
    Extract the main content from a news article or webpage, removing boilerplate.
    
    Args:
        url: URL to extract
        
    Returns:
        JSON with title and main content text
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Heuristic extraction
        # 1. Remove obvious junk
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
            tag.decompose()
            
        # 2. Look for article tag
        article = soup.find('article')
        if article:
            content_elem = article
        else:
            # Fallback: Find largest block of p tags
            # This is naive but works for many implementation
            content_elem = soup.body
            
        text = content_elem.get_text(separator='\n', strip=True) if content_elem else ""
        
        # Initial cleanup
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20] # Filter short lines
        clean_text = "\n".join(lines)
        
        return json.dumps({
            "url": url,
            "title": soup.title.string if soup.title else "",
            "content": clean_text[:5000], # Limit for context window
            "length": len(clean_text)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"url": url, "error": str(e)})

@expose_as_ccp_tool
def cross_check_fact(claim: str) -> str:
    """
    Cross-check a factual claim against web sources.
    
    Args:
        claim: The statement to verify
        
    Returns:
        JSON with verification results
    """
    # 1. Search for the claim
    search_res = json.loads(search_web_advanced(claim, max_results=5))
    results = search_res.get("results", [])
    
    # 2. Simple verification (In a real system, an LLM would analyze these snippets)
    # Here we just return the sources found for the claim
    
    return json.dumps({
        "claim": claim,
        "verification_sources": results,
        "status": "sources_retrieved"
    }, indent=2)
