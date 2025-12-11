"""
Query expansion and preprocessing tools for CCP.
Enhances search queries with synonyms, related terms, and context.
"""
import re
from typing import List, Dict
import json
from src.ccp.functions.utils import expose_as_ccp_tool


@expose_as_ccp_tool
def expand_query(query: str, expansion_type: str = "synonyms") -> str:
    """
    Expand a search query with related terms.
    
    Args:
        query: Original search query
        expansion_type: Type of expansion ("synonyms", "related", "contextual")
    
    Returns:
        Expanded query terms as JSON string
    """
    try:
        # Extract key terms
        terms = re.findall(r'\b\w+\b', query.lower())
        
        # Simple synonym/related term mapping (in production, use WordNet or LLM)
        expansion_map = {
            "find": ["search", "locate", "discover", "retrieve"],
            "data": ["information", "dataset", "records", "content"],
            "analyze": ["examine", "study", "investigate", "evaluate"],
            "create": ["generate", "build", "make", "produce"],
            "delete": ["remove", "erase", "eliminate"],
            "update": ["modify", "change", "edit", "revise"],
            "average": ["mean", "median", "typical"],
            "calculate": ["compute", "determine", "evaluate"],
        }
        
        expanded_terms = set(terms)
        
        if expansion_type in ["synonyms", "related"]:
            for term in terms:
                if term in expansion_map:
                    expanded_terms.update(expansion_map[term])
        
        if expansion_type == "contextual":
            # Add contextual terms based on query intent
            if any(word in terms for word in ["find", "search", "get"]):
                expanded_terms.update(["retrieve", "fetch", "query"])
            if any(word in terms for word in ["analyze", "calculate"]):
                expanded_terms.update(["statistics", "metrics", "analysis"])
        
        return json.dumps({
            "original_query": query,
            "original_terms": list(terms),
            "expanded_terms": list(expanded_terms),
            "expansion_type": expansion_type
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"error": str(e)})


@expose_as_ccp_tool
def extract_query_intent(query: str) -> str:
    """
    Extract the intent from a search query.
    
    Args:
        query: Search query
    
    Returns:
        Query intent classification as JSON string
    """
    try:
        query_lower = query.lower()
        
        # Intent patterns
        intents = {
            "search": ["find", "search", "look for", "get", "retrieve"],
            "create": ["create", "make", "generate", "build", "add"],
            "update": ["update", "modify", "change", "edit", "revise"],
            "delete": ["delete", "remove", "erase", "eliminate"],
            "analyze": ["analyze", "calculate", "compute", "evaluate", "assess"],
            "list": ["list", "show", "display", "enumerate"],
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        primary_intent = detected_intents[0] if detected_intents else "unknown"
        
        return json.dumps({
            "query": query,
            "primary_intent": primary_intent,
            "all_intents": detected_intents
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"error": str(e)})
