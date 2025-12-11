import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from src.ccp.functions.utils import expose_as_ccp_tool
import logging

logger = logging.getLogger(__name__)

@expose_as_ccp_tool
def pandas_transform(json_data: str, operation: str, query: Optional[str] = None, group_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs data transformations on JSON data using Pandas.
    
    Args:
        json_data (str): JSON string representing a list of records.
        operation (str): The operation to perform: 'filter', 'group_count', 'describe', 'sort'.
        query (Optional[str]): The query string for filtering (e.g. "age > 30"). Used if operation is 'filter'.
        group_by (Optional[str]): Column name to group by. Used if operation is 'group_count'.
        
    Returns:
        Dict[str, Any]: Result of the transformation (JSON serializable).
    """
    try:
        data = json.loads(json_data)
        if not isinstance(data, list):
            # Try to convert dict to list if possible or just fail
            return {"error": "Input JSON must be a list of records"}
            
        df = pd.DataFrame(data)
        
        if operation == "filter" and query:
            filtered_df = df.query(query)
            return {"result": filtered_df.to_dict(orient="records")}
            
        elif operation == "group_count" and group_by:
            if group_by not in df.columns:
                 return {"error": f"Column {group_by} not found"}
            counts = df.groupby(group_by).size().to_dict()
            return {"result": counts}
            
        elif operation == "describe":
            return {"result": df.describe().to_dict()}
            
        elif operation == "sort" and query: # using query arg as sort column for simplicity
            sorted_df = df.sort_values(by=query)
            return {"result": sorted_df.to_dict(orient="records")}
            
        else:
            return {"error": f"Unknown operation: {operation} or missing parameters"}
            
    except Exception as e:
        logger.error(f"Pandas transform error: {e}")
        return {"error": str(e)}

@expose_as_ccp_tool
def web_scraper_advanced(url: str) -> Dict[str, Any]:
    """
    Extracts main content from a URL using BeautifulSoup.
    
    Args:
        url (str): The URL to scrape.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'title': Page title.
            - 'text_content': Cleaned text content of the page (limited length).
            - 'links': List of main links found.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        # Limit text return to avoid context overflow issues
        cleaned_text = " ".join(text.split())[:5000]
        
        links = []
        for a in soup.find_all('a', href=True):
             link = a['href']
             if link.startswith('http'):
                 links.append(link)
        
        return {
            "title": soup.title.string if soup.title else "No Title",
            "text_content": cleaned_text,
            "links": links[:20] # Limit links
        }
        
    except Exception as e:
        logger.error(f"Scraper error: {e}")
        return {"error": str(e)}
