"""
Domain scraper - Autonomous web scraping for domain-specific content.
Enhanced for 200+ samples with parallel scraping, retry logic, and MongoDB integration.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from src.ccp.distillation.models import ScrapedData
from src.ccp.neural.training_config import ScraperConfig
from src.ccp.storage.mongo import MongoStorage

logger = logging.getLogger(__name__)


class DomainScraper:
    """Scrapes web sources for domain-specific content with parallel execution and MongoDB persistence."""
    
    def __init__(self, config: ScraperConfig = None, mongo_storage: Optional[MongoStorage] = None):
        self.config = config or ScraperConfig()
        self.headers = {'User-Agent': 'Mozilla/5.0 (compatible; CCPBot/1.0)'}
        self.mongo = mongo_storage  # Optional MongoDB storage
    
    async def scrape_domain(self, domain: str, subdomains: List[str], max_samples: int = None) -> List[ScrapedData]:
        """
        Scrape multiple sources for domain content with parallel execution.
        
        Args:
            domain: Main domain name
            subdomains: List of subdomains to scrape
            max_samples: Maximum samples (default: config.max_samples)
        
        Returns:
            List of scraped data (minimum 200 samples)
        """
        max_samples = max_samples or self.config.max_samples
        min_samples = self.config.min_samples
        
        logger.info(f"[DomainScraper] Scraping domain: {domain} (target: {max_samples}, min: {min_samples})")
        
        all_data = []
        
        # Parallel scraping from all sources
        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                # Wikipedia
                futures.append(executor.submit(self._scrape_wikipedia_sync, domain, subdomains, max_samples // 3))
                
                # StackOverflow
                futures.append(executor.submit(self._scrape_stackoverflow_sync, domain, subdomains, max_samples // 3))
                
                # Web search
                futures.append(executor.submit(self._scrape_web_search_sync, domain, subdomains, max_samples // 3))
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        all_data.extend(data)
                    except Exception as e:
                        logger.error(f"[DomainScraper] Scraping error: {e}")
        else:
            # Sequential scraping
            all_data.extend(await self.scrape_wikipedia(domain, subdomains, max_samples // 3))
            all_data.extend(await self.scrape_stackoverflow(domain, subdomains, max_samples // 3))
            all_data.extend(await self.scrape_web_search(domain, subdomains, max_samples // 3))
        
        # Ensure minimum samples
        if len(all_data) < min_samples:
            logger.warning(f"[DomainScraper] Only {len(all_data)} samples collected, retrying...")
            # Retry with increased limits
            additional_data = await self._retry_scraping(domain, subdomains, min_samples - len(all_data))
            all_data.extend(additional_data)
        
        logger.info(f"[DomainScraper] Collected {len(all_data)} samples for domain: {domain}")
        return all_data[:max_samples]
    
    def _scrape_wikipedia_sync(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Synchronous wrapper for Wikipedia scraping."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.scrape_wikipedia(domain, subdomains, limit))
    
    def _scrape_stackoverflow_sync(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Synchronous wrapper for StackOverflow scraping."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.scrape_stackoverflow(domain, subdomains, limit))
    
    def _scrape_web_search_sync(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Synchronous wrapper for web search scraping."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.scrape_web_search(domain, subdomains, limit))
    
    async def _retry_scraping(self, domain: str, subdomains: List[str], needed: int) -> List[ScrapedData]:
        """Retry scraping to meet minimum sample requirement."""
        data = []
        for attempt in range(self.config.max_retries):
            logger.info(f"[DomainScraper] Retry attempt {attempt + 1}/{self.config.max_retries}")
            
            # Try each source again
            data.extend(await self.scrape_wikipedia(domain, subdomains, needed // 3))
            data.extend(await self.scrape_stackoverflow(domain, subdomains, needed // 3))
            data.extend(await self.scrape_web_search(domain, subdomains, needed // 3))
            
            if len(data) >= needed:
                break
            
            await asyncio.sleep(self.config.retry_delay)
        
        return data
    
    async def scrape_wikipedia(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Scrape Wikipedia articles for domain knowledge."""
        data = []
        
        try:
            # Search Wikipedia for domain and subdomains
            search_terms = [domain] + subdomains[:5]
            
            for term in search_terms:
                search_url = f"https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "opensearch",
                    "search": term,
                    "limit": min(limit // len(search_terms), 10),
                    "format": "json"
                }
                
                response = requests.get(search_url, params=params, headers=self.headers, timeout=self.config.timeout)
                results = response.json()
                
                if len(results) >= 4:
                    titles = results[1]
                    urls = results[3]
                    
                    for title, url in zip(titles, urls):
                        try:
                            # Get article content
                            article_response = requests.get(url, headers=self.headers, timeout=self.config.timeout)
                            soup = BeautifulSoup(article_response.text, 'html.parser')
                            
                            # Extract main content
                            content_div = soup.find('div', {'id': 'mw-content-text'})
                            if content_div:
                                # Get paragraphs
                                paragraphs = content_div.find_all('p', limit=10)
                                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                                
                                if len(content) > 100:  # Only substantial content
                                    data.append(ScrapedData(
                                        source="wikipedia",
                                        url=url,
                                        title=title,
                                        content=content,
                                        metadata={"domain": domain, "subdomain": term}
                                    ))
                        except Exception as e:
                            logger.warning(f"Error scraping Wikipedia article {url}: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Error scraping Wikipedia for {domain}: {e}")
        
        return data
    
    async def scrape_stackoverflow(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Scrape StackOverflow Q&A for domain."""
        data = []
        
        try:
            # Search for domain and subdomains
            search_terms = [domain] + subdomains[:3]
            
            for term in search_terms:
                # Use StackExchange API
                api_url = "https://api.stackexchange.com/2.3/search"
                params = {
                    "order": "desc",
                    "sort": "votes",
                    "intitle": term,
                    "site": "stackoverflow",
                    "pagesize": min(limit // len(search_terms), 30)
                }
                
                response = requests.get(api_url, params=params, headers=self.headers, timeout=self.config.timeout)
                results = response.json()
                
                if "items" in results:
                    for item in results["items"]:
                        question_id = item.get("question_id")
                        title = item.get("title", "")
                        link = item.get("link", "")
                        
                        # Get question body
                        try:
                            question_url = f"https://api.stackexchange.com/2.3/questions/{question_id}"
                            q_params = {"site": "stackoverflow", "filter": "withbody"}
                            q_response = requests.get(question_url, params=q_params, headers=self.headers, timeout=self.config.timeout)
                            q_data = q_response.json()
                            
                            if "items" in q_data and len(q_data["items"]) > 0:
                                body = q_data["items"][0].get("body", "")
                                
                                # Clean HTML
                                soup = BeautifulSoup(body, 'html.parser')
                                content = soup.get_text(strip=True)
                                
                                if len(content) > 50:
                                    data.append(ScrapedData(
                                        source="stackoverflow",
                                        url=link,
                                        title=title,
                                        content=content[:2000],  # Limit content length
                                        metadata={"domain": domain, "subdomain": term, "question_id": question_id}
                                    ))
                        except Exception as e:
                            logger.warning(f"Error getting question {question_id}: {e}")
                            continue
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error scraping StackOverflow for {domain}: {e}")
        
        return data
    
    async def scrape_web_search(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Scrape general web search results."""
        data = []
        
        try:
            # Search for domain and subdomains
            search_terms = [f"{domain} tutorial", f"{domain} guide"] + [f"{s} {domain}" for s in subdomains[:3]]
            
            for term in search_terms[:limit // 10]:
                # Use DuckDuckGo HTML search
                search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(term)}"
                
                response = requests.get(search_url, headers=self.headers, timeout=self.config.timeout)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = soup.select('.result')[:10]
                
                for result in results:
                    title_elem = result.select_one('.result__title')
                    snippet_elem = result.select_one('.result__snippet')
                    url_elem = result.select_one('.result__url')
                    
                    if title_elem and snippet_elem:
                        content = snippet_elem.get_text(strip=True)
                        if len(content) > 50:
                            data.append(ScrapedData(
                                source="web_search",
                                url=url_elem.get_text(strip=True) if url_elem else "",
                                title=title_elem.get_text(strip=True),
                                content=content,
                                metadata={"domain": domain, "search_query": term}
                            ))
                
                await asyncio.sleep(0.5)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Error in web search for {domain}: {e}")
        
        return data

        """
        Scrape multiple sources for domain content.
        
        Args:
            domain: Main domain name
            subdomains: List of subdomains to scrape
            max_samples: Maximum samples to collect
        
        Returns:
            List of scraped data
        """
        all_data = []
        samples_per_source = max_samples // 3  # Distribute across sources
        
        # Scrape Wikipedia
        wiki_data = await self.scrape_wikipedia(domain, subdomains, samples_per_source)
        all_data.extend(wiki_data)
        
        # Scrape StackOverflow
        so_data = await self.scrape_stackoverflow(domain, subdomains, samples_per_source)
        all_data.extend(so_data)
        
        # Scrape general web
        web_data = await self.scrape_web_search(domain, subdomains, samples_per_source)
        all_data.extend(web_data)
        
        logger.info(f"[DomainScraper] Collected {len(all_data)} samples for domain: {domain}")
        return all_data[:max_samples]
    
    async def scrape_wikipedia(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Scrape Wikipedia articles for domain knowledge."""
        data = []
        
        try:
            # Search Wikipedia for domain
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch",
                "search": domain,
                "limit": min(limit, 10),
                "format": "json"
            }
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=10)
            results = response.json()
            
            if len(results) >= 4:
                titles = results[1]
                urls = results[3]
                
                for title, url in zip(titles, urls):
                    try:
                        # Get article content
                        article_response = requests.get(url, headers=self.headers, timeout=10)
                        soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        # Extract main content
                        content_div = soup.find('div', {'id': 'mw-content-text'})
                        if content_div:
                            # Get paragraphs
                            paragraphs = content_div.find_all('p', limit=5)
                            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                            
                            data.append(ScrapedData(
                                source="wikipedia",
                                url=url,
                                title=title,
                                content=content,
                                metadata={"domain": domain}
                            ))
                    except Exception as e:
                        logger.warning(f"Error scraping Wikipedia article {url}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error scraping Wikipedia for {domain}: {e}")
        
        return data
    
    async def scrape_stackoverflow(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Scrape StackOverflow Q&A for domain."""
        data = []
        
        try:
            # Use StackExchange API
            api_url = "https://api.stackexchange.com/2.3/search"
            params = {
                "order": "desc",
                "sort": "votes",
                "intitle": domain,
                "site": "stackoverflow",
                "pagesize": min(limit, 30)
            }
            
            response = requests.get(api_url, params=params, headers=self.headers, timeout=10)
            results = response.json()
            
            if "items" in results:
                for item in results["items"]:
                    # Get question details
                    question_id = item.get("question_id")
                    title = item.get("title", "")
                    link = item.get("link", "")
                    
                    # Get question body
                    try:
                        question_url = f"https://api.stackexchange.com/2.3/questions/{question_id}"
                        q_params = {"site": "stackoverflow", "filter": "withbody"}
                        q_response = requests.get(question_url, params=q_params, headers=self.headers, timeout=10)
                        q_data = q_response.json()
                        
                        if "items" in q_data and len(q_data["items"]) > 0:
                            body = q_data["items"][0].get("body", "")
                            
                            # Clean HTML
                            soup = BeautifulSoup(body, 'html.parser')
                            content = soup.get_text(strip=True)
                            
                            data.append(ScrapedData(
                                source="stackoverflow",
                                url=link,
                                title=title,
                                content=content[:1000],  # Limit content length
                                metadata={"domain": domain, "question_id": question_id}
                            ))
                    except Exception as e:
                        logger.warning(f"Error getting question {question_id}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error scraping StackOverflow for {domain}: {e}")
        
        return data
    
    async def scrape_web_search(self, domain: str, subdomains: List[str], limit: int) -> List[ScrapedData]:
        """Scrape general web search results."""
        data = []
        
        try:
            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(domain + ' tutorial')}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = soup.select('.result')[:limit]
            
            for result in results:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                url_elem = result.select_one('.result__url')
                
                if title_elem and snippet_elem and url_elem:
                    data.append(ScrapedData(
                        source="web_search",
                        url=url_elem.get_text(strip=True),
                        title=title_elem.get_text(strip=True),
                        content=snippet_elem.get_text(strip=True),
                        metadata={"domain": domain, "search_query": domain + " tutorial"}
                    ))
        
        except Exception as e:
            logger.error(f"Error in web search for {domain}: {e}")
        
        return data
