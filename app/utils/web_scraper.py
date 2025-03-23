import requests
from bs4 import BeautifulSoup
import os
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def scrape_website(url: str, output_dir: str) -> Optional[str]:
    """
    Scrape content from a website and save it as a text file.
    
    Args:
        url: The URL to scrape
        output_dir: Directory to save the output file
    
    Returns:
        Path to the saved file if successful, None otherwise
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
            
        # Create filename from URL
        domain = parsed_url.netloc.replace("www.", "")
        path = parsed_url.path.strip("/").replace("/", "_")
        if not path:
            path = "homepage"
        filename = f"{domain}_{path}.txt"
        
        # Request the page
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style elements and comments
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Extract text
        text = soup.get_text(separator='\n')
        
        # Clean up text (remove excessive newlines)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n\n".join(lines)
        
        # Save to file
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Source URL: {url}\n\n")
            f.write(cleaned_text)
            
        logger.info(f"Successfully scraped and saved {url} to {filename}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None