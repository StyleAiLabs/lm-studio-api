import requests
from bs4 import BeautifulSoup
import os
import logging
import time
import hashlib
from datetime import datetime
from typing import Optional, Set, List
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


class WebsiteScrapeForbidden(Exception):
    """Raised when a website returns forbidden (403) after retries."""
    pass

MAX_TOTAL_BYTES = 250_000  # Limit combined text size (~250 KB)
MAX_PAGE_BYTES = 120_000   # Per page cap
BOILERPLATE_KEYWORDS = [
    'privacy policy', 'terms of use', 'terms & conditions', 'copyright', 'all rights reserved',
    'cookie policy', 'login', 'sign in', 'create account', 'subscribe', 'newsletter'
]

def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    # Remove nav/footer elements when identifiable
    for selector in ["nav", "footer", "header", "aside"]:
        for el in soup.select(selector):
            el.decompose()
    text = soup.get_text(separator='\n')
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        # Filter boilerplate lines (very short nav items or keyword lines)
        if any(k in lower for k in BOILERPLATE_KEYWORDS) and len(stripped) < 120:
            continue
        lines.append(stripped)
    cleaned = "\n\n".join(lines)
    if len(cleaned.encode('utf-8')) > MAX_PAGE_BYTES:
        # Truncate softly at paragraph boundaries
        acc = []
        total = 0
        for para in cleaned.split("\n\n"):
            b = len(para.encode('utf-8')) + 2
            if total + b > MAX_PAGE_BYTES:
                break
            acc.append(para)
            total += b
        cleaned = "\n\n".join(acc)
    return cleaned

def _hash_dom(html: str) -> str:
    # Hash without whitespace extremes to detect near duplicates quickly
    return hashlib.sha256(' '.join(html.split()).encode('utf-8')).hexdigest()

def _fetch(url: str, headers: dict) -> Optional[str]:
    max_attempts = 3
    backoff = 2
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 403:
                if attempt == max_attempts:
                    logger.error(f"Persistent 403 after {attempt} attempts for {url}")
                    raise WebsiteScrapeForbidden(f"Access forbidden (403) when scraping {url}")
                logger.warning(f"Attempt {attempt}/3 403 for {url}; retrying in {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.text
        except WebsiteScrapeForbidden:
            raise
        except requests.RequestException as re:
            if attempt == max_attempts:
                logger.error(f"Failed fetching {url} after {attempt} attempts: {re}")
                return None
            logger.warning(f"Error fetching {url} (attempt {attempt}/3): {re}; retry {backoff}s")
            time.sleep(backoff)
            backoff *= 2
    return None

def scrape_website(url: str, output_dir: str, depth: int = 1) -> Optional[str]:
    """Scrape a root URL plus same-domain links up to depth=1.

    Returns path to aggregated text file or None.
    Raises WebsiteScrapeForbidden for persistent 403 on root.
    """
    # Validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        logger.error(f"Invalid URL format: {url}")
        return None

    # Build deterministic filename
    domain = parsed_url.netloc.replace("www.", "")
    path = parsed_url.path.strip("/").replace("/", "_") or "homepage"
    filename = f"{domain}_{path}.txt"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    }

    root_html = _fetch(url, headers)
    if root_html is None:
        return None

    visited_hashes: Set[str] = set()
    aggregated_parts: List[str] = []
    total_bytes = 0

    def consider(html: str, page_url: str):
        nonlocal total_bytes
        dom_hash = _hash_dom(html)
        if dom_hash in visited_hashes:
            logger.info(f"Skipping duplicate DOM {page_url}")
            return
        cleaned = _clean_text(html)
        size = len(cleaned.encode('utf-8'))
        if size == 0:
            return
        if total_bytes + size > MAX_TOTAL_BYTES:
            logger.info(f"Size cap reached; skipping remaining content at {page_url}")
            return
        visited_hashes.add(dom_hash)
        total_bytes += size
        aggregated_parts.append(f"\n\n===== PAGE: {page_url} =====\n\n{cleaned}")

    consider(root_html, url)

    # Crawl same-domain links (depth 1 only)
    if depth >= 1 and total_bytes < MAX_TOTAL_BYTES:
        soup = BeautifulSoup(root_html, 'html.parser')
        root_domain = parsed_url.netloc
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if href.startswith('#') or href.lower().startswith('javascript:'):
                continue
            full = urljoin(url, href)
            parsed_link = urlparse(full)
            if parsed_link.netloc != root_domain:
                continue
            # Limit to same scheme and avoid binary assets
            if any(full.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.svg', '.zip']):
                continue
            links.add(full.split('#')[0])
            if len(links) >= 12:  # small cap
                break
        for link in links:
            if total_bytes >= MAX_TOTAL_BYTES:
                break
            html = _fetch(link, headers)
            if not html:
                continue
            consider(html, link)

    # Persist
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            meta = {
                'root_url': url,
                'fetched_at': datetime.utcnow().isoformat() + 'Z',
                'pages': len(aggregated_parts),
                'total_bytes': total_bytes,
                'depth': depth,
            }
            f.write("METADATA::" + str(meta) + "\n\n")
            f.write('\n'.join(aggregated_parts))
    except OSError as oe:
        logger.error(f"Failed writing scraped file {file_path}: {oe}")
        return None

    logger.info(f"Successfully scraped and saved {url} to {filename} ({total_bytes} bytes, {len(aggregated_parts)} pages)")
    return file_path