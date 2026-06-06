import hashlib
import pickle
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from opentelemetry.trace import Status, StatusCode
from seleniumbase import SB

from .tracing import get_tracer

_TRACER = get_tracer(__name__)


class ScraperCache:
    def __init__(self, cache_dir: str = ".scraper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, url: str) -> Optional[BeautifulSoup]:
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def set(self, url: str, soup: BeautifulSoup) -> None:
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(soup, f)
        except Exception:
            pass


_cache = ScraperCache()


def scrape_url(url: str, use_cache: bool = True) -> BeautifulSoup:
    parsed_url = urlparse(url)
    with _TRACER.start_as_current_span("dlib.scraper.scrape_url") as span:
        span.set_attribute("http.url", url)
        span.set_attribute("http.host", parsed_url.netloc)
        span.set_attribute("scraper.use_cache", use_cache)

        try:
            if use_cache:
                with _TRACER.start_as_current_span(
                    "dlib.scraper.cache_lookup"
                ) as cache_span:
                    cached_soup = _cache.get(url)
                    cache_hit = cached_soup is not None
                    cache_span.set_attribute("scraper.cache_hit", cache_hit)
                if cached_soup is not None:
                    span.set_attribute("scraper.cache_hit", True)
                    return cached_soup

            with _TRACER.start_as_current_span("dlib.scraper.fetch_page"):
                with SB(test=True, uc=True) as sb:
                    sb.open(url)
                    html = sb.get_page_source()

            with _TRACER.start_as_current_span("dlib.scraper.parse_html"):
                soup = BeautifulSoup(html, "html.parser")

            if use_cache:
                with _TRACER.start_as_current_span("dlib.scraper.cache_store"):
                    _cache.set(url, soup)

            span.set_attribute("scraper.cache_hit", False)
            return soup
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


if __name__ == "__main__":
    url = "https://httpbin.org/html"
    print("Testing scraper...")
    soup = scrape_url(url)
    print(f"Title: {soup.title.text if soup.title else 'No title'}")
    print(f"Found {len(soup.find_all('p'))} paragraphs")
    print("\nTesting cache...")
    scrape_url(url)
    print("Cache test completed")
