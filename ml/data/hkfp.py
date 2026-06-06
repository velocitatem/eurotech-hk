from __future__ import annotations
import os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

_BASE = "https://hongkongfp.com"
_ARCHIVE = f"{_BASE}/archive/"
_CACHE = Path("ml/data/processed/.cache/hkfp")
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
_CUTOFF_DAYS = 730


def _load_proxies() -> list[str | None]:
    """Read PROXY_LIST env var (comma-separated URLs). Falls back to [None] (direct)."""
    raw = os.environ.get("PROXY_LIST", "")
    ps = [p.strip() for p in raw.split(",") if p.strip()]
    return ps or [None]


def _page_url(n: int) -> str:
    return _ARCHIVE if n == 1 else f"{_ARCHIVE}page/{n}/"


def _cache_path(url: str, cache_dir: Path) -> Path:
    safe = (url.replace("https://", "").replace("http://", "")
               .replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_"))
    return cache_dir / f"{safe[:200]}.html"


def _cached_pages(cache_dir: Path) -> set[int]:
    pages: set[int] = set()
    if _cache_path(_page_url(1), cache_dir).exists():
        pages.add(1)
    for p in cache_dir.glob("hongkongfp.com_archive_page_*.html"):
        try:
            pages.add(int(p.stem.split("_page_")[1].rstrip("_")))
        except (IndexError, ValueError):
            pass
    return pages


def _total_pages(html: str) -> int | None:
    soup = BeautifulSoup(html, "html.parser")
    nums = [
        int(el.get_text(strip=True).replace(",", ""))
        for el in soup.select(".page-numbers")
        if not any(c in (el.get("class") or []) for c in ("next", "prev", "dots"))
        and el.get_text(strip=True).replace(",", "").isdigit()
    ]
    return max(nums) if nums else None


def _parse_articles(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for art in soup.select("article"):
        title_el = art.select_one(".entry-title a, h2 a, h3 a, .post-title a")
        date_el  = art.select_one("time[datetime], time")
        auth_el  = art.select_one("[rel='author'], .author a, .byline a, .post-author a")
        if not title_el:
            continue
        raw = (date_el.get("datetime") or date_el.get_text(strip=True)) if date_el else ""
        results.append({
            "title":  title_el.get_text(strip=True),
            "url":    title_el.get("href", ""),
            "date":   raw[:10] if raw else "",
            "author": auth_el.get_text(strip=True) if auth_el else "",
        })
    return results


def _oldest_date(html: str) -> date | None:
    dates = []
    for art in _parse_articles(html):
        try:
            if art["date"]:
                dates.append(date.fromisoformat(art["date"]))
        except ValueError:
            pass
    return min(dates) if dates else None


def _fetch_page(n: int, cache_dir: Path, proxy: str | None, retries: int = 5) -> str:
    url = _page_url(n)
    p = _cache_path(url, cache_dir)
    if p.exists():
        return p.read_text(encoding="utf-8")
    prx = {"http": proxy, "https": proxy} if proxy else None
    for attempt in range(retries):
        time.sleep(random.uniform(0.5, 1.5) + (2 ** attempt - 1))
        try:
            r = requests.get(url, headers={"User-Agent": _UA}, proxies=prx, timeout=25)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 15 * (2 ** attempt)))
                print(f"\n  429 page={n} proxy={proxy or 'direct'} — sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            p.write_text(r.text, encoding="utf-8")
            return r.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"\n  page {n} failed: {e}")
    return ""


def fetch_headlines(
    days: int = _CUTOFF_DAYS,
    cache_dir: Path = _CACHE,
    workers: int = 5,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cutoff = date.today() - timedelta(days=days)
    proxies = _load_proxies()

    p1_html = _fetch_page(1, cache_dir, proxies[0])
    total_avail = _total_pages(p1_html) or 9999

    cached = _cached_pages(cache_dir)
    max_cached = max(cached) if len(cached) > 1 else 1
    oldest_html_path = _cache_path(_page_url(max_cached), cache_dir)
    oldest_cached = (
        _oldest_date(oldest_html_path.read_text(encoding="utf-8"))
        if oldest_html_path.exists() and max_cached > 1 else None
    )

    if oldest_cached and (date.today() - oldest_cached).days > 7:
        rate = max_cached / (date.today() - oldest_cached).days  # pages/day
        upper = min(int(days * rate) + 50, total_avail)
    else:
        upper = min(600, total_avail)

    uncached = [n for n in range(1, upper + 1) if n not in cached]
    print(f"  upper={upper} | {len(cached)} cached | {len(uncached)} to fetch | {len(proxies)} proxies | {workers} workers")

    if uncached:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_fetch_page, n, cache_dir, proxies[i % len(proxies)]): n
                for i, n in enumerate(uncached)
            }
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"\n  page {futs[fut]}: {e}")
                done += 1
                print(f"  {done}/{len(uncached)} fetched", end="\r")
        print()

    records: list[dict] = []
    for n in range(1, upper + 1):
        p = _cache_path(_page_url(n), cache_dir)
        if not p.exists():
            continue
        html = p.read_text(encoding="utf-8")
        if not html:
            continue
        exhausted = False
        for art in _parse_articles(html):
            try:
                art_date = date.fromisoformat(art["date"]) if art["date"] else None
            except ValueError:
                art_date = None
            if art_date is not None and art_date < cutoff:
                exhausted = True
                break
            records.append(art)
        if exhausted:
            break

    cols = ["title", "url", "date", "author"]
    return pd.DataFrame(records, columns=cols) if records else pd.DataFrame(columns=cols)
