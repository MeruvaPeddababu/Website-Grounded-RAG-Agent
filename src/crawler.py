"""
Crawler module - scrapes publicly accessible websites and extracts clean content.

Two strategies:
  1. MintlifyCrawler - for SPA docs sites (Mintlify/Fumadocs) that expose
     llms.txt markdown indexes (e.g. python.langchain.com). Fetches clean
     markdown directly, bypassing JS-rendering problems.
  2. GenericCrawler - classic HTML BFS crawl with BeautifulSoup extraction.

WebsiteCrawler auto-detects which strategy to use.
"""
import re
import time
import hashlib
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langsmith import traceable


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    headings: list[str] = field(default_factory=list)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.url.encode()).hexdigest()[:12]


class GenericCrawler:
    """BFS HTML crawler with content extraction."""

    def __init__(
        self,
        base_url: str,
        max_pages: int = 50,
        delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.exclude_patterns = [
            r"/api/", r"/_", r"\.json$", r"\.xml$", r"\.css$", r"\.js$",
            r"\.png$", r"\.jpg$", r"\.svg$", r"#$", r"\.pdf$",
        ]
        self.visited: set[str] = set()
        self.pages: list[PageContent] = []
        self.session = self._make_session()

    @staticmethod
    def _make_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": "RAGBot/1.0 (educational project)"})
        return session

    def _is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != self.domain:
            return False
        for pattern in self.exclude_patterns:
            if re.search(pattern, url):
                return False
        return True

    def _extract_text(self, soup: BeautifulSoup) -> tuple[str, list[str]]:
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        headings = []
        for h in soup.find_all(re.compile(r"^h[1-6]$")):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)

        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main is None:
            main = soup

        text = main.get_text(separator="\n", strip=True)
        # Strip lines that are just raw URLs
        lines = [
            line for line in text.splitlines()
            if not re.match(r"^\s*https?://\S+\s*$", line)
        ]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip(), headings

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(current_url, href).split("#")[0]
            if self._is_valid_url(full_url) and full_url not in self.visited:
                links.append(full_url)
        return links

    @traceable(
        name="scrape_html",
        run_type="tool",
        process_outputs=lambda pages: {
            "pages_scraped": len(pages),
            "urls": [p.url for p in pages],
        },
    )
    def crawl(self) -> list[PageContent]:
        queue = [self.base_url]

        while queue and len(self.visited) < self.max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue

            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[SKIP] {url}: {e}")
                self.visited.add(url)
                continue

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                self.visited.add(url)
                continue

            self.visited.add(url)
            soup = BeautifulSoup(resp.text, "lxml")

            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            text, headings = self._extract_text(soup)
            if len(text) < 200:
                continue

            page = PageContent(url=url, title=title, text=text, headings=headings)
            self.pages.append(page)
            print(f"[OK] ({len(self.pages)}) {title[:60]} - {url}")

            queue.extend(self._extract_links(soup, url))
            time.sleep(self.delay)

        print(f"\nHTML crawl complete: {len(self.pages)} pages.")
        return self.pages


class MintlifyCrawler:
    """Crawler for Mintlify-based docs that expose llms.txt indexes."""

    def __init__(
        self,
        base_url: str,
        max_pages: int = 100,
        delay: float = 0.1,
        section_prefix: str | None = None,
        preferred_sections: list[str] | None = None,
        include_urls: list[str] | None = None,
        skip_patterns: list[str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        self.delay = delay
        # Only crawl sections under this prefix (e.g. "/oss/python").
        # If None, crawl the root index sections.
        self.section_prefix = section_prefix
        # Sections to crawl first (moved to front of queue).
        self.preferred_sections = preferred_sections or []
        # Specific pages to always crawl first, regardless of index order.
        self.include_urls = include_urls or []
        # URL patterns to skip (noise pages like changelogs).
        self.skip_patterns = skip_patterns or [
            r"changelog",
            r"release-notes",
            r"errors/",
        ]
        self.pages: list[PageContent] = []
        self.session = self._make_session()

    @staticmethod
    def _make_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": "RAGBot/1.0 (educational project)"})
        return session

    @staticmethod
    def _parse_llms_index(text: str) -> list[str]:
        """Extract markdown (.md) page URLs from an llms.txt index."""
        urls = []
        for line in text.splitlines():
            m = re.search(r"\((https?://[^)]+\.md)\)", line)
            if m:
                urls.append(m.group(1))
            else:
                m = re.search(r"https?://\S+\.md", line)
                if m:
                    urls.append(m.group(1).rstrip(")"))
        # Dedupe preserving order
        seen = set()
        unique = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    def _fetch_llms_index(self, url: str) -> str | None:
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200 and "<!DOCTYPE" not in resp.text[:200]:
                return resp.text
        except requests.RequestException:
            pass
        return None

    @staticmethod
    def _parse_section_urls(text: str, base_url: str) -> list[str]:
        """Extract sub-section llms.txt URLs from an index."""
        urls = []
        for line in text.splitlines():
            m = re.search(r"\((\S+/llms\.txt)\)", line)
            if m:
                url = m.group(1).split(")")[0]
                if not url.startswith("http"):
                    url = base_url.rstrip("/") + "/" + url.lstrip("/")
                urls.append(url)
        return urls

    def _collect_all_page_urls(self) -> list[str]:
        """BFS over all llms.txt indexes to collect every .md page URL."""
        seen_indexes: set[str] = set()
        page_urls: list[str] = []
        seen_pages: set[str] = set()

        queue = [f"{self.base_url}/llms.txt"]

        while queue:
            index_url = queue.pop(0)
            if index_url in seen_indexes:
                continue
            seen_indexes.add(index_url)

            text = self._fetch_llms_index(index_url)
            if not text:
                continue

            # Collect .md page URLs from this index
            for url in self._parse_llms_index(text):
                if url not in seen_pages:
                    seen_pages.add(url)
                    page_urls.append(url)

            # Enqueue sub-section indexes (BFS deeper)
            for sub_url in self._parse_section_urls(text, self.base_url):
                if sub_url not in seen_indexes:
                    queue.append(sub_url)

            time.sleep(self.delay)

        return page_urls

    @traceable(
        name="scrape_mintlify",
        run_type="tool",
        process_outputs=lambda pages: {
            "pages_scraped": len(pages),
            "urls": [p.url for p in pages],
        },
    )
    def crawl(self) -> list[PageContent]:
        # 1. Try llms-full.txt first (flat complete index — all pages in one file)
        page_urls: list[str] = []
        full_index = self._fetch_llms_index(f"{self.base_url}/llms-full.txt")
        if full_index:
            print("[Mintlify] Using llms-full.txt (complete flat index).")
            page_urls = self._parse_llms_index(full_index)

        # 2. Fall back: BFS over all nested llms.txt section indexes
        if not page_urls:
            root_index = self._fetch_llms_index(f"{self.base_url}/llms.txt")
            if not root_index:
                print("[Mintlify] No llms.txt found; nothing to crawl.")
                return []
            print("[Mintlify] BFS crawling all section indexes...")
            page_urls = self._collect_all_page_urls()

        if not page_urls:
            print("[Mintlify] No markdown pages found in any index.")
            return []

        # 3. Filter by section_prefix if set
        if self.section_prefix:
            page_urls = [u for u in page_urls if self.section_prefix in u]

        # 4. Dedupe
        seen: set[str] = set()
        unique_pages: list[str] = []
        for u in page_urls:
            if u not in seen:
                seen.add(u)
                unique_pages.append(u)

        # 5. Force-schedule high-value pages first
        forced, remaining, forced_set = [], [], set()
        for u in unique_pages:
            if any(inc in u for inc in self.include_urls):
                if u not in forced_set:
                    forced_set.add(u)
                    forced.append(u)
            else:
                remaining.append(u)
        unique_pages = forced + remaining

        # 6. Apply preferred_sections ordering
        if self.preferred_sections:
            def section_rank(url: str) -> int:
                for i, p in enumerate(self.preferred_sections):
                    if p in url:
                        return i
                return len(self.preferred_sections)
            unique_pages.sort(key=section_rank)

        unique_pages = unique_pages[: self.max_pages]

        print(f"[Mintlify] {len(unique_pages)} pages to fetch.")

        # 4. Fetch each markdown page
        for url in unique_pages:
            # Skip noise pages (changelogs, release notes, error codes)
            if any(re.search(p, url, re.IGNORECASE) for p in self.skip_patterns):
                print(f"[SKIP] {url} (noise page)")
                continue

            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                md_text = resp.text

                # Strip HTML comments
                md_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.DOTALL)

                # Clean markdown: convert [text](url) → text, strip bare URLs
                md_text = self._clean_markdown(md_text)

                # Extract title from first # heading
                title_match = re.search(r"^#\s+(.+)$", md_text, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else url.split("/")[-1]

                # Extract headings for metadata
                headings = re.findall(r"^#{1,6}\s+(.+)$", md_text, re.MULTILINE)
                headings = [h.strip() for h in headings]

                if len(md_text) < 200:
                    continue

                # Convert to html-friendly URL for display (strip .md)
                display_url = re.sub(r"\.md$", "", url)

                page = PageContent(
                    url=display_url,
                    title=title,
                    text=md_text.strip(),
                    headings=headings,
                )
                self.pages.append(page)
                print(f"[OK] ({len(self.pages)}) {title[:60]} - {display_url}")
                time.sleep(self.delay)

            except requests.RequestException as e:
                print(f"[SKIP] {url}: {e}")
                continue

        print(f"\nMintlify crawl complete: {len(self.pages)} pages.")
        return self.pages

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Strip URLs from markdown so LLM answers aren't polluted with raw links."""
        # [label](url) → label
        text = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", text)
        # ![alt](url) → alt
        text = re.sub(r"!\[([^\]]*)\]\(https?://[^\)]+\)", r"\1", text)
        # bare URLs on their own line → remove
        text = re.sub(r"^\s*https?://\S+\s*$", "", text, flags=re.MULTILINE)
        # inline bare URLs not in code spans → remove
        text = re.sub(r"(?<![`(])https?://\S+", "", text)
        # collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class WebsiteCrawler:
    """Auto-detecting crawler: uses Mintlify strategy when llms.txt exists."""

    def __init__(
        self,
        base_url: str,
        max_pages: int = 50,
        delay: float = 0.5,
        section_prefix: str | None = None,
        preferred_sections: list[str] | None = None,
        include_urls: list[str] | None = None,
    ):
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.section_prefix = section_prefix
        self.preferred_sections = preferred_sections or []
        self.include_urls = include_urls or []

    @staticmethod
    def _has_llms_txt(base_url: str) -> bool:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/llms.txt", timeout=15)
            return resp.status_code == 200 and "<!DOCTYPE" not in resp.text[:200]
        except requests.RequestException:
            return False

    @traceable(
        name="website_crawl",
        run_type="chain",
        process_outputs=lambda pages: {
            "pages_scraped": len(pages),
            "urls": [p.url for p in pages],
        },
    )
    def crawl(self) -> list[PageContent]:
        if self._has_llms_txt(self.base_url):
            print(f"[Crawler] Detected Mintlify-style docs at {self.base_url}")
            crawler = MintlifyCrawler(
                base_url=self.base_url,
                max_pages=self.max_pages,
                delay=self.delay,
                section_prefix=self.section_prefix,
                preferred_sections=self.preferred_sections,
                include_urls=self.include_urls,
            )
            return crawler.crawl()

        print(f"[Crawler] Using generic HTML crawl for {self.base_url}")
        crawler = GenericCrawler(
            base_url=self.base_url,
            max_pages=self.max_pages,
            delay=self.delay,
        )
        return crawler.crawl()
