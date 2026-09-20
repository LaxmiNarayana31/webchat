import concurrent.futures
import json
import re
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
import requests
import trafilatura

from backend.app.core.config import settings
from backend.app.core.logging import logger





class ScraperService:
    """Multi-tier scraper bypassing soft paywalls and extracting article contents."""

    PAYWALL_INDICATORS = [
        "member-only story",
        "members-only story",
        "sign up to continue reading",
        "create an account to read",
        "you've read your free articles",
        "you have reached your free article limit",
        "subscribe to read the full story",
        "subscribe to continue",
        "upgrade to a paid subscription",
        "this content is for subscribers only",
        "unlock this story",
        "read the rest of this story with a free account",
        "register for free to continue",
    ]

    PAYWALL_CSS_CLASSES = [
        "paywall",
        "meteredContent",
        "gate-content",
        "subscription-gate",
        "signup-prompt",
        "modal-backdrop",
        "blur-overlay",
        "article-gate",
        "subscriber-only",
        "membership-prompt",
    ]

    def __init__(self):
        """Initializes scraper service headers and request timeout settings."""
        try:
            self.timeout = settings.SCRAPER_TIMEOUT
            self.default_headers = {
                "User-Agent": settings.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Sec-Ch-Ua": '"Chromium";v="133", "Google Chrome";v="133", "Not?A_Brand";v="99"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                # Social / Search referer exemption: grants free access on many news and Medium sites
                "Referer": "https://www.google.com/",
            }
        except Exception as e:
            logger.error(f"Error initializing ScraperService: {e}", exc_info=True)
            raise

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validates format and scheme of target URL."""
        try:
            if not url or not isinstance(url, str):
                return False, "Please provide a valid URL."

            url = url.strip()
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format. Please include http:// or https://"

            if parsed.scheme not in ["http", "https"]:
                return False, "URL protocol must be http or https."

            return True, ""
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}", exc_info=True)
            return False, f"URL validation failed: {e}"

    def scrape_url(self, url: str, strategy: str = "auto") -> Dict[str, Any]:
        """Extracts content using the optimal strategy, cascading through fallbacks."""
        try:
            is_valid, err = self.validate_url(url)
            if not is_valid:
                return {
                    "success": False,
                    "error": err,
                    "url": url,
                    "title": "",
                    "content": "",
                    "word_count": 0,
                    "strategy_used": "none",
                    "paywall_detected": False,
                    "paywall_bypassed": False,
                }

            url = url.strip()
            is_medium = self._is_medium_domain(url)

            # Execution order for 'auto':
            # Medium specific (if medium domain)
            # Direct stealth fetch with referer spoofing
            # Jina AI Reader API (if paywall or direct fails)
            # Wayback Machine Archive fallback

            direct_res = {}
            if strategy == "medium" or (strategy == "auto" and is_medium):
                # Jina Reader bypasses Medium Cloudflare TLS anti-bot and extracts full markdown with diagrams
                if settings.JINA_READER_ENABLED:
                    jina_res = self._scrape_jina_reader(url)
                    if jina_res.get("success") and jina_res.get("word_count", 0) > 150:
                        jina_res["paywall_bypassed"] = True
                        return jina_res

                medium_res = self._scrape_medium_apollo(url)
                if medium_res.get("success") and medium_res.get("word_count", 0) > 200:
                    return medium_res

            if strategy in ["auto", "direct"]:
                direct_res = self._scrape_direct_stealth(url)
                # If direct extraction got substantial content (>300 words), it's the full article regardless of footer signup prompts
                if direct_res.get("success") and direct_res.get("word_count", 0) > 300:
                    return direct_res
                # If direct worked and no paywall detected, return it
                if direct_res.get("success") and not direct_res.get("paywall_detected") and direct_res.get("word_count", 0) >= 10:
                    return direct_res

                # If paywall was detected or content was suspiciously short, continue to fallback
                if direct_res.get("paywall_detected") or not direct_res.get("success"):
                    logger.info(f"Direct scrape encountered paywall or low content for {url}. Attempting fallbacks...")

            # Freedium proxy for Medium articles (bypasses member-only paywall)
            if is_medium and strategy in ["auto", "medium"]:
                freedium_res = self._scrape_freedium(url)
                if freedium_res.get("success") and freedium_res.get("word_count", 0) > 150:
                    return freedium_res

            # Jina AI Reader (handles JS rendering, sign-in modals, soft paywalls)
            if settings.JINA_READER_ENABLED and strategy in ["auto", "jina", "direct"]:
                jina_res = self._scrape_jina_reader(url)
                if jina_res.get("success") and jina_res.get("word_count", 0) > 150:
                    # Check if it bypassed paywall
                    jina_res["paywall_bypassed"] = True
                    return jina_res

            # Google Webcache fallback
            if strategy in ["auto"]:
                cache_res = self._scrape_google_cache(url)
                if cache_res.get("success") and cache_res.get("word_count", 0) > 150:
                    cache_res["paywall_bypassed"] = True
                    return cache_res

            # Wayback Machine Archival fallback
            if settings.ARCHIVE_FALLBACK_ENABLED and strategy in ["auto", "archive"]:
                archive_res = self._scrape_wayback_archive(url)
                if archive_res.get("success") and archive_res.get("word_count", 0) > 150:
                    archive_res["paywall_bypassed"] = True
                    return archive_res

            # Fallback to direct result even if partial, or return descriptive error
            if 'direct_res' in locals() and direct_res.get("success"):
                return direct_res

            return {
                "success": False,
                "error": "Failed to bypass site protection or extract readable article content.",
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "strategy_used": "failed",
                "paywall_detected": True,
                "paywall_bypassed": False,
            }
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Scraping error: {str(e)}",
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "strategy_used": "failed",
                "paywall_detected": False,
                "paywall_bypassed": False,
            }

    def _is_medium_domain(self, url: str) -> bool:
        """Checks if URL belongs to Medium or Medium-hosted publication network."""
        try:
            domain = urlparse(url).netloc.lower()
            return (
                "medium.com" in domain
                or "towardsdatascience.com" in domain
                or "betterprogramming.pub" in domain
                or "levelup.gitconnected.com" in domain
                or "plainenglish.io" in domain
            )
        except Exception as e:
            logger.debug(f"Error checking medium domain for {url}: {e}")
            return False

    def _extract_content_images(self, soup: BeautifulSoup, base_url: str, max_images: int = 20) -> List[Dict[str, str]]:
        """Extracts non-decorative content images (diagrams, photos, charts) with absolute URLs and captions."""
        extracted_images = []
        seen_urls = set()

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            # Handle modern responsive picture and srcset tags (used by Medium, Substack, etc.)
            if not src:
                srcset = img.get("srcset") or img.get("data-srcset")
                if not srcset:
                    picture = img.find_parent("picture")
                    if picture:
                        source = picture.find("source")
                        if source:
                            srcset = source.get("srcset") or source.get("data-srcset")
                if srcset:
                    parts = [p.strip().split()[0] for p in srcset.split(",") if p.strip()]
                    if parts:
                        src = parts[-1]

            if not src or src.startswith("data:"):
                continue

            abs_url = urljoin(base_url, src)
            if abs_url in seen_urls:
                continue

            lower_url = abs_url.lower()
            # Filter out UI chrome, avatars, icons, badges, and small thumbnails
            if any(bad in lower_url for bad in [
                "avatar", "logo", "icon", "favicon", "pixel", "badge", "button", "sprite", ".svg",
                "resize:fill",
            ]):
                continue

            # Filter out small dimensions if specified
            w = img.get("width")
            h = img.get("height")
            try:
                if w and int(w) < 120 and h and int(h) < 120:
                    continue
            except (ValueError, TypeError):
                pass

            alt_text = (img.get("alt") or "").strip()
            title_text = (img.get("title") or "").strip()

            parent_caption = ""
            parent_fig = img.find_parent(["figure", "div", "p"])
            if parent_fig:
                fig_caption = parent_fig.find(["figcaption", "span", "p"])
                if fig_caption:
                    parent_caption = fig_caption.get_text().strip()
                if not parent_caption:
                    # Look at adjacent heading or paragraph to infer diagram context
                    prev_sib = parent_fig.find_previous_sibling(["h1", "h2", "h3", "h4", "p"])
                    if prev_sib:
                        parent_caption = prev_sib.get_text().strip()[:200]
                    else:
                        next_sib = parent_fig.find_next_sibling(["h1", "h2", "h3", "h4", "p"])
                        if next_sib:
                            parent_caption = next_sib.get_text().strip()[:200]

            combined_context = f"{alt_text} {title_text} {parent_caption}".strip()

            seen_urls.add(abs_url)
            extracted_images.append({
                "url": abs_url,
                "alt": alt_text or (parent_caption[:60] if parent_caption else "Content Diagram/Image"),
                "context": combined_context or "Visual content diagram",
            })

            if len(extracted_images) >= max_images:
                break

        return extracted_images

    def _scrape_direct_stealth(self, url: str) -> Dict[str, Any]:
        """Direct request using stealth headers and referer spoofing."""
        try:
            req_headers = dict(self.default_headers)
            req_headers["Connection"] = "close"
            # Try with Google referer first
            resp = requests.get(url, headers=req_headers, timeout=self.timeout, allow_redirects=True)
            if resp.status_code >= 400:
                # Try Twitter/X referer if Google referer failed
                alt_headers = dict(req_headers)
                alt_headers["Referer"] = "https://t.co/"
                resp = requests.get(url, headers=alt_headers, timeout=self.timeout, allow_redirects=True)
                if resp.status_code >= 400:
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status_code}",
                        "paywall_detected": resp.status_code in [401, 403, 429],
                    }

            html = resp.text
            soup = BeautifulSoup(html, "lxml")

            # Extract content images before removing tags
            content_images = self._extract_content_images(soup, url)

            # Remove unwanted tags
            for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript", "aside"]):
                tag.decompose()

            # Remove known paywall and modal elements
            for cls_name in self.PAYWALL_CSS_CLASSES:
                for el in soup.find_all(attrs={"class": re.compile(cls_name, re.I)}):  # type: ignore
                    el.decompose()

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            # Detect paywall text in body
            raw_text = soup.get_text()
            paywall_detected = any(ind in raw_text.lower() for ind in self.PAYWALL_INDICATORS)

            # Use trafilatura for high quality article extraction
            extracted = trafilatura.extract(str(soup), include_comments=False, include_tables=True)
            if not extracted:
                extracted = trafilatura.extract(html)

            content = extracted or raw_text
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            words = len(content.split())

            return {
                "success": bool(content and words >= 10),
                "url": url,
                "title": title or "Website Content",
                "content": content,
                "images": content_images,
                "word_count": words,
                "strategy_used": "direct_stealth",
                "paywall_detected": paywall_detected or (words < 120 and paywall_detected),
                "paywall_bypassed": False,
            }
        except Exception as e:
            logger.warning(f"Direct stealth scrape failed for {url}: {e}")
            return {"success": False, "error": str(e), "paywall_detected": True}

    def _scrape_medium_apollo(self, url: str) -> Dict[str, Any]:
        """Extracts article paragraphs from Medium embedded Apollo Client state."""
        try:
            headers = dict(self.default_headers)
            headers["Referer"] = "https://www.google.com/"
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            html = resp.text
            match = re.search(r"window\.__APOLLO_STATE__\s*=\s*(\{.+?\});?</script>", html)
            if not match:
                match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.+?\});?</script>", html)

            if match:
                data = json.loads(match.group(1))
                paragraphs = []
                title = ""

                for key, val in data.items():
                    if isinstance(val, dict):
                        if val.get("__typename") == "Post":
                            title = val.get("title", title)
                        if val.get("__typename") == "Paragraph":
                            text = val.get("text", "")
                            if text:
                                paragraphs.append(text)

                if paragraphs:
                    full_content = "\n\n".join(paragraphs)
                    words = len(full_content.split())
                    soup = BeautifulSoup(html, "lxml")
                    content_images = self._extract_content_images(soup, url)
                    return {
                        "success": True,
                        "url": url,
                        "title": title or "Medium Article",
                        "content": full_content,
                        "word_count": words,
                        "images": content_images,
                        "strategy_used": "medium_apollo_state",
                        "paywall_detected": True,
                        "paywall_bypassed": True,
                    }
        except Exception as e:
            logger.debug(f"Medium Apollo state extraction failed for {url}: {e}")

        return {"success": False}

    def _scrape_freedium(self, url: str) -> Dict[str, Any]:
        """Fetches Medium article content via Freedium proxy to bypass member-only paywall."""
        try:
            freedium_url = f"https://freedium.cfd/{url}"
            logger.info(f"Attempting Freedium proxy for Medium URL: {url}")
            resp = requests.get(
                freedium_url,
                headers=self.default_headers,
                timeout=self.timeout + 5,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                return {"success": False, "error": f"Freedium HTTP {resp.status_code}"}

            html = resp.text
            # Use trafilatura for clean article extraction from the proxy page
            extracted = trafilatura.extract(html, include_comments=False, include_tables=True)

            if not extracted:
                soup = BeautifulSoup(html, "lxml")
                # Remove Freedium UI chrome
                for tag in soup(["script", "style", "nav", "footer", "iframe", "header"]):
                    tag.decompose()
                # Try to find the article body
                article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|article|post", re.I))
                if article:
                    extracted = article.get_text(separator="\n", strip=True)
                else:
                    extracted = soup.get_text(separator="\n", strip=True)

            if not extracted:
                return {"success": False, "error": "Freedium returned no extractable content"}

            content = re.sub(r"\n{3,}", "\n\n", extracted).strip()
            words = len(content.split())

            # Extract title
            title = "Medium Article"
            soup_title = BeautifulSoup(resp.text, "lxml")
            if soup_title.title and soup_title.title.string:
                raw_title = soup_title.title.string.strip()
                # Strip "Freedium" prefix if present
                raw_title = re.sub(r"^Freedium\s*[-–—|:]\s*", "", raw_title).strip()
                if raw_title:
                    title = raw_title

            return {
                "success": words > 100,
                "url": url,
                "title": title,
                "content": content,
                "word_count": words,
                "strategy_used": "freedium_proxy",
                "paywall_detected": True,
                "paywall_bypassed": True,
            }
        except Exception as e:
            logger.warning(f"Freedium proxy scrape failed for {url}: {e}")
            return {"success": False, "error": str(e)}

    def _scrape_google_cache(self, url: str) -> Dict[str, Any]:
        """Fetches cached page from Google Webcache as a paywall fallback."""
        try:
            cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
            headers = dict(self.default_headers)
            headers["Referer"] = "https://www.google.com/"
            resp = requests.get(cache_url, headers=headers, timeout=self.timeout, allow_redirects=True)
            if resp.status_code != 200:
                return {"success": False, "error": f"Google Cache HTTP {resp.status_code}"}

            extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
            if not extracted:
                return {"success": False, "error": "Google Cache returned no extractable content"}

            content = re.sub(r"\n{3,}", "\n\n", extracted).strip()
            words = len(content.split())

            title = "Cached Page"
            soup = BeautifulSoup(resp.text, "lxml")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            return {
                "success": words > 50,
                "url": url,
                "title": title,
                "content": content,
                "word_count": words,
                "strategy_used": "google_cache",
                "paywall_detected": True,
                "paywall_bypassed": True,
            }
        except Exception as e:
            logger.debug(f"Google Cache fetch failed for {url}: {e}")
            return {"success": False, "error": str(e)}

    def _scrape_jina_reader(self, url: str) -> Dict[str, Any]:
        """Fetches and cleans article Markdown using Jina AI Reader API."""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "X-Return-Format": "markdown",
                "X-No-Cache": "true",
            }
            resp = requests.get(jina_url, headers=headers, timeout=self.timeout + 10)
            if resp.status_code == 200 and resp.text:
                text = resp.text.strip()
                # Parse title from first header if present
                title = "Website Content"
                lines = text.splitlines()
                for line in lines[:5]:
                    if line.startswith("# "):
                        title = line.replace("# ", "").strip()
                        break

                words = len(text.split())

                # Extract markdown image diagrams with their adjacent context
                extracted_images = []
                seen_img_urls = set()
                bad_context_tokens = {"share", "listen", "follow", "sign up", "sign in", "bookmark", "responses"}
                for i, line in enumerate(lines):
                    m = re.search(r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)', line)
                    if m:
                        alt = m.group(1).strip()
                        img_url = m.group(2).strip()
                        if img_url in seen_img_urls:
                            continue
                        lower_img = img_url.lower()
                        if any(bad in lower_img for bad in [
                            "avatar", "icon", "logo", "pixel", "badge", "sprite", ".svg",
                            "resize:fill",
                        ]):
                            continue
                        if any(bad in alt.lower() for bad in ["avatar", "user", "author", "unknown user"]):
                            continue
                        seen_img_urls.add(img_url)

                        def clean_line_text(raw_line: str) -> str:
                            cl = re.sub(r'\[([^\]]*)\]\([^\)]+\)', r'\1', raw_line).strip()
                            cl = re.sub(r'https?://\S+', '', cl).strip()
                            cl = re.sub(r'^[#\*\-\s]+', '', cl).strip()
                            return cl

                        context_before = []
                        for l in lines[max(0, i - 10):i]:
                            cl = clean_line_text(l)
                            if (
                                cl
                                and not l.strip().startswith("![")
                                and "click to view" not in cl.lower()
                                and cl.lower() not in bad_context_tokens
                                and len(cl) > 3
                            ):
                                context_before.append(cl)

                        context_after = []
                        for l in lines[i + 1:min(len(lines), i + 6)]:
                            cl = clean_line_text(l)
                            if (
                                cl
                                and not l.strip().startswith("![")
                                and "click to view" not in cl.lower()
                                and cl.lower() not in bad_context_tokens
                                and len(cl) > 3
                            ):
                                context_after.append(cl)

                        ctx_b = context_before[-1] if context_before else ""
                        ctx_a = context_after[0] if context_after else ""
                        desc = ctx_b or ctx_a or alt or "Architecture Diagram"
                        full_context = f"{desc}. {ctx_a[:150]}" if (ctx_a and ctx_a != desc) else desc

                        extracted_images.append({
                            "url": img_url,
                            "alt": desc[:100],
                            "context": full_context,
                        })
                        if len(extracted_images) >= 20:
                            break

                return {
                    "success": words > 50,
                    "url": url,
                    "title": title,
                    "content": text,
                    "images": extracted_images,
                    "word_count": words,
                    "strategy_used": "jina_reader",
                    "paywall_detected": True,
                    "paywall_bypassed": True,
                }
        except Exception as e:
            logger.warning(f"Jina Reader fetch failed for {url}: {e}")

        return {"success": False}

    def _scrape_wayback_archive(self, url: str) -> Dict[str, Any]:
        """Fetches snapshot from Wayback Machine as fallback for paywalled pages."""
        try:
            api_url = f"https://archive.org/wayback/available?url={url}"
            resp = requests.get(api_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                snapshots = data.get("archived_snapshots", {})
                closest = snapshots.get("closest", {})
                if closest.get("available") and closest.get("url"):
                    snapshot_url = closest.get("url")
                    logger.info(f"Found Wayback snapshot for {url}: {snapshot_url}")

                    # Fetch archived page
                    arch_resp = requests.get(snapshot_url, headers=self.default_headers, timeout=self.timeout)
                    if arch_resp.status_code == 200:
                        content = trafilatura.extract(arch_resp.text)
                        if content:
                            words = len(content.split())
                            return {
                                "success": True,
                                "url": url,
                                "title": f"[Archive] {urlparse(url).netloc}",
                                "content": content,
                                "word_count": words,
                                "strategy_used": "wayback_archive",
                                "paywall_detected": True,
                                "paywall_bypassed": True,
                            }
        except Exception as e:
            logger.warning(f"Wayback archive query failed for {url}: {e}")

        return {"success": False}

    # def extract_from_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    #     """Parses and extracts readable text from uploaded documents."""
    #     try:
    #         lower_name = filename.lower()
    #         text_content = ""
    #         title = filename
    # 
    #         if lower_name.endswith(".pdf"):
    #             pdf_file = io.BytesIO(file_bytes)
    #             reader = PdfReader(pdf_file)
    #             extracted_pages = []
    #             for i, page in enumerate(reader.pages):
    #                 page_text = page.extract_text()
    #                 if page_text and page_text.strip():
    #                     extracted_pages.append(f"--- Page {i + 1} ---\n{page_text.strip()}")
    #             text_content = "\n\n".join(extracted_pages)
    #             # Attempt to extract metadata title if available
    #             if reader.metadata and reader.metadata.title:
    #                 title = reader.metadata.title
    #         else:
    #             # Text / Markdown / CSV / JSON decoding
    #             try:
    #                 text_content = file_bytes.decode("utf-8")
    #             except UnicodeDecodeError:
    #                 text_content = file_bytes.decode("latin-1", errors="ignore")
    # 
    #         text_content = re.sub(r"\n{3,}", "\n\n", text_content).strip()
    #         word_count = len(text_content.split())
    # 
    #         if not text_content or word_count < 5:
    #             return {
    #                 "success": False,
    #                 "error": f"Uploaded file '{filename}' contains insufficient readable text.",
    #                 "filename": filename,
    #                 "title": title,
    #                 "content": "",
    #                 "word_count": 0,
    #             }
    # 
    #         return {
    #             "success": True,
    #             "filename": filename,
    #             "title": title,
    #             "content": text_content,
    #             "word_count": word_count,
    #             "file_type": filename.split(".")[-1].upper() if "." in filename else "TEXT",
    #         }
    #     except Exception as e:
    #         logger.error(f"Error extracting text from file '{filename}': {e}", exc_info=True)
    #         return {
    #             "success": False,
    #             "error": f"Failed to extract document content: {str(e)}",
    #             "filename": filename,
    #             "title": filename,
    #             "content": "",
    #             "word_count": 0,
    #         }

    def should_auto_crawl(self, url: str) -> bool:
        """Determines if a URL should automatically be deep-crawled as a site or documentation hub."""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            path = parsed.path.lower()

            # Multi-tenant article/social platforms should not auto-crawl entire domains
            multi_tenant_portals = {
                "wikipedia.org", "medium.com", "substack.com", "reddit.com",
                "youtube.com", "twitter.com", "x.com", "arxiv.org", "github.com",
                "stackoverflow.com", "quora.com", "nytimes.com", "bbc.com", "cnn.com"
            }
            if any(portal in netloc for portal in multi_tenant_portals):
                return False

            # Site root / domain landing (e.g. https://docs.qdrant.tech/ or https://mysite.com/)
            if path in ["", "/", "/index.html", "/index.htm"]:
                return True

            # Documentation subdomains or paths
            if any(doc_kw in netloc for doc_kw in ["docs.", "documentation.", "wiki.", "api.", "manual.", "guide.", "help.", "learn."]):
                return True

            if any(path.startswith(p) for p in ["/docs", "/doc/", "/documentation", "/guide", "/manual", "/reference", "/tutorial", "/learn", "/api/"]):
                return True

            # Generic project/company sites with shallow depth (<= 1)
            depth = len([p for p in path.split("/") if p])
            if depth <= 1:
                return True

            return False
        except Exception:
            return False

    def crawl_domain(
        self,
        start_url: str,
        max_pages: int = 8,
        max_depth: int = 2,
        strategy: str = "auto",
    ) -> Dict[str, Any]:
        """Recursively crawls linked pages on the same root domain up to max depth using parallel extraction."""
        try:
            is_valid, err = self.validate_url(start_url)
            if not is_valid:
                return {"success": False, "error": err, "root_url": start_url, "pages": []}

            parsed_root = urlparse(start_url)
            root_domain = parsed_root.netloc.lower()

            visited: Set[str] = set()
            crawled_pages: List[Dict[str, Any]] = []
            combined_text_chunks: List[str] = []

            logger.info(f"ScraperService: Initiating parallel domain crawl for {start_url} (target max_pages={max_pages})")

            # Scrape root/start page first
            clean_start, _ = urldefrag(start_url)
            visited.add(clean_start)
            root_res = self.scrape_url(clean_start, strategy=strategy)

            if not root_res.get("success") or root_res.get("word_count", 0) < 30:
                return {
                    "success": False,
                    "error": root_res.get("error", f"Failed to extract readable content from {start_url}"),
                    "root_url": start_url,
                    "pages": [],
                    "content": "",
                    "total_words": 0,
                }

            crawled_pages.append(root_res)
            combined_text_chunks.append(
                f"# Source Page: {root_res.get('title', clean_start)}\nURL: {clean_start}\n\n{root_res.get('content', '')}"
            )

            # Discover links from the start page
            initial_links = self._extract_internal_links(clean_start, root_domain)
            candidate_links = [l for l in initial_links if l not in visited]

            # Prioritize documentation, guides, concepts, tutorials
            def link_priority(link: str) -> int:
                try:
                    low = link.lower()
                    if any(k in low for k in ["doc", "guide", "quick", "concept", "api", "tutorial", "overview", "intro", "start"]):
                        return 0
                    return 1
                except Exception:
                    return 1

            candidate_links.sort(key=link_priority)

            # Select candidate URLs up to max candidate limit
            urls_to_scrape = candidate_links[: max_pages - 1]
            for u in urls_to_scrape:
                visited.add(u)

            # Parallel scrape candidate pages using ThreadPoolExecutor
            if urls_to_scrape:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(urls_to_scrape), 6)) as executor:
                    future_to_url = {
                        executor.submit(self.scrape_url, target, strategy): target
                        for target in urls_to_scrape
                    }
                    for future in concurrent.futures.as_completed(future_to_url):
                        target = future_to_url[future]
                        try:
                            res = future.result()
                            if res.get("success") and res.get("word_count", 0) >= 40:
                                crawled_pages.append(res)
                                combined_text_chunks.append(
                                    f"# Source Page: {res.get('title', target)}\nURL: {target}\n\n{res.get('content', '')}"
                                )
                        except Exception as e:
                            logger.warning(f"Error crawling sub-page {target}: {e}")

            total_words = sum(p.get("word_count", 0) for p in crawled_pages)
            separator = "\n\n" + ("=" * 40) + "\n\n"
            aggregated_content = separator.join(combined_text_chunks)

            return {
                "success": True,
                "root_url": start_url,
                "pages_crawled": len(crawled_pages),
                "total_words": total_words,
                "pages": crawled_pages,
                "content": aggregated_content,
                "title": f"Site Crawl: {parsed_root.netloc} ({len(crawled_pages)} pages)",
            }
        except Exception as e:
            logger.error(f"Error crawling domain for {start_url}: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Domain crawl error: {str(e)}",
                "root_url": start_url,
                "pages": [],
                "content": "",
                "total_words": 0,
            }

    def _extract_internal_links(self, current_url: str, root_domain: str) -> List[str]:
        """Fetches page HTML and extracts unique internal links on the same root domain."""
        links: List[str] = []
        try:
            resp = requests.get(current_url, headers=self.default_headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "lxml")
            for a_tag in soup.find_all("a", href=True):
                href = str(a_tag.get("href", "")).strip()
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue

                abs_url = urljoin(current_url, href)
                parsed = urlparse(abs_url)

                # Skip non-http schemes
                if parsed.scheme not in ["http", "https"]:
                    continue

                # Ensure same root domain
                if parsed.netloc.lower() != root_domain:
                    continue

                # Skip asset and media files
                path_lower = parsed.path.lower()
                if any(path_lower.endswith(ext) for ext in [
                    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
                    ".css", ".js", ".json", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".pdf"
                ]):
                    continue

                clean_link, _ = urldefrag(abs_url)
                if clean_link not in links:
                    links.append(clean_link)
        except Exception as e:
            logger.debug(f"Error extracting links from {current_url}: {e}")

        return links


# Global singleton instance
scraper_service = ScraperService()

