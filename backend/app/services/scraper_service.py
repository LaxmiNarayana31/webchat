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

    BOT_CHALLENGE_INDICATORS = [
        "just a moment...",
        "security | glassdoor",
        "security check | glassdoor",
        "security check",
        "humans only",
        "attention required! | cloudflare",
        "access denied | www.glassdoor",
        "used cloudflare to restrict access",
        "ddos protection by cloudflare",
        "cloudflare ray id",
        "ray id:",
        "checking your browser before accessing",
        "enable javascript and cookies to continue",
        "please turn javascript on and reload the page",
        "verify you are human",
        "verifying you are human",
        "completing the captcha proves you are a human",
        "our systems have detected unusual traffic",
        "press & hold to confirm you are a human",
        "glassdoor has been built on the contributions of real employees",
        "we use advanced security systems to keep our site safe and prevent misuse",
        "access to this page has been denied",
        "robot or human?",
        "are you a human?",
        "challenge-platform",
    ]

    def _is_bot_challenge_page(self, title: str = "", text: str = "", html: str = "") -> bool:
        """Detects whether extracted content is an anti-bot roadblock, CAPTCHA, or Cloudflare challenge."""
        title_lower = (title or "").lower().strip()
        text_lower = (text or "").lower()
        html_lower = (html or "").lower()

        # Check explicit challenge titles
        bad_titles = [
            "just a moment...",
            "security | glassdoor",
            "security check",
            "attention required!",
            "access denied",
            "ddos protection by cloudflare",
            "robot or human?",
            "are you a human?",
            "verify you are human",
        ]
        if any(bt in title_lower for bt in bad_titles):
            return True

        # Check high-confidence anti-bot challenge signatures in content or html
        critical_hits = [
            "humans only",
            "just a moment...",
            "security | glassdoor",
            "used cloudflare to restrict access",
            "checking your browser before accessing",
            "completing the captcha proves you are a human",
            "enable javascript and cookies to continue",
            "please turn javascript on and reload the page",
            "glassdoor has been built on the contributions of real employees",
        ]
        if any(ch in text_lower or ch in html_lower for ch in critical_hits):
            return True

        matched_indicators = [ind for ind in self.BOT_CHALLENGE_INDICATORS if ind in text_lower or ind in html_lower]
        if len(matched_indicators) >= 2:
            return True

        return False

    def _extract_structured_json_ld(self, soup: BeautifulSoup, base_url: str = "") -> Tuple[str, str, int]:
        """Extracts rich structured data (FAQ, Salaries, Company Ratings, Articles) from JSON-LD scripts.

        Returns: (markdown_content, title_hint, word_count)
        """
        if not soup:
            return "", "", 0

        sections: List[str] = []
        title_hint = ""

        try:
            for tag in soup.find_all("script", type=re.compile(r"ld\+json", re.I)):
                content_str = tag.string or tag.get_text() or ""
                if not content_str.strip():
                    continue
                try:
                    data = json.loads(content_str)
                except Exception:
                    continue

                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    schema_type = item.get("@type", "")

                    if schema_type == "FAQPage":
                        faq_lines = ["## Frequently Asked Questions & Salary Data"]
                        for q_entry in item.get("mainEntity", []):
                            if not isinstance(q_entry, dict):
                                continue
                            question = q_entry.get("name", "").strip()
                            ans_raw = q_entry.get("acceptedAnswer", {}).get("text", "")
                            ans_clean = re.sub(r"<[^>]+>", " ", ans_raw).strip()
                            ans_clean = re.sub(r"\s+", " ", ans_clean)
                            if question and ans_clean:
                                faq_lines.append(f"### {question}\n{ans_clean}\n")
                        if len(faq_lines) > 1:
                            sections.append("\n".join(faq_lines))

                    elif schema_type == "EmployerAggregateRating":
                        org_name = item.get("itemReviewed", {}).get("name", "Company")
                        val = item.get("ratingValue", "")
                        best = item.get("bestRating", "5")
                        count = item.get("ratingCount", "")
                        sections.append(f"## {org_name} - Employer Rating\n- Rating: {val} / {best} stars (based on {count} employee reviews)\n")

                    elif schema_type in ["JobPosting", "Salary"]:
                        j_title = item.get("title") or item.get("name", "")
                        base_sal = item.get("baseSalary", {})
                        sal_info = ""
                        if isinstance(base_sal, dict):
                            val_obj = base_sal.get("value", {})
                            currency = base_sal.get("currency", "")
                            if isinstance(val_obj, dict):
                                min_v = val_obj.get("minValue")
                                max_v = val_obj.get("maxValue")
                                unit = val_obj.get("unitText", "YEAR")
                                sal_info = f"- Base Salary: {min_v} - {max_v} {currency} per {unit}"
                        desc = re.sub(r"<[^>]+>", " ", item.get("description", "")).strip()[:500]
                        sections.append(f"## {j_title}\n{sal_info}\n{desc}\n")

                    elif schema_type == "BreadcrumbList":
                        crumbs = []
                        for cr in item.get("itemListElement", []):
                            if isinstance(cr, dict) and cr.get("name"):
                                crumbs.append(cr["name"].strip())
                        if crumbs:
                            title_hint = " > ".join(crumbs)

                    elif schema_type in ["Article", "NewsArticle", "TechArticle"]:
                        headline = item.get("headline", "")
                        body = item.get("articleBody", "")
                        desc = item.get("description", "")
                        if headline and not title_hint:
                            title_hint = headline
                        if body:
                            sections.append(f"## {headline}\n\n{body}\n")
                        elif desc:
                            sections.append(f"## {headline}\n\n{desc}\n")
        except Exception as e:
            logger.debug(f"JSON-LD extraction note: {e}")

        combined_text = "\n\n".join(s.strip() for s in sections if s.strip())
        word_count = len(combined_text.split())
        return combined_text, title_hint, word_count

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
            is_substack = self._is_substack_domain(url)
            is_news = self._is_news_domain(url)
            is_glassdoor = "glassdoor" in url.lower()

            direct_res = {}

            # Tier 1: Platform-Specific Archetype Handlers

            # Substack native REST API (bypasses subscribe modals & email walls)
            if (is_substack and strategy in ["auto", "substack"]) or strategy == "substack":
                substack_res = self._scrape_substack(url)
                if substack_res.get("success") and substack_res.get("word_count", 0) > 40:
                    return substack_res

            # Medium via active Freedium mirror cluster or Apollo State
            if (is_medium and strategy in ["auto", "medium"]) or strategy == "medium":
                freedium_res = self._scrape_freedium(url)
                if freedium_res.get("success") and freedium_res.get("word_count", 0) > 150:
                    return freedium_res

                if settings.JINA_READER_ENABLED:
                    jina_res = self._scrape_jina_reader(url)
                    jina_has_paywall = any(p in jina_res.get("content", "").lower() for p in self.PAYWALL_INDICATORS)
                    if jina_res.get("success") and jina_res.get("word_count", 0) > 200 and not jina_has_paywall:
                        jina_res["paywall_bypassed"] = True
                        return jina_res

                medium_res = self._scrape_medium_apollo(url)
                if medium_res.get("success") and medium_res.get("word_count", 0) > 150:
                    return medium_res

            # Major news publications with hard paywalls
            if is_news and strategy in ["auto", "news"]:
                news_res = self._scrape_news_publication(url)
                if news_res.get("success") and news_res.get("word_count", 0) > 100:
                    return news_res

            # Tier 2: TLS Fingerprint Impersonation (bypasses Cloudflare / Glassdoor / Turnstile WAFs)
            if strategy in ["auto", "stealth", "direct"] or is_glassdoor:
                impersonate_res = self._scrape_curl_impersonate(url)
                if impersonate_res.get("success") and impersonate_res.get("word_count", 0) >= 40:
                    return impersonate_res

            # Tier 3: Standard Direct Stealth Scrape with Referer Spoofing & Schema.org JSON-LD
            if strategy in ["auto", "direct"]:
                direct_res = self._scrape_direct_stealth(url)
                if direct_res.get("success") and direct_res.get("word_count", 0) > 300:
                    return direct_res
                if direct_res.get("success") and not direct_res.get("paywall_detected") and direct_res.get("word_count", 0) >= 10:
                    return direct_res
                if direct_res.get("paywall_detected") or not direct_res.get("success"):
                    logger.info(f"Direct scrape encountered paywall or low content for {url}. Attempting fallbacks...")

            # Tier 4: Jina AI Reader API (handles JS rendering, sign-in modals, soft paywalls)
            if settings.JINA_READER_ENABLED and strategy in ["auto", "jina"]:
                jina_res = self._scrape_jina_reader(url)
                if jina_res.get("success") and jina_res.get("word_count", 0) > 150:
                    jina_res["paywall_bypassed"] = True
                    return jina_res

            # Tier 5: Google Webcache fallback
            if strategy in ["auto"]:
                cache_res = self._scrape_google_cache(url)
                if cache_res.get("success") and cache_res.get("word_count", 0) > 150:
                    cache_res["paywall_bypassed"] = True
                    return cache_res

            # Tier 6: Wayback Machine Archival fallback
            if settings.ARCHIVE_FALLBACK_ENABLED and strategy in ["auto", "archive"]:
                archive_res = self._scrape_wayback_archive(url)
                if archive_res.get("success") and archive_res.get("word_count", 0) > 150:
                    archive_res["paywall_bypassed"] = True
                    return archive_res

            # Tier 7: Final check on direct result if valid and not a bot challenge
            if direct_res.get("success") and not self._is_bot_challenge_page(direct_res.get("title", ""), direct_res.get("content", "")):
                return direct_res

            # If all tiers failed, construct clear, honest diagnostic error
            antibot_detected = False
            if "anti-bot" in str(direct_res.get("error", "")).lower() or direct_res.get("strategy_used") == "blocked_by_antibot":
                antibot_detected = True

            err_msg = (
                "This website is protected by Cloudflare Bot Management / Turnstile (anti-bot security challenge). "
                "Automated scrapers and proxies were blocked by the site's security gateway."
                if (antibot_detected or is_glassdoor)
                else "Failed to bypass site protection or extract readable article content."
            )

            return {
                "success": False,
                "error": err_msg,
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "strategy_used": "blocked_by_antibot" if (antibot_detected or is_glassdoor) else "failed",
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

    def _is_substack_domain(self, url: str) -> bool:
        """Checks if URL is a Substack newsletter or post."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            return "substack.com" in domain or ("/p/" in path and ("newsletter" in domain or "blog" in domain or "pub" in domain))
        except Exception:
            return False

    def _scrape_substack(self, url: str) -> Dict[str, Any]:
        """Extracts full Substack article content via native REST API (/api/v1/posts/{slug}).
        Bypasses email subscribe gates, paywall modals, and sign-in overlays.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path

            m = re.search(r"/(?:p|post)/([a-zA-Z0-9_\-]+)", path)
            if not m:
                return {"success": False, "error": "No Substack post slug found in URL"}

            slug = m.group(1)
            api_url = f"https://{domain}/api/v1/posts/{slug}"
            logger.info(f"Attempting Substack API extraction: {api_url}")

            headers = dict(self.default_headers)
            headers["Accept"] = "application/json"
            resp = requests.get(api_url, headers=headers, timeout=self.timeout + 5)

            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title") or "Substack Article"
                subtitle = data.get("subtitle") or ""
                body_html = data.get("body_html") or ""

                if body_html:
                    soup = BeautifulSoup(body_html, "lxml")
                    content_images = self._extract_content_images(soup, url)
                    extracted_text = trafilatura.extract(body_html, include_comments=False, include_tables=True)
                    if not extracted_text:
                        extracted_text = soup.get_text(separator="\n", strip=True)

                    full_text = f"# {title}\n"
                    if subtitle:
                        full_text += f"_{subtitle}_\n\n"
                    full_text += extracted_text
                    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
                    words = len(full_text.split())

                    if words >= 40:
                        return {
                            "success": True,
                            "url": url,
                            "title": title,
                            "content": full_text,
                            "images": content_images,
                            "word_count": words,
                            "strategy_used": "substack_api",
                            "paywall_detected": True,
                            "paywall_bypassed": True,
                        }
        except Exception as e:
            logger.debug(f"Substack API extraction failed for {url}: {e}")

        return {"success": False}

    def _is_news_domain(self, url: str) -> bool:
        """Checks if URL belongs to major paywalled news publication domains."""
        try:
            domain = urlparse(url).netloc.lower()
            news_domains = [
                "nytimes.com", "wsj.com", "bloomberg.com", "theatlantic.com",
                "wired.com", "ft.com", "economist.com", "washingtonpost.com",
                "businessinsider.com", "newyorker.com", "hbr.org", "reuters.com",
                "latimes.com", "forbes.com", "fortune.com", "technologyreview.com",
            ]
            return any(nd in domain for nd in news_domains)
        except Exception:
            return False

    def _scrape_news_publication(self, url: str) -> Dict[str, Any]:
        """Extracts content from paywalled news publications using:
        1. Googlebot crawler header spoofing + search referer
        2. curl_cffi TLS impersonation + Twitter/X referer
        3. Schema.org NewsArticle/Article extraction
        4. Wayback Machine snapshot
        """
        try:
            # 1. Googlebot crawler exemption
            googlebot_headers = {
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
            }
            try:
                resp = requests.get(url, headers=googlebot_headers, timeout=self.timeout)
                if resp.status_code == 200 and resp.text:
                    soup = BeautifulSoup(resp.text, "lxml")
                    title = soup.title.string.strip() if soup.title and soup.title.string else ""
                    if not self._is_bot_challenge_page(title, soup.get_text(), resp.text):
                        json_ld_text, json_ld_title, _ = self._extract_structured_json_ld(soup, url)
                        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
                        content = extracted or json_ld_text
                        if content and len(content.split()) >= 100:
                            content_images = self._extract_content_images(soup, url)
                            return {
                                "success": True,
                                "url": url,
                                "title": json_ld_title or title or "News Article",
                                "content": content,
                                "images": content_images,
                                "word_count": len(content.split()),
                                "strategy_used": "news_googlebot_crawler",
                                "paywall_detected": True,
                                "paywall_bypassed": True,
                            }
            except Exception as e:
                logger.debug(f"Googlebot news scrape failed for {url}: {e}")

            # 2. curl_cffi TLS impersonation
            impersonate_res = self._scrape_curl_impersonate(url)
            if impersonate_res.get("success") and impersonate_res.get("word_count", 0) > 150:
                impersonate_res["strategy_used"] = "news_impersonate"
                return impersonate_res

            # 3. Wayback Machine archive fallback
            archive_res = self._scrape_wayback_archive(url)
            if archive_res.get("success") and archive_res.get("word_count", 0) > 150:
                return archive_res
        except Exception as e:
            logger.debug(f"News publication scraper failed for {url}: {e}")

        return {"success": False}

    def _scrape_curl_impersonate(self, url: str) -> Dict[str, Any]:
        """Fetches page content using curl_cffi with Chrome 120 TLS fingerprint impersonation.
        Bypasses Cloudflare Bot Management, Turnstile, and advanced anti-bot WAFs.
        """
        try:
            from curl_cffi import requests as curl_requests
            # Use only Referer so curl_cffi generates authentic Chrome TLS & HTTP/2 fingerprint
            headers = {"Referer": "https://www.google.com/"}
            resp = curl_requests.get(url, impersonate="chrome120", headers=headers, timeout=self.timeout + 5)
            if resp.status_code >= 400 and resp.status_code != 403:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            html = resp.text
            soup = BeautifulSoup(html, "lxml")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract JSON-LD structured data first
            json_ld_text, json_ld_title, json_ld_words = self._extract_structured_json_ld(soup, url)
            if json_ld_title and (not title or "security" in title.lower() or "just a moment" in title.lower()):
                title = json_ld_title

            raw_text = soup.get_text()
            if self._is_bot_challenge_page(title, raw_text, html):
                if json_ld_words >= 80:
                    return {
                        "success": True,
                        "url": url,
                        "title": title or "Structured Knowledge Base",
                        "content": json_ld_text,
                        "images": [],
                        "word_count": json_ld_words,
                        "strategy_used": "curl_impersonate_json_ld",
                        "paywall_detected": True,
                        "paywall_bypassed": True,
                    }
                return {"success": False, "error": "Anti-bot challenge page encountered"}

            # Extract content images before tag cleanup
            content_images = self._extract_content_images(soup, url)

            # Remove scripts, styles, navigation, paywall modals
            for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript", "aside"]):
                tag.decompose()
            for cls_name in self.PAYWALL_CSS_CLASSES:
                for el in soup.find_all(attrs={"class": re.compile(cls_name, re.I)}):
                    el.decompose()

            extracted = trafilatura.extract(str(soup), include_comments=False, include_tables=True)
            if not extracted:
                extracted = trafilatura.extract(html)

            content = extracted or raw_text
            if json_ld_text and json_ld_text not in content:
                content = f"{content}\n\n{json_ld_text}".strip()

            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            words = len(content.split())

            if words >= 10 and not self._is_bot_challenge_page(title, content):
                return {
                    "success": True,
                    "url": url,
                    "title": title or "Website Content",
                    "content": content,
                    "images": content_images,
                    "word_count": words,
                    "strategy_used": "curl_impersonate",
                    "paywall_detected": True,
                    "paywall_bypassed": True,
                }
        except ImportError:
            logger.debug("curl_cffi not installed, skipping TLS impersonation.")
        except Exception as e:
            logger.debug(f"curl_cffi impersonate failed for {url}: {e}")

        return {"success": False}

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
        """Direct request using stealth headers, referer spoofing, and structured JSON-LD extraction."""
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

            html = resp.text if resp is not None else ""
            soup = BeautifulSoup(html, "lxml") if html else None

            # 1. Extract rich JSON-LD structured data BEFORE script tags are decomposed
            json_ld_text, json_ld_title, json_ld_words = self._extract_structured_json_ld(soup, url) if soup else ("", "", 0)

            title = ""
            if soup and soup.title and soup.title.string:
                title = soup.title.string.strip()
            if json_ld_title and (not title or "security" in title.lower() or "just a moment" in title.lower()):
                title = json_ld_title

            raw_text = soup.get_text() if soup else ""

            # 2. Check for Bot Challenge / Cloudflare Turnstile roadblock
            if self._is_bot_challenge_page(title, raw_text, html):
                # If site gave a challenge page, but embedded complete JSON-LD data (e.g. Glassdoor FAQ/Salary schema)
                if json_ld_words >= 80:
                    logger.info(f"Direct scrape bypassed anti-bot challenge using embedded structured JSON-LD ({json_ld_words} words)")
                    return {
                        "success": True,
                        "url": url,
                        "title": title or "Structured Knowledge Base",
                        "content": json_ld_text,
                        "images": [],
                        "word_count": json_ld_words,
                        "strategy_used": "direct_json_ld",
                        "paywall_detected": True,
                        "paywall_bypassed": True,
                    }
                return {
                    "success": False,
                    "error": "Site security (Cloudflare Turnstile/anti-bot challenge) blocked access.",
                    "paywall_detected": True,
                }

            html = resp.text
            soup = BeautifulSoup(html, "lxml")
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}",
                    "paywall_detected": resp.status_code in [401, 403, 429],
                }

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
            if json_ld_text and json_ld_text not in content:
                content = f"{content}\n\n{json_ld_text}".strip()

            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            words = len(content.split())

            # Double check final extracted text for bot challenge
            if self._is_bot_challenge_page(title, content):
                return {
                    "success": False,
                    "error": "Site security (anti-bot challenge) detected in extracted content.",
                    "paywall_detected": True,
                }

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
        """Fetches Medium article content via active Freedium mirrors to bypass member-only paywall."""
        mirrors = [
            "https://freedium-mirror.cfd/",
            "https://freedium.cfd/",
        ]
        last_error = "All Freedium mirrors failed"

        for base_mirror in mirrors:
            try:
                freedium_url = f"{base_mirror.rstrip('/')}/{url.strip()}"
                logger.info(f"Attempting Freedium mirror ({base_mirror}) for Medium URL: {url}")
                resp = requests.get(
                    freedium_url,
                    headers=self.default_headers,
                    timeout=self.timeout + 5,
                    allow_redirects=True,
                )
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code} from {base_mirror}"
                    continue

                html = resp.text
                extracted = trafilatura.extract(html, include_comments=False, include_tables=True)

                if not extracted:
                    soup = BeautifulSoup(html, "lxml")
                    for tag in soup(["script", "style", "nav", "footer", "iframe", "header"]):
                        tag.decompose()
                    article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|article|post", re.I))
                    if article:
                        extracted = article.get_text(separator="\n", strip=True)
                    else:
                        extracted = soup.get_text(separator="\n", strip=True)

                if not extracted or len(extracted.split()) < 80:
                    last_error = f"Insufficient content from {base_mirror}"
                    continue

                content = re.sub(r"\n{3,}", "\n\n", extracted).strip()
                words = len(content.split())

                # Extract title
                title = "Medium Article"
                soup_title = BeautifulSoup(html, "lxml")
                if soup_title.title and soup_title.title.string:
                    raw_title = soup_title.title.string.strip()
                    raw_title = re.sub(r"^\s*Freedium\s*[-–—|:]\s*", "", raw_title).strip()
                    raw_title = re.sub(r"\s*[-–—|:]\s*Freedium\s*$", "", raw_title).strip()
                    if raw_title:
                        title = raw_title

                soup_for_img = BeautifulSoup(html, "lxml")
                content_images = self._extract_content_images(soup_for_img, url)

                return {
                    "success": words > 100,
                    "url": url,
                    "title": title,
                    "content": content,
                    "images": content_images,
                    "word_count": words,
                    "strategy_used": "freedium_mirror",
                    "paywall_detected": True,
                    "paywall_bypassed": True,
                }
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Freedium mirror {base_mirror} failed for {url}: {e}")

        return {"success": False, "error": last_error}

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

            if self._is_bot_challenge_page(title, content, resp.text):
                return {"success": False, "error": "Google Cache snapshot contains anti-bot challenge"}

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
                # Parse title from first header or Title metadata if present
                title = "Website Content"
                lines = text.splitlines()
                for line in lines[:8]:
                    line_clean = line.strip()
                    if line_clean.startswith("Title:"):
                        cand = line_clean.replace("Title:", "").strip()
                        if cand and not self._is_bot_challenge_page(cand):
                            title = cand
                            break
                    elif line_clean.startswith("# "):
                        cand = line_clean.replace("# ", "").strip()
                        if cand and not self._is_bot_challenge_page(cand):
                            title = cand
                            break

                # Critical Anti-Bot Check: Jina Reader frequently returns Cloudflare challenge pages
                if self._is_bot_challenge_page(title=title, text=text):
                    logger.warning(f"Jina Reader returned anti-bot/Cloudflare challenge for {url}")
                    return {
                        "success": False,
                        "error": "Site security (Cloudflare Turnstile/anti-bot challenge) blocked Jina Reader proxy.",
                        "paywall_detected": True,
                    }

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
                            arch_title = f"[Archive] {urlparse(url).netloc}"
                            if self._is_bot_challenge_page(arch_title, content, arch_resp.text):
                                return {"success": False, "error": "Wayback archive snapshot was an anti-bot challenge page"}
                            return {
                                "success": True,
                                "url": url,
                                "title": arch_title,
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

