from typing import List, Optional

from pydantic import BaseModel, Field


class ScrapeRequestDto(BaseModel):
    """Payload to trigger website scraping and vector indexing."""
    url: str = Field(default="", description="Target web page or article URL to extract (single or comma/newline-separated)")
    urls: Optional[List[str]] = Field(default=None, description="Optional list of multiple URLs to extract simultaneously")
    strategy: str = Field(
        default="auto",
        description="Extraction strategy: 'auto', 'direct', 'jina', 'archive', 'medium'",
    )
    auto_crawl: bool = Field(
        default=True,
        description="Whether to automatically crawl linked pages if URL is a site root or documentation",
    )


class ScrapeResponseDto(BaseModel):
    """Response returned upon website scraping."""
    success: bool = Field(..., description="Whether scraping and text extraction succeeded")
    url: str = Field(..., description="Final resolved or original target URL")
    title: str = Field(default="", description="Extracted web page or article title")
    content: str = Field(default="", description="Extracted clean markdown or text content")
    word_count: int = Field(default=0, description="Total words extracted")
    strategy_used: str = Field(default="", description="Strategy that successfully bypassed or scraped the page")
    paywall_detected: bool = Field(default=False, description="Whether a paywall or sign-up wall was detected")
    paywall_bypassed: bool = Field(default=False, description="Whether paywall was successfully bypassed")
    pages_crawled: Optional[int] = Field(default=None, description="Number of sub-pages crawled if domain crawl was executed")
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")


class CrawlRequestDto(BaseModel):
    """Payload to trigger recursive multi-page domain scraping and indexing."""
    url: str = Field(..., description="Root target URL to begin crawling")
    max_pages: int = Field(default=5, ge=1, le=20, description="Maximum number of sub-pages to crawl")
    max_depth: int = Field(default=2, ge=1, le=4, description="Maximum link traversal depth")
    strategy: str = Field(default="auto", description="Extraction strategy for crawled pages")


class CrawlPageDto(BaseModel):
    """Metadata summary of a single crawled sub-page."""
    url: str
    title: str = ""
    word_count: int = 0
    strategy_used: str = ""


class CrawlResponseDto(BaseModel):
    """Response returned upon multi-page domain crawling."""
    success: bool = Field(..., description="Whether crawling succeeded")
    root_url: str = Field(..., description="Root URL crawled")
    pages_crawled: int = Field(default=0, description="Total pages successfully crawled")
    total_words: int = Field(default=0, description="Total aggregate words indexed")
    pages: List[CrawlPageDto] = Field(default_factory=list, description="List of crawled sub-pages")
    error: Optional[str] = Field(default=None, description="Error message if crawl failed")


class FileUploadResponseDto(BaseModel):
    """Response returned upon document file upload and indexing."""
    success: bool = Field(..., description="Whether file extraction and indexing succeeded")
    filename: str = Field(..., description="Original filename uploaded")
    file_type: str = Field(..., description="File format (e.g. PDF, Markdown, Text)")
    title: str = Field(default="", description="Document title extracted from file")
    word_count: int = Field(default=0, description="Total words extracted")
    virtual_url: str = Field(default="", description="Internal virtual URL used for indexing and session tracking")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")

