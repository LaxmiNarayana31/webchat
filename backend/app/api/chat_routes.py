from typing import List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from backend.app.core.errors import WebChatException
from backend.app.core.logging import logger
from backend.app.core.rate_limiter import get_client_ip
from backend.app.dtos.chat_dto import ChatRequestDto, ChatResponseDto
from backend.app.dtos.model_dto import ModelCatalogItemDto
from backend.app.dtos.scrape_dto import (
    CrawlRequestDto,
    CrawlResponseDto,
    ScrapeRequestDto,
    ScrapeResponseDto,
)
from backend.app.services.chat_service import chat_service


router = APIRouter(tags=["Chat & Ingestion"])


@router.post("/chat", response_model=ChatResponseDto)
async def chat_endpoint(req: ChatRequestDto, request: Request):
    """Handles conversational RAG query via standard JSON or SSE token stream with IP defense."""
    try:
        client_ip = get_client_ip(request)
        if req.stream:
            stream_gen = chat_service.handle_chat_stream_async(req, client_ip=client_ip)
            return StreamingResponse(stream_gen, media_type="text/event-stream")

        return await chat_service.handle_chat_query_async(req, client_ip=client_ip)

    except WebChatException as wce:
        logger.warning(f"Chat request handled exception: {wce.message}")
        raise HTTPException(
            status_code=wce.status_code,
            detail=wce.details if wce.details else wce.message,
        )
    except Exception as e:
        logger.error(f"Unexpected chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/scrape", response_model=ScrapeResponseDto)
def scrape_endpoint(req: ScrapeRequestDto):
    """Extracts web page content with paywall bypass and indexes embeddings into cache."""
    try:
        return chat_service.handle_scrape_and_index(req)
    except WebChatException as wce:
        logger.warning(f"Scrape request handled exception: {wce.message}")
        raise HTTPException(
            status_code=wce.status_code,
            detail=wce.details if wce.details else wce.message,
        )
    except Exception as e:
        logger.error(f"Unexpected scrape endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/crawl", response_model=CrawlResponseDto)
def crawl_endpoint(req: CrawlRequestDto):
    """Performs recursive domain crawl and indexes pages into hierarchical vector store."""
    try:
        return chat_service.handle_crawl_and_index(req)
    except WebChatException as wce:
        logger.warning(f"Crawl request handled exception: {wce.message}")
        raise HTTPException(
            status_code=wce.status_code,
            detail=wce.details if wce.details else wce.message,
        )
    except Exception as e:
        logger.error(f"Unexpected crawl endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# @router.post("/upload-file", response_model=FileUploadResponseDto)
# async def upload_file_endpoint(file: UploadFile = File(...)):
#     """Accepts document file uploads, extracts readable text, and indexes into vector cache."""
#     try:
#         file_bytes = await file.read()
#         filename = file.filename or "uploaded_document.txt"
#         return chat_service.handle_file_upload_and_index(file_bytes=file_bytes, filename=filename)
#     except WebChatException as wce:
#         logger.warning(f"File upload handled exception: {wce.message}")
#         raise HTTPException(
#             status_code=wce.status_code,
#             detail=wce.details if wce.details else wce.message,
#         )
#     except Exception as e:
#         logger.error(f"Unexpected file upload error: {e}", exc_info=True)
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models", response_model=List[ModelCatalogItemDto])
def list_models():
    """Returns priority model chain and live rate-limiter telemetry."""
    try:
        return chat_service.get_model_catalog()
    except Exception as e:
        logger.error(f"Error fetching model catalog: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

