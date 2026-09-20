import concurrent.futures
import os
from typing import AsyncGenerator, Generator, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from backend.app.core.errors import LLMProviderException
from backend.app.core.logging import logger
from backend.app.services.memory_service import memory_service


load_dotenv()


class GeminiClient:
    """Encapsulates interaction with the official Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None):
        """Initializes Gemini API client configuration."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Initializes and returns the Gemini SDK client instance registered with Memori."""
        if not self._client:
            if not self.api_key:
                raise LLMProviderException(
                    "GEMINI_API_KEY is not configured.",
                    details={"provider": "gemini"},
                )
            base_client = genai.Client(api_key=self.api_key)
            memory_service.set_embed_fn(self.embed_texts)
            self._client = memory_service.register_llm(base_client)
        return self._client

    def generate_text(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
    ) -> str:
        """Executes non-streaming generation via Google Gemini."""
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction if system_instruction else None,
        )
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def generate_stream(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
    ) -> Generator[str, None, None]:
        """Streams text chunks via Google Gemini."""
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction if system_instruction else None,
        )
        response = self.client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def generate_text_async(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
    ) -> str:
        """Executes non-streaming generation via Google Gemini asynchronously."""
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction if system_instruction else None,
        )
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_stream_async(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
    ) -> AsyncGenerator[str, None]:
        """Streams text chunks via Google Gemini asynchronously."""
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction if system_instruction else None,
        )
        response = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def analyze_image_async(
        self,
        image_url: str,
        context: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ) -> str:
        """Fetches an image URL and generates visual descriptions & OCR breakdown via Gemini Vision."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as h_client:
                resp = await h_client.get(image_url)
                if resp.status_code != 200:
                    return f"[Visual Image Diagram: {context or image_url}]"
                image_bytes = resp.content
                mime_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]

            prompt_text = (
                "Analyze this web article image diagram in detail. "
                "Describe visual elements, flowcharts, architecture, text/OCR content, or key data trends shown. "
                f"Image Context: {context or 'Web visual asset'}"
            )

            image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[image_part, prompt_text],
            )
            return response.text or context or f"Visual Diagram at {image_url}"
        except Exception as e:
            logger.warning(f"GeminiClient: Vision analysis for {image_url} failed: {e}")
            return context or f"Visual Diagram at {image_url}"

    def embed_texts(self, texts: List[str], model: str = "gemini-embedding-2") -> List[List[float]]:
        """Generates 3072-dimensional vector embeddings using Google GenAI SDK with retry and rate limiting."""
        def _embed_single(single_text: str) -> List[float]:
            if not single_text or not single_text.strip():
                return [0.0] * 3072

            for attempt in range(3):
                try:
                    resp = self.client.models.embed_content(
                        model=model,
                        contents=single_text,
                    )
                    if resp.embeddings and len(resp.embeddings) > 0:
                        return resp.embeddings[0].values or [0.0] * 3072
                    return [0.0] * 3072
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        import time
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    logger.warning(f"Embedding with {model} failed: {e}. Trying gemini-embedding-001 fallback.")
                    try:
                        single_resp = self.client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=single_text,
                        )
                        if single_resp.embeddings and len(single_resp.embeddings) > 0:
                            return single_resp.embeddings[0].values or [0.0] * 3072
                        return [0.0] * 3072
                    except Exception as err:
                        logger.error(f"Single embedding fallback failed: {err}")
                        return [0.0] * 3072
            return [0.0] * 3072

        if len(texts) <= 1:
            return [_embed_single(t) for t in texts]

        max_workers = min(2, len(texts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_embed_single, texts))




gemini_client = GeminiClient()
