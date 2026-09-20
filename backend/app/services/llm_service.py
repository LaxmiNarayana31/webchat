from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from backend.app.clients.llm_client import ResilientLLMClient, llm_client
from backend.app.core.logging import logger


class LLMService:
    """High-level LLM Service managing multi-model orchestration, streaming, and telemetry."""

    def __init__(self, client: Optional[ResilientLLMClient] = None):
        """Initializes LLM service with resilient multi-provider client."""
        self.client = client or llm_client

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Generates response using transparent failover priority chain."""
        try:
            return self.client.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                selected_model=selected_model,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"LLMService: generate_response failed: {e}", exc_info=True)
            raise

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams generation tokens using transparent failover priority chain."""
        try:
            for chunk in self.client.generate_stream(
                prompt=prompt,
                system_instruction=system_instruction,
                selected_model=selected_model,
                temperature=temperature,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLMService: generate_stream failed: {e}", exc_info=True)
            raise

    async def generate_response_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Generates response asynchronously using transparent failover priority chain."""
        try:
            return await self.client.generate_response_async(
                prompt=prompt,
                system_instruction=system_instruction,
                selected_model=selected_model,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"LLMService: generate_response_async failed: {e}", exc_info=True)
            raise

    async def generate_stream_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streams generation tokens asynchronously using transparent failover priority chain."""
        try:
            async for chunk in self.client.generate_stream_async(
                prompt=prompt,
                system_instruction=system_instruction,
                selected_model=selected_model,
                temperature=temperature,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLMService: generate_stream_async failed: {e}", exc_info=True)
            raise

    def get_model_catalog(self) -> List[Dict[str, Any]]:
        """Returns model priority list with live rate limiter telemetry."""
        try:
            return self.client.get_model_catalog()
        except Exception as e:
            logger.error(f"LLMService: get_model_catalog failed: {e}", exc_info=True)
            return []


# Global singleton instance
llm_service = LLMService()
MultiProviderLLMService = LLMService