import asyncio
import re
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from backend.app.clients.gemini_client import GeminiClient, gemini_client
from backend.app.clients.groq_client import GroqClient, groq_client
from backend.app.core.config import settings
from backend.app.core.errors import RateLimitExceededException
from backend.app.core.logging import logger
from backend.app.core.rate_limiter import global_rate_limiter



class ResilientLLMClient:
    """Multi-provider LLM client orchestrating Google Gemini and Groq with fallback."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        custom_chain: Optional[List[Dict[str, str]]] = None,
    ):
        """Initializes resilient LLM client with Gemini, Groq, and rate limiters."""
        self.gemini = GeminiClient(api_key=gemini_api_key) if gemini_api_key else gemini_client
        self.groq = GroqClient(api_key=groq_api_key) if groq_api_key else groq_client
        self.model_chain = custom_chain or list(settings.DEFAULT_MODELS)
        self.rate_limiter = global_rate_limiter

    def _build_execution_chain(self, selected_model: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Constructs the prioritized model execution chain with optional user override."""
        if not selected_model or not selected_model.get("model"):
            return list(self.model_chain)

        primary_prov = selected_model.get("provider", "gemini").lower()
        primary_mod = selected_model.get("model", "")
        chain = [{"provider": primary_prov, "model": primary_mod}]

        for m in self.model_chain:
            m_prov = m.get("provider", "").lower()
            m_mod = m.get("model", "")
            if not (m_prov == primary_prov and m_mod.lower() == primary_mod.lower()):
                chain.append(m)
        return chain

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Executes non-streaming text generation across the fallback chain."""
        chain_to_try = self._build_execution_chain(selected_model)
        attempt_errors: List[str] = []
        start_time = time.time()

        for idx, model_info in enumerate(chain_to_try):
            provider = model_info.get("provider", "gemini").lower()
            model_name = model_info.get("model", "")
            rpm_limit = settings.GEMINI_RPM_LIMIT if provider == "gemini" else settings.GROQ_RPM_LIMIT

            # Check rate limiter
            allowed, reason = self.rate_limiter.can_proceed(provider, model_name, rpm_limit=rpm_limit)
            if not allowed:
                attempt_errors.append(f"Skipped {provider}/{model_name}: {reason}")
                logger.warning(f"RateLimiter: Skipping {provider}/{model_name} - {reason}")
                continue

            # Attempt API call with retries
            for attempt in range(settings.RETRY_MAX_ATTEMPTS):
                try:
                    logger.info(f"LLM attempt: {provider}/{model_name} (candidate {idx + 1}/{len(chain_to_try)})")
                    if provider == "gemini":
                        text = self.gemini.generate_text(
                            model=model_name,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                        )
                    elif provider == "groq":
                        text = self.groq.generate_text(
                            model=model_name,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider}")

                    self.rate_limiter.record_success(provider, model_name)
                    latency = round(time.time() - start_time, 2)

                    return {
                        "success": True,
                        "text": text,
                        "model_used": model_name,
                        "provider": provider,
                        "fallback_triggered": (idx > 0),
                        "latency_sec": latency,
                    }

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "429" in err_str
                        or "resource_exhausted" in err_str
                        or "rate limit" in err_str
                        or "quota" in err_str
                    )

                    if is_rate_limit:
                        retry_after = self._parse_retry_after(str(e))
                        self.rate_limiter.record_rate_limit_error(
                            provider, model_name, retry_after=retry_after
                        )
                        logger.warning(
                            f"Rate limit on {provider}/{model_name}: {e}. Retrying/Failing over..."
                        )

                        if attempt < settings.RETRY_MAX_ATTEMPTS - 1:
                            delay = self.rate_limiter.calculate_retry_delay(attempt)
                            time.sleep(delay)
                            continue
                        else:
                            attempt_errors.append(f"{provider}/{model_name} RateLimited: {e}")
                            break
                    else:
                        logger.warning(f"Error on {provider}/{model_name}: {e}. Cascading to fallback model...")
                        attempt_errors.append(f"{provider}/{model_name} Error: {e}")
                        break

        # All models failed
        logger.error(f"All LLM candidates exhausted. Details: {attempt_errors}")
        raise RateLimitExceededException(
            "All LLM candidates exhausted or rate limited.",
            details={"errors": attempt_errors},
        )

    async def generate_response_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Executes non-streaming text generation asynchronously across the fallback chain."""
        chain_to_try = self._build_execution_chain(selected_model)
        attempt_errors: List[str] = []
        start_time = time.time()

        for idx, model_info in enumerate(chain_to_try):
            provider = model_info.get("provider", "gemini").lower()
            model_name = model_info.get("model", "")
            rpm_limit = settings.GEMINI_RPM_LIMIT if provider == "gemini" else settings.GROQ_RPM_LIMIT

            # Check rate limiter
            allowed, reason = self.rate_limiter.can_proceed(provider, model_name, rpm_limit=rpm_limit)
            if not allowed:
                attempt_errors.append(f"Skipped {provider}/{model_name}: {reason}")
                logger.warning(f"RateLimiter: Skipping {provider}/{model_name} - {reason}")
                continue

            # Attempt API call with retries
            for attempt in range(settings.RETRY_MAX_ATTEMPTS):
                try:
                    logger.info(f"LLM async attempt: {provider}/{model_name} (candidate {idx + 1}/{len(chain_to_try)})")
                    if provider == "gemini":
                        text = await self.gemini.generate_text_async(
                            model=model_name,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                        )
                    elif provider == "groq":
                        # If groq async isn't implemented, fallback to sync wrapped in thread
                        text = await asyncio.to_thread(
                            self.groq.generate_text,
                            model=model_name,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider}")

                    self.rate_limiter.record_success(provider, model_name)
                    latency = round(time.time() - start_time, 2)

                    return {
                        "success": True,
                        "text": text,
                        "model_used": model_name,
                        "provider": provider,
                        "fallback_triggered": (idx > 0),
                        "latency_sec": latency,
                    }

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "429" in err_str
                        or "resource_exhausted" in err_str
                        or "rate limit" in err_str
                        or "quota" in err_str
                    )

                    if is_rate_limit:
                        retry_after = self._parse_retry_after(str(e))
                        self.rate_limiter.record_rate_limit_error(
                            provider, model_name, retry_after=retry_after
                        )
                        logger.warning(
                            f"Rate limit on {provider}/{model_name}: {e}. Retrying/Failing over..."
                        )

                        if attempt < settings.RETRY_MAX_ATTEMPTS - 1:
                            delay = self.rate_limiter.calculate_retry_delay(attempt)
                            await asyncio.sleep(delay)
                            continue
                        else:
                            attempt_errors.append(f"{provider}/{model_name} RateLimited: {e}")
                            break
                    else:
                        logger.warning(f"Error on {provider}/{model_name}: {e}. Cascading to fallback model...")
                        attempt_errors.append(f"{provider}/{model_name} Error: {e}")
                        break

        # All models failed
        logger.error(f"All LLM candidates exhausted. Details: {attempt_errors}")
        raise RateLimitExceededException(
            "All LLM candidates exhausted or rate limited.",
            details={"errors": attempt_errors},
        )

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams response tokens across the fallback chain."""
        chain_to_try = self._build_execution_chain(selected_model)
        start_time = time.time()
        success = False

        for idx, model_info in enumerate(chain_to_try):
            provider = model_info.get("provider", "gemini").lower()
            model_name = model_info.get("model", "")
            rpm_limit = settings.GEMINI_RPM_LIMIT if provider == "gemini" else settings.GROQ_RPM_LIMIT

            allowed, reason = self.rate_limiter.can_proceed(provider, model_name, rpm_limit=rpm_limit)
            if not allowed:
                logger.warning(f"Stream: Skipping {provider}/{model_name} - {reason}")
                continue

            try:
                logger.info(f"Initiating stream with {provider}/{model_name} (step {idx + 1}/{len(chain_to_try)})")
                if provider == "gemini":
                    stream_gen = self.gemini.generate_stream(
                        model=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                elif provider == "groq":
                    stream_gen = self.groq.generate_stream(
                        model=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                else:
                    continue

                # Stream tokens
                yielded_any = False
                for token in stream_gen:
                    yielded_any = True
                    yield {
                        "chunk": token,
                        "model_used": model_name,
                        "provider": provider,
                        "fallback_triggered": (idx > 0),
                        "done": False,
                    }

                self.rate_limiter.record_success(provider, model_name)
                success = True

                yield {
                    "chunk": "",
                    "model_used": model_name,
                    "provider": provider,
                    "fallback_triggered": (idx > 0),
                    "latency_sec": round(time.time() - start_time, 2),
                    "done": True,
                }
                return

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str
                if is_rate_limit:
                    retry_after = self._parse_retry_after(str(e))
                    self.rate_limiter.record_rate_limit_error(provider, model_name, retry_after=retry_after)
                logger.warning(f"Stream failed on {provider}/{model_name}: {e}. Cascading to next candidate...")

        if not success:
            yield {
                "chunk": "Error: All language models in the priority fallback cascade are currently exhausted or rate-limited. Please retry in a few moments.",
                "model_used": "none",
                "provider": "none",
                "fallback_triggered": True,
                "done": True,
            }

    async def generate_stream_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        selected_model: Optional[Dict[str, str]] = None,
        temperature: float = 0.4,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streams response tokens asynchronously across the fallback chain."""
        chain_to_try = self._build_execution_chain(selected_model)
        start_time = time.time()
        success = False

        for idx, model_info in enumerate(chain_to_try):
            provider = model_info.get("provider", "gemini").lower()
            model_name = model_info.get("model", "")
            rpm_limit = settings.GEMINI_RPM_LIMIT if provider == "gemini" else settings.GROQ_RPM_LIMIT

            allowed, reason = self.rate_limiter.can_proceed(provider, model_name, rpm_limit=rpm_limit)
            if not allowed:
                logger.warning(f"Async Stream: Skipping {provider}/{model_name} - {reason}")
                continue

            try:
                logger.info(f"Initiating async stream with {provider}/{model_name} (step {idx + 1}/{len(chain_to_try)})")
                if provider == "gemini":
                    # We have generate_stream_async in gemini_client
                    stream_gen = self.gemini.generate_stream_async(
                        model=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                    
                    yielded_any = False
                    async for token in stream_gen:
                        yielded_any = True
                        yield {
                            "chunk": token,
                            "model_used": model_name,
                            "provider": provider,
                            "fallback_triggered": (idx > 0),
                            "done": False,
                        }
                elif provider == "groq":
                    # groq doesn't have an async stream yet, we mock an async generator by converting sync generator
                    # For production, we'd add async to groq_client too.
                    stream_gen = self.groq.generate_stream(
                        model=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                    
                    yielded_any = False
                    for token in stream_gen:
                        yielded_any = True
                        yield {
                            "chunk": token,
                            "model_used": model_name,
                            "provider": provider,
                            "fallback_triggered": (idx > 0),
                            "done": False,
                        }
                        await asyncio.sleep(0) # yield control to event loop
                else:
                    continue

                self.rate_limiter.record_success(provider, model_name)
                success = True

                yield {
                    "chunk": "",
                    "model_used": model_name,
                    "provider": provider,
                    "fallback_triggered": (idx > 0),
                    "latency_sec": round(time.time() - start_time, 2),
                    "done": True,
                }
                return

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str
                if is_rate_limit:
                    retry_after = self._parse_retry_after(str(e))
                    self.rate_limiter.record_rate_limit_error(provider, model_name, retry_after=retry_after)
                logger.warning(f"Async Stream failed on {provider}/{model_name}: {e}. Cascading to next candidate...")

        if not success:
            yield {
                "chunk": "Error: All language models in the priority fallback cascade are currently exhausted or rate-limited. Please retry in a few moments.",
                "model_used": "none",
                "provider": "none",
                "fallback_triggered": True,
                "done": True,
            }

    def get_model_catalog(self) -> List[Dict[str, Any]]:
        """Returns catalog of models combined with real-time rate limit telemetry."""
        status = self.rate_limiter.get_status()
        catalog: List[Dict[str, Any]] = []

        for m in self.model_chain:
            provider = m["provider"]
            model = m["model"]
            key = f"{provider.lower()}:{model.lower()}"
            telemetry = status.get(key, {})

            catalog.append({
                "provider": provider,
                "model": model,
                "in_cooldown": telemetry.get("in_cooldown", False),
                "cooldown_remaining_sec": telemetry.get("cooldown_remaining_sec", 0.0),
                "requests_last_minute": telemetry.get("requests_last_minute", 0),
                "tokens_available": telemetry.get("tokens_available", 0.0),
            })
        return catalog

    @staticmethod
    def _parse_retry_after(error_msg: str) -> Optional[int]:
        """Extracts seconds from retry-after messages if present."""
        match = re.search(r"retry\s+after\s+(\d+)\s+seconds", error_msg, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match_s = re.search(r"(\d+)\s*s\b", error_msg)
        if match_s:
            return int(match_s.group(1))
        return None


llm_client = ResilientLLMClient()
LLMClient = ResilientLLMClient
