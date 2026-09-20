import os
from typing import Generator, Optional

from dotenv import load_dotenv
from groq import Groq

from backend.app.core.errors import LLMProviderException
from backend.app.core.logging import logger

load_dotenv()



class GroqClient:
    """Encapsulates interaction with Groq API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initializes Groq API client configuration."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        """Initializes and returns the Groq client instance."""
        if not self._client:
            if not self.api_key:
                raise LLMProviderException(
                    "GROQ_API_KEY is not configured.",
                    details={"provider": "groq"},
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_text(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        """Executes non-streaming generation via Groq."""
        messages: list = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        chat_completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content or ""

    def generate_stream(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """Streams text chunks via Groq."""
        messages: list = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


groq_client = GroqClient()
