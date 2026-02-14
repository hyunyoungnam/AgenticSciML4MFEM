"""
OpenAI LLM provider implementation.

Provides integration with OpenAI's API (GPT-4, GPT-4-turbo, etc.)
"""

import os
from typing import Any, Dict, List, Optional

from agents.llm.provider import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, and other OpenAI models.
    """

    SUPPORTED_MODELS = [
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            base_url: Custom API base URL (for proxies/alternatives)
            **kwargs: Additional configuration
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.base_url = base_url
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4-turbo"

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai>=1.0.0"
                )

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.organization:
                client_kwargs["organization"] = self.organization
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = AsyncOpenAI(**client_kwargs)

        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI API.

        Args:
            system_prompt: System message
            user_prompt: User message
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters (top_p, presence_penalty, etc.)

        Returns:
            LLMResponse with generated content
        """
        client = self._get_client()
        model = model or self.default_model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add optional parameters
        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if param in kwargs:
                request_params[param] = kwargs[param]

        response = await client.chat.completions.create(**request_params)

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            metadata={
                "id": response.id,
                "created": response.created,
            },
        )

    async def generate_with_history(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response with conversation history.

        Args:
            system_prompt: System message
            messages: List of {"role": "user/assistant", "content": "..."}
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        client = self._get_client()
        model = model or self.default_model

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        request_params = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if param in kwargs:
                request_params[param] = kwargs[param]

        response = await client.chat.completions.create(**request_params)

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            metadata={
                "id": response.id,
                "created": response.created,
            },
        )

    def validate_model(self, model: str) -> bool:
        """Check if a model is supported."""
        return model in self.SUPPORTED_MODELS or model.startswith("gpt-")

    async def check_connection(self) -> bool:
        """
        Check if the API connection is working.

        Returns:
            True if connection is successful
        """
        try:
            client = self._get_client()
            # Make a minimal API call to check connection
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False
