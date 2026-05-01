"""
Anthropic LLM provider implementation.

Provides integration with Anthropic's API (Claude models).
"""

import os
from typing import Any, Dict, List, Optional

from piano.agents.llm.provider import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider.

    Supports Claude 4 models (Opus 4.7, Sonnet 4.6, Haiku 4.5).
    """

    SUPPORTED_MODELS = [
        # Claude 4 (current)
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        # Claude 3.5 (legacy)
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        # Claude 3 (legacy)
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    # Model aliases for convenience
    MODEL_ALIASES = {
        # Claude 4 aliases
        "claude-opus": "claude-opus-4-7",
        "claude-sonnet": "claude-sonnet-4-6",
        "claude-haiku": "claude-haiku-4-5-20251001",
        # Legacy Claude 3 aliases
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL (optional)
            **kwargs: Additional configuration
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-haiku-4-5-20251001"

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model name."""
        return self.MODEL_ALIASES.get(model, model)

    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic>=0.18.0"
                )

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = AsyncAnthropic(**client_kwargs)

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
        Generate a response using Anthropic API.

        Args:
            system_prompt: System message
            user_prompt: User message
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters (top_p, top_k, etc.)

        Returns:
            LLMResponse with generated content
        """
        client = self._get_client()
        model = self._resolve_model(model or self.default_model)

        messages = [
            {"role": "user", "content": user_prompt},
        ]

        # Build request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add optional parameters
        if "top_p" in kwargs:
            request_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            request_params["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs:
            request_params["stop_sequences"] = kwargs["stop_sequences"]

        response = await client.messages.create(**request_params)

        # Extract content from response
        content = ""
        if response.content:
            content = response.content[0].text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            model=response.model,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
            metadata={
                "id": response.id,
                "type": response.type,
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
        model = self._resolve_model(model or self.default_model)

        # Anthropic requires alternating user/assistant messages
        # Ensure the format is correct
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            # Anthropic only accepts "user" or "assistant"
            if role not in ("user", "assistant"):
                role = "user"
            formatted_messages.append({
                "role": role,
                "content": msg.get("content", ""),
            })

        request_params = {
            "model": model,
            "messages": formatted_messages,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if "top_p" in kwargs:
            request_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            request_params["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs:
            request_params["stop_sequences"] = kwargs["stop_sequences"]

        response = await client.messages.create(**request_params)

        content = ""
        if response.content:
            content = response.content[0].text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            model=response.model,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
            metadata={
                "id": response.id,
                "type": response.type,
            },
        )

    def validate_model(self, model: str) -> bool:
        """Check if a model is supported."""
        resolved = self._resolve_model(model)
        return resolved in self.SUPPORTED_MODELS or model.startswith("claude-")

    async def check_connection(self) -> bool:
        """
        Check if the API connection is working.

        Returns:
            True if connection is successful
        """
        try:
            client = self._get_client()
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False
