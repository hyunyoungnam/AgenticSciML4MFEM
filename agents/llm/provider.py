"""
Abstract LLM provider interface.

Defines the common interface for all LLM backends (OpenAI, Anthropic, etc.)
and provides a factory function for creating providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio


@dataclass
class LLMResponse:
    """
    Response from an LLM provider.

    Attributes:
        content: The generated text content
        model: Model that generated the response
        finish_reason: Why generation stopped (stop, length, etc.)
        usage: Token usage statistics
        metadata: Additional provider-specific metadata
    """
    content: str
    model: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        """Number of prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Number of completion tokens used."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations handle the specifics of each provider's API
    while exposing a unified interface to the agent system.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider (can also be set via env var)
        """
        self.api_key = api_key

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        pass

    @abstractmethod
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
        Generate a response from the LLM.

        Args:
            system_prompt: System message setting context
            user_prompt: User message with the actual request
            model: Model to use (defaults to provider's default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
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
            system_prompt: System message setting context
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the generated content
        """
        pass

    def generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """
        Synchronous wrapper for generate().

        Args:
            Same as generate()

        Returns:
            LLMResponse with the generated content
        """
        return asyncio.run(
            self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

    def validate_model(self, model: str) -> bool:
        """
        Check if a model is supported by this provider.

        Args:
            model: Model identifier

        Returns:
            True if model is supported
        """
        return True  # Override in implementations

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name})"


def create_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'claude-code')
        api_key: API key (optional, can use env vars)
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    provider_name_lower = provider_name.lower()

    if provider_name_lower == "openai":
        from agents.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, **kwargs)

    elif provider_name_lower == "anthropic":
        from agents.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, **kwargs)

    elif provider_name_lower == "claude-code":
        from agents.llm.claude_code_provider import ClaudeCodeProvider
        return ClaudeCodeProvider(api_key=api_key, **kwargs)

    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: openai, anthropic, claude-code"
        )


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses without making actual API calls.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize mock provider.

        Args:
            responses: Dict mapping patterns to responses
        """
        super().__init__()
        self.responses = responses or {}
        self.call_history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock response."""
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })

        # Find matching response
        content = "Mock response for testing."
        for pattern, response in self.responses.items():
            if pattern in user_prompt or pattern in system_prompt:
                content = response
                break

        return LLMResponse(
            content=content,
            model=model or self.default_model,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
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
        """Generate a mock response with history."""
        # Combine messages for pattern matching
        combined = " ".join(m.get("content", "") for m in messages)

        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=combined,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
