"""
LLM Provider abstraction layer.

Supports multiple LLM backends (OpenAI, Anthropic, Claude Code) with a unified interface.
"""

from inpforge.agents.llm.provider import LLMProvider, LLMResponse, create_provider
from inpforge.agents.llm.claude_code_provider import ClaudeCodeProvider, ClaudeCodeResult

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "create_provider",
    "ClaudeCodeProvider",
    "ClaudeCodeResult",
]
