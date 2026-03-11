"""
Claude Code CLI provider.

Provides integration with Claude Code CLI for agentic task execution.
Claude Code can read files, execute code, and perform multi-step tasks
with built-in error recovery.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from inpforge.agents.llm.provider import LLMProvider, LLMResponse


@dataclass
class ClaudeCodeResult:
    """Result from a Claude Code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    session_id: Optional[str] = None
    num_turns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaudeCodeProvider(LLMProvider):
    """
    LLM Provider that uses Claude Code CLI for execution.

    This provider invokes the `claude` CLI in non-interactive mode,
    allowing Claude to use tools like file reading, code execution,
    and bash commands to complete tasks.

    Key advantages over standard LLM calls:
    - Can read actual source files to understand current API
    - Can execute code and see results
    - Can iterate on errors and self-correct
    - Returns verified results, not just generated text
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonnet",
        max_turns: int = 20,
        allowed_tools: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize the Claude Code provider.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model to use (sonnet, opus, haiku)
            max_turns: Maximum agentic turns before stopping
            allowed_tools: List of allowed tools (Edit, Write, Read, Bash, etc.)
            working_dir: Working directory for Claude Code
            timeout: Timeout in seconds for CLI execution
        """
        super().__init__(api_key)
        self.model = model
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["Edit", "Write", "Read", "Bash", "Glob", "Grep"]
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "claude-code"

    @property
    def default_model(self) -> str:
        return "sonnet"

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
        Generate a response using Claude Code CLI.

        The system_prompt and user_prompt are combined into a single
        task prompt for Claude Code.

        Args:
            system_prompt: System context (role, constraints)
            user_prompt: The actual task to perform
            model: Model override (sonnet, opus, haiku)
            temperature: Ignored (Claude Code manages this)
            max_tokens: Ignored (Claude Code manages this)
            **kwargs: Additional arguments
                - allowed_tools: Override allowed tools
                - max_turns: Override max turns
                - output_format: "text" or "json"

        Returns:
            LLMResponse with the result
        """
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        # Get overrides from kwargs
        allowed_tools = kwargs.get("allowed_tools", self.allowed_tools)
        max_turns = kwargs.get("max_turns", self.max_turns)
        output_format = kwargs.get("output_format", "text")

        # Execute Claude Code
        result = await self._execute_claude_code(
            prompt=full_prompt,
            model=model or self.model,
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            output_format=output_format,
        )

        return LLMResponse(
            content=result.output,
            model=f"claude-code-{model or self.model}",
            finish_reason="stop" if result.success else "error",
            usage={
                "num_turns": result.num_turns,
                "cost_usd": result.cost_usd,
                "duration_ms": result.duration_ms,
            },
            metadata={
                "session_id": result.session_id,
                "success": result.success,
                "error": result.error,
            }
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
        Generate with conversation history.

        For Claude Code, we combine history into the prompt.
        """
        # Combine messages into prompt
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        )

        user_prompt = f"Conversation history:\n{history_text}"

        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def execute_task(
        self,
        task: str,
        context_files: Optional[List[str]] = None,
        output_format: str = "json",
        max_turns: Optional[int] = None,
    ) -> ClaudeCodeResult:
        """
        Execute a task with Claude Code.

        This is a higher-level method that's more suited for agentic tasks
        than the standard generate() method.

        Args:
            task: Task description/prompt
            context_files: Files to read before starting task
            output_format: "text" or "json"
            max_turns: Override max turns

        Returns:
            ClaudeCodeResult with execution details
        """
        # Build prompt with context
        if context_files:
            context_prompt = "First, read and understand these files:\n"
            for f in context_files:
                context_prompt += f"- {f}\n"
            context_prompt += "\nThen:\n"
            task = context_prompt + task

        return await self._execute_claude_code(
            prompt=task,
            model=self.model,
            allowed_tools=self.allowed_tools,
            max_turns=max_turns or self.max_turns,
            output_format=output_format,
        )

    async def _execute_claude_code(
        self,
        prompt: str,
        model: str,
        allowed_tools: List[str],
        max_turns: int,
        output_format: str = "text",
    ) -> ClaudeCodeResult:
        """
        Execute the Claude Code CLI.

        Args:
            prompt: The prompt/task
            model: Model to use
            allowed_tools: Tools Claude can use
            max_turns: Maximum turns
            output_format: Output format

        Returns:
            ClaudeCodeResult
        """
        # Build command
        cmd = [
            "claude",
            "-p", prompt,  # Non-interactive print mode
            "--model", model,
            "--max-turns", str(max_turns),
            "--output-format", output_format,
            "--dangerously-skip-permissions",  # For automation
        ]

        # Add allowed tools
        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        # Run the command
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                # Parse output based on format
                if output_format == "json":
                    try:
                        data = json.loads(stdout_str)
                        return ClaudeCodeResult(
                            success=True,
                            output=data.get("result", stdout_str),
                            cost_usd=data.get("cost_usd", 0.0),
                            duration_ms=data.get("duration_ms", 0),
                            session_id=data.get("session_id"),
                            num_turns=data.get("num_turns", 0),
                            metadata=data,
                        )
                    except json.JSONDecodeError:
                        # Not valid JSON, return as text
                        return ClaudeCodeResult(
                            success=True,
                            output=stdout_str,
                        )
                else:
                    return ClaudeCodeResult(
                        success=True,
                        output=stdout_str,
                    )
            else:
                return ClaudeCodeResult(
                    success=False,
                    output=stdout_str,
                    error=stderr_str or f"Exit code: {proc.returncode}",
                )

        except asyncio.TimeoutError:
            return ClaudeCodeResult(
                success=False,
                output="",
                error=f"Timeout after {self.timeout} seconds",
            )
        except FileNotFoundError:
            return ClaudeCodeResult(
                success=False,
                output="",
                error="Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
            )
        except Exception as e:
            return ClaudeCodeResult(
                success=False,
                output="",
                error=str(e),
            )

    def execute_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        max_turns: Optional[int] = None,
    ) -> ClaudeCodeResult:
        """
        Synchronous execution wrapper.

        Args:
            prompt: Task prompt
            model: Model override
            allowed_tools: Tools override
            max_turns: Max turns override

        Returns:
            ClaudeCodeResult
        """
        return asyncio.run(
            self._execute_claude_code(
                prompt=prompt,
                model=model or self.model,
                allowed_tools=allowed_tools or self.allowed_tools,
                max_turns=max_turns or self.max_turns,
            )
        )
