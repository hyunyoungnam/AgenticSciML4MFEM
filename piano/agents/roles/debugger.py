"""
Debugger Agent.

Called by EngineerAgent when code execution produces an error traceback.
Analyzes the traceback, identifies the root cause, and returns a targeted
fix description that EngineerAgent can apply.

Kept separate from EngineerAgent so that:
- EngineerAgent focuses on implementing changes (writes code)
- DebuggerAgent focuses on diagnosing failures (reads tracebacks)

Both use ClaudeCodeProvider but with different prompts and allowed tools.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from piano.agents.llm.claude_code_provider import ClaudeCodeProvider

logger = logging.getLogger(__name__)

_DEBUGGER_SYSTEM = (
    "You are a senior Python/PyTorch debugger for a physics-informed machine learning framework (piano/). "
    "You receive a traceback or test failure and a short description of what change was attempted. "
    "Your ONLY task is to:\n"
    "1. Identify the root cause of the error\n"
    "2. State the minimal fix required (do NOT over-engineer)\n"
    "3. Confirm which file(s) need to change and which lines\n\n"
    "Do NOT attempt to implement the fix yourself. Return a precise fix description "
    "that EngineerAgent can execute. Keep your response under 400 words."
)


@dataclass
class DebugResult:
    """Structured output of the Debugger Agent."""
    root_cause: str = ""
    fix_description: str = ""
    files_to_change: List[str] = None
    confidence: str = "medium"
    raw_response: str = ""

    def __post_init__(self):
        if self.files_to_change is None:
            self.files_to_change = []

    def to_engineer_prompt(self, original_change: str) -> str:
        return (
            f"ORIGINAL CHANGE ATTEMPTED:\n{original_change}\n\n"
            f"ERROR ROOT CAUSE:\n{self.root_cause}\n\n"
            f"MINIMAL FIX:\n{self.fix_description}\n\n"
            f"FILES TO CHANGE: {', '.join(self.files_to_change) or 'unknown'}"
        )


class DebuggerAgent:
    """
    Debugger Agent — diagnoses code failures and produces fix descriptions.

    Called by EngineerAgent on failure; returns a DebugResult that EngineerAgent
    uses to make a second, more targeted implementation attempt.
    """

    def __init__(
        self,
        working_dir: str,
        model: str = "sonnet",
        max_turns: int = 8,
        timeout: int = 180,
    ):
        self._provider = ClaudeCodeProvider(
            model=model,
            max_turns=max_turns,
            working_dir=working_dir,
            timeout=timeout,
            allowed_tools=["Read", "Bash", "Glob", "Grep"],
        )
        self._working_dir = working_dir

    async def diagnose(
        self,
        traceback: str,
        attempted_change: str,
        context_files: Optional[List[str]] = None,
    ) -> DebugResult:
        """
        Diagnose a code failure.

        Args:
            traceback: Full error traceback from the failed execution
            attempted_change: Description of the change EngineerAgent tried to make
            context_files: Files involved in the failure (for context reading)

        Returns:
            DebugResult with root cause and fix description
        """
        task = (
            f"TRACEBACK:\n```\n{traceback}\n```\n\n"
            f"ATTEMPTED CHANGE:\n{attempted_change}\n\n"
        )
        if context_files:
            task += f"RELEVANT FILES (read these to understand context): {', '.join(context_files)}\n\n"

        task += (
            "Identify root cause and provide a minimal fix description. "
            "Use the format:\n"
            "ROOT_CAUSE: <one sentence>\n"
            "FIX: <precise description of the minimal change>\n"
            "FILES: <comma-separated file paths>\n"
            "CONFIDENCE: <high|medium|low>"
        )

        # Prepend system context to the task since ClaudeCodeProvider
        # doesn't support a separate system_override parameter
        full_task = f"[CONTEXT]\n{_DEBUGGER_SYSTEM}\n\n[TASK]\n{task}"
        result = await self._provider.execute_task(
            task=full_task,
            context_files=context_files,
            output_format="text",
        )

        if not result.success:
            logger.warning(f"DebuggerAgent: diagnosis failed — {result.error}")
            return DebugResult(
                root_cause="Debugger could not diagnose",
                fix_description=result.error or "Unknown error",
                confidence="low",
            )

        return self._parse(result.output, traceback)

    def _parse(self, response: str, traceback: str) -> DebugResult:
        import re
        result = DebugResult(raw_response=response)

        m = re.search(r"ROOT_CAUSE:\s*(.+?)(?=FIX:|FILES:|CONFIDENCE:|$)", response, re.DOTALL)
        if m:
            result.root_cause = m.group(1).strip()

        m = re.search(r"FIX:\s*(.+?)(?=FILES:|CONFIDENCE:|$)", response, re.DOTALL)
        if m:
            result.fix_description = m.group(1).strip()

        m = re.search(r"FILES:\s*(.+?)(?=CONFIDENCE:|$)", response, re.DOTALL)
        if m:
            files_str = m.group(1).strip()
            result.files_to_change = [f.strip() for f in files_str.split(",") if f.strip()]

        m = re.search(r"CONFIDENCE:\s*(\S+)", response, re.IGNORECASE)
        if m:
            result.confidence = m.group(1).strip(".,").lower()

        # Fallback: extract file paths from traceback
        if not result.files_to_change:
            file_matches = re.findall(r'File "([^"]+\.py)"', traceback)
            result.files_to_change = list(dict.fromkeys(
                f for f in file_matches if "site-packages" not in f
            ))

        return result

    def diagnose_sync(
        self,
        traceback: str,
        attempted_change: str,
        context_files: Optional[List[str]] = None,
    ) -> DebugResult:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.diagnose(traceback, attempted_change, context_files)
            )
        finally:
            loop.close()
