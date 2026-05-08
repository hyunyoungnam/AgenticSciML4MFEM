"""
Engineer Agent implementation.

Uses Claude Code CLI to implement code-level changes that cannot be
expressed as hyperparameter config updates — e.g. fixing a loss term
normalization, adding a new physics constraint, or patching a training bug.

Invoked by AgenticSurrogateTrainer when the Architect flags a change that
requires editing source files rather than just tweaking config values.

On failure, delegates to DebuggerAgent to diagnose the traceback and then
makes a second, more targeted implementation attempt.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from piano.agents.llm.claude_code_provider import ClaudeCodeProvider

logger = logging.getLogger(__name__)

_ENGINEER_SYSTEM = (
    "You are implementing a targeted code change inside an Agentic SciML framework "
    "(piano/). The framework trains neural operators (DeepONet, Transolver) for physics "
    "simulations with PINO losses. Your task is precisely scoped — do NOT refactor "
    "unrelated code, do NOT add new features beyond what is requested, and do NOT remove "
    "existing functionality unless explicitly told to."
)


@dataclass
class EngineerResult:
    success: bool
    changes_made: str = ""
    error: Optional[str] = None
    debug_attempted: bool = False


class EngineerAgent:
    """
    Engineer Agent that uses Claude Code CLI to apply code-level changes.

    The Architect/Physicist diagnose issues and propose changes; when a change
    requires editing source files (not just config values) this agent is invoked
    to implement it autonomously.

    On first failure, DebuggerAgent diagnoses the traceback and the engineer
    retries with the debugger's fix description.
    """

    def __init__(
        self,
        working_dir: str,
        model: str = "sonnet",
        max_turns: int = 15,
        timeout: int = 300,
        use_debugger: bool = True,
    ):
        self._provider = ClaudeCodeProvider(
            model=model,
            max_turns=max_turns,
            working_dir=working_dir,
            timeout=timeout,
            allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
        )
        self._working_dir = working_dir
        self._debugger = None
        if use_debugger:
            from piano.agents.roles.debugger import DebuggerAgent
            self._debugger = DebuggerAgent(
                working_dir=working_dir,
                model=model,
                timeout=timeout,
            )

    async def implement_change(
        self,
        change_description: str,
        context_files: Optional[List[str]] = None,
        validation_command: Optional[str] = None,
    ) -> EngineerResult:
        """
        Implement a code change described in natural language.

        On failure, DebuggerAgent diagnoses the error and one retry is made.

        Args:
            change_description: What to change and why (from Architect/Physicist)
            context_files: Files most relevant to the change (read first)
            validation_command: Shell command to validate after editing (e.g. pytest)
        """
        result = await self._attempt(change_description, context_files, validation_command)

        if not result.success and self._debugger is not None:
            logger.info("EngineerAgent: first attempt failed, delegating to DebuggerAgent")
            debug = await self._debugger.diagnose(
                traceback=result.error or "",
                attempted_change=change_description,
                context_files=context_files,
            )
            logger.info(f"DebuggerAgent: root_cause={debug.root_cause[:120]}")
            logger.info(f"DebuggerAgent: fix={debug.fix_description[:120]}")

            retry_desc = debug.to_engineer_prompt(change_description)
            retry_files = debug.files_to_change or context_files
            result = await self._attempt(retry_desc, retry_files, validation_command)
            result.debug_attempted = True

        return result

    async def _attempt(
        self,
        change_description: str,
        context_files: Optional[List[str]],
        validation_command: Optional[str],
    ) -> EngineerResult:
        task = (
            f"CHANGE REQUIRED:\n{change_description}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the relevant source files to understand the current implementation.\n"
            "2. Implement the minimal change that addresses the issue.\n"
            "3. Do NOT add unrelated improvements or refactors.\n"
        )
        if context_files:
            task += f"4. Start by reading: {', '.join(context_files)}\n"
        if validation_command:
            task += f"5. After editing, run `{validation_command}` and confirm it passes.\n"
        task += "\nReport what files you changed and what you changed, concisely."

        result = await self._provider.execute_task(
            task=task,
            context_files=context_files,
            output_format="text",
        )

        if result.success:
            logger.info("EngineerAgent: change applied successfully")
            return EngineerResult(success=True, changes_made=result.output)
        else:
            logger.error(f"EngineerAgent: change failed — {result.error}")
            return EngineerResult(success=False, error=result.error or result.output)

    def implement_change_sync(
        self,
        change_description: str,
        context_files: Optional[List[str]] = None,
        validation_command: Optional[str] = None,
    ) -> EngineerResult:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.implement_change(change_description, context_files, validation_command)
            )
        finally:
            loop.close()
