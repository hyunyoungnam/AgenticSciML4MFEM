"""
Claude Code-based Debugger Agent.

Replaces the hand-coded DebuggerAgent with a Claude Code-powered implementation
that can investigate errors, read logs, examine files, and test fixes.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentRole
from agents.llm.claude_code_provider import ClaudeCodeProvider, ClaudeCodeResult
from agents.roles.debugger import (
    DebugAnalysis,
    ErrorCategory,
    ErrorSeverity,
    RetryStrategy,
)


@dataclass
class ClaudeCodeDebuggerConfig:
    """Configuration for the Claude Code Debugger."""
    model: str = "sonnet"
    max_turns: int = 20
    timeout: int = 240
    working_dir: Optional[str] = None

    # Files to read for debugging context
    context_files: List[str] = field(default_factory=lambda: [
        "manager.py",
        "morphing.py",
        "validator.py",
    ])


class ClaudeCodeDebugger:
    """
    Debugger Agent powered by Claude Code.

    Unlike the hand-coded DebuggerAgent, this agent:
    1. Can read actual error logs and source files
    2. Can investigate by examining the codebase
    3. Can test hypotheses by running diagnostic code
    4. Provides verified, actionable fixes

    Key advantages:
    - Investigative capability: Can trace through code
    - Context-aware: Reads actual files, not static prompts
    - Verification: Can test fixes before suggesting
    - Adaptive: Handles novel error patterns
    """

    def __init__(
        self,
        config: Optional[ClaudeCodeDebuggerConfig] = None,
        provider: Optional[ClaudeCodeProvider] = None,
    ):
        """
        Initialize the Claude Code Debugger.

        Args:
            config: Configuration options
            provider: Claude Code provider (created if not provided)
        """
        self.config = config or ClaudeCodeDebuggerConfig()
        self.provider = provider or ClaudeCodeProvider(
            model=self.config.model,
            max_turns=self.config.max_turns,
            timeout=self.config.timeout,
            working_dir=self.config.working_dir,
            allowed_tools=["Read", "Bash", "Glob", "Grep"],
        )
        self.role = AgentRole.DEBUGGER

    async def diagnose_error(
        self,
        context: AgentContext,
        error_type: str,
        error_message: str,
        mutation: str = "",
        delta_R: Optional[float] = None,
        logs: str = "",
    ) -> DebugAnalysis:
        """
        Diagnose an error using Claude Code.

        Claude Code will:
        1. Analyze the error message
        2. Read relevant source files
        3. Search for similar patterns in codebase
        4. Identify root cause
        5. Suggest specific, actionable fixes

        Args:
            context: Agent context
            error_type: Type of error (implementation, evaluation, etc.)
            error_message: Error message
            mutation: Description of the mutation that caused the error
            delta_R: Delta R value used
            logs: Recent logs

        Returns:
            DebugAnalysis with diagnosis and fixes
        """
        task = self._build_diagnosis_task(
            error_type=error_type,
            error_message=error_message,
            mutation=mutation,
            delta_R=delta_R,
            logs=logs,
            context=context,
        )

        result = await self.provider.execute_task(
            task=task,
            context_files=self.config.context_files,
            output_format="text",
        )

        return self._parse_diagnosis(result, error_message)

    def _build_diagnosis_task(
        self,
        error_type: str,
        error_message: str,
        mutation: str,
        delta_R: Optional[float],
        logs: str,
        context: AgentContext,
    ) -> str:
        """Build the diagnosis task prompt."""

        task = f"""
You are debugging an FEA model generation error. Investigate and provide a diagnosis.

## Error Information
- **Error Type**: {error_type}
- **Error Message**:
```
{error_message}
```

## Context
- **Current delta_R**: {delta_R if delta_R is not None else 'Not specified'}
- **Mutation applied**: {mutation if mutation else 'Not specified'}

## Recent Logs
```
{logs if logs else 'No logs available'}
```

## Previous Failures
{self._format_failure_history(context.failure_history)}

## Investigation Steps
1. Read the relevant source files to understand what might have gone wrong
2. Search for the error pattern in the codebase
3. Identify the root cause
4. Determine if this is recoverable or fatal

## Required Output
Provide your analysis in this EXACT format:

**Category**: [mesh|convergence|material|bc|implementation|solver|other]
**Severity**: [recoverable|requires_rollback|fatal]

**Root Cause Analysis**:
[Detailed explanation of what caused the error]

**Primary Fix**:
[The most likely fix to resolve the issue]

**Alternative Fixes**:
- [Alternative fix 1]
- [Alternative fix 2]

**Prevention**:
[How to prevent this error in future]

**Suggested delta_R**: [number or "N/A" if not applicable]

```python
# Fix code if applicable
[Python code to fix the issue, or "N/A"]
```

Investigate now and provide your diagnosis.
"""
        return task

    def _format_failure_history(self, failures: List[Dict[str, Any]]) -> str:
        """Format failure history for the prompt."""
        if not failures:
            return "No previous failures recorded."

        history = []
        for i, f in enumerate(failures[:5], 1):  # Last 5 failures
            history.append(
                f"{i}. delta_R={f.get('delta_R', 'N/A')}: {f.get('error', 'Unknown error')[:100]}"
            )

        return "\n".join(history)

    def _parse_diagnosis(
        self,
        cc_result: ClaudeCodeResult,
        original_error: str,
    ) -> DebugAnalysis:
        """Parse Claude Code result into DebugAnalysis."""

        analysis = DebugAnalysis(raw_response=cc_result.output)

        if not cc_result.success:
            analysis.category = ErrorCategory.OTHER
            analysis.severity = ErrorSeverity.RECOVERABLE
            analysis.root_cause = f"Diagnosis failed: {cc_result.error}"
            return analysis

        output = cc_result.output

        # Extract category
        cat_match = re.search(r'\*\*Category\*\*:\s*\[?(\w+)', output, re.IGNORECASE)
        if cat_match:
            cat_text = cat_match.group(1).lower()
            for cat in ErrorCategory:
                if cat.value in cat_text or cat_text in cat.value:
                    analysis.category = cat
                    break

        # Extract severity
        sev_match = re.search(r'\*\*Severity\*\*:\s*\[?(\w+)', output, re.IGNORECASE)
        if sev_match:
            sev_text = sev_match.group(1).lower()
            if "recover" in sev_text:
                analysis.severity = ErrorSeverity.RECOVERABLE
            elif "rollback" in sev_text:
                analysis.severity = ErrorSeverity.REQUIRES_ROLLBACK
            elif "fatal" in sev_text:
                analysis.severity = ErrorSeverity.FATAL

        # Extract root cause
        root_match = re.search(
            r'\*\*Root Cause(?:\s+Analysis)?\*\*:?\s*\n?(.*?)(?=\*\*|$)',
            output, re.DOTALL | re.IGNORECASE
        )
        if root_match:
            analysis.root_cause = root_match.group(1).strip()

        # Extract primary fix
        primary_match = re.search(
            r'\*\*Primary Fix\*\*:?\s*\n?(.*?)(?=\*\*|$)',
            output, re.DOTALL | re.IGNORECASE
        )
        if primary_match:
            analysis.primary_fix = primary_match.group(1).strip()

        # Extract alternative fixes
        alt_match = re.search(
            r'\*\*Alternative Fixes\*\*:?\s*\n?(.*?)(?=\*\*|$)',
            output, re.DOTALL | re.IGNORECASE
        )
        if alt_match:
            text = alt_match.group(1)
            analysis.alternative_fixes = [
                line.strip().lstrip('- ').lstrip('0123456789. ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract prevention
        prevent_match = re.search(
            r'\*\*Prevention\*\*:?\s*\n?(.*?)(?=\*\*|```|$)',
            output, re.DOTALL | re.IGNORECASE
        )
        if prevent_match:
            analysis.prevention_advice = prevent_match.group(1).strip()

        # Extract suggested delta_R
        delta_r_match = re.search(
            r'\*\*Suggested delta_R\*\*:?\s*\[?([+-]?[0-9.]+)',
            output, re.IGNORECASE
        )
        if delta_r_match:
            try:
                analysis.suggested_delta_R = float(delta_r_match.group(1))
            except ValueError:
                pass

        # Extract fix code
        code_match = re.search(r'```python\s*(.*?)\s*```', output, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if code.lower() != "n/a" and code:
                analysis.fix_code = code

        return analysis

    async def diagnose_mesh_error(
        self,
        context: AgentContext,
        error_message: str,
        min_jacobian: float,
        problem_elements: List[int],
        delta_R: float,
        successful_delta_r: Optional[float] = None,
    ) -> DebugAnalysis:
        """
        Diagnose a mesh quality error.

        Claude Code will:
        1. Analyze the Jacobian values
        2. Read morphing.py to understand constraints
        3. Determine safe delta_R range
        4. Suggest specific config changes

        Args:
            context: Agent context
            error_message: Error message
            min_jacobian: Minimum Jacobian ratio
            problem_elements: List of problem element IDs
            delta_R: Delta R used
            successful_delta_r: Previous successful delta R

        Returns:
            DebugAnalysis focused on mesh issues
        """
        task = f"""
You are debugging a mesh quality error in FEA model morphing.

## Error Details
- **Error Message**: {error_message}
- **Minimum Jacobian**: {min_jacobian}
- **Problem Elements**: {problem_elements[:20]}... ({len(problem_elements)} total)
- **Current delta_R**: {delta_R}
- **Last successful delta_R**: {successful_delta_r if successful_delta_r else 'None'}

## Investigation Tasks
1. Read morphing.py to understand the morphing algorithm
2. Analyze why the Jacobian became negative/small
3. Determine a safe delta_R value based on the geometry
4. Suggest config changes to improve mesh quality

## Required Output
**Category**: mesh
**Severity**: [recoverable|requires_rollback|fatal]

**Root Cause Analysis**:
[Why did the mesh quality degrade?]

**Primary Fix**:
[The recommended fix]

**Alternative Fixes**:
- [Alternative 1]
- [Alternative 2]

**Prevention**:
[How to prevent this]

**Suggested delta_R**: [Safe value between 0 and {successful_delta_r or delta_R/2}]

Investigate and provide your diagnosis.
"""
        result = await self.provider.execute_task(
            task=task,
            context_files=["morphing.py", "validator.py"],
            output_format="text",
        )

        analysis = self._parse_diagnosis(result, error_message)
        analysis.category = ErrorCategory.MESH
        return analysis

    async def diagnose_convergence_failure(
        self,
        context: AgentContext,
        error_message: str,
        convergence_history: str = "",
        solver_settings: str = "",
        load_info: str = "",
    ) -> DebugAnalysis:
        """
        Diagnose a solver convergence failure.

        Args:
            context: Agent context
            error_message: Error message
            convergence_history: History of convergence attempts
            solver_settings: Current solver settings
            load_info: Loading conditions

        Returns:
            DebugAnalysis focused on convergence
        """
        task = f"""
You are debugging a solver convergence failure.

## Error Details
- **Error Message**: {error_message}

## Convergence History
```
{convergence_history if convergence_history else 'Not available'}
```

## Solver Settings
```
{solver_settings if solver_settings else 'Default settings'}
```

## Loading Info
```
{load_info if load_info else 'Not specified'}
```

## Investigation Tasks
1. Analyze the convergence pattern
2. Identify potential causes (material instability, contact issues, load stepping)
3. Suggest solver parameter adjustments
4. Recommend load stepping strategies if needed

## Required Output
**Category**: convergence
**Severity**: [recoverable|requires_rollback|fatal]

**Root Cause Analysis**:
[Why did convergence fail?]

**Primary Fix**:
[The recommended solver setting changes]

**Alternative Fixes**:
- [Alternative 1]
- [Alternative 2]

**Prevention**:
[How to prevent convergence issues]

**Suggested delta_R**: [If geometry change could help, or N/A]

Investigate and provide your diagnosis.
"""
        result = await self.provider.execute_task(
            task=task,
            output_format="text",
        )

        analysis = self._parse_diagnosis(result, error_message)
        analysis.category = ErrorCategory.CONVERGENCE
        return analysis

    async def suggest_retry_strategy(
        self,
        context: AgentContext,
        attempt_number: int,
        max_attempts: int,
        error_history: str,
        previous_fixes: str,
    ) -> RetryStrategy:
        """
        Suggest a retry strategy after failure.

        Claude Code will:
        1. Analyze the pattern of failures
        2. Determine if retry is worthwhile
        3. Suggest parameter adjustments
        4. Provide strategy description

        Args:
            context: Agent context
            attempt_number: Current attempt number
            max_attempts: Maximum allowed attempts
            error_history: History of errors
            previous_fixes: Previously attempted fixes

        Returns:
            RetryStrategy
        """
        remaining = max_attempts - attempt_number

        task = f"""
Analyze whether to retry this failed FEA operation.

## Attempt Info
- **Attempt**: {attempt_number} of {max_attempts}
- **Remaining attempts**: {remaining}

## Error History
{error_history}

## Previous Fixes Attempted
{previous_fixes if previous_fixes else 'None yet'}

## Failure Memory
{self._format_failure_history(context.failure_history)}

## Task
1. Analyze the pattern of failures
2. Determine if retry is likely to succeed
3. If yes, suggest specific parameter adjustments
4. If no, explain why we should stop

## Required Output (JSON format)
```json
{{
    "should_retry": true/false,
    "adjustments": {{
        "delta_R": <new value or null>,
        "other_params": {{}}
    }},
    "strategy_description": "<what to do differently>",
    "abandon_reason": "<if should_retry is false, explain why>"
}}
```

Provide your recommendation.
"""
        result = await self.provider.execute_task(
            task=task,
            output_format="text",
        )

        return self._parse_retry_strategy(result, attempt_number, max_attempts)

    def _parse_retry_strategy(
        self,
        cc_result: ClaudeCodeResult,
        attempt_number: int,
        max_attempts: int,
    ) -> RetryStrategy:
        """Parse Claude Code result into RetryStrategy."""

        strategy = RetryStrategy()

        if not cc_result.success:
            strategy.should_retry = attempt_number < max_attempts
            strategy.strategy_description = "Continue with default parameters"
            return strategy

        output = cc_result.output

        # Try to parse JSON
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                strategy.should_retry = data.get("should_retry", attempt_number < max_attempts)
                strategy.adjustments = data.get("adjustments", {})
                strategy.strategy_description = data.get("strategy_description", "")
                strategy.abandon_reason = data.get("abandon_reason")
                return strategy
            except json.JSONDecodeError:
                pass

        # Fallback: parse from text
        if "should_retry" in output.lower():
            if "false" in output.lower() or "no" in output.lower():
                strategy.should_retry = False
            else:
                strategy.should_retry = True

        # Extract delta_R suggestion
        delta_r_match = re.search(r'delta_R["\s:]+([+-]?[0-9.]+)', output)
        if delta_r_match:
            try:
                strategy.adjustments["delta_R"] = float(delta_r_match.group(1))
            except ValueError:
                pass

        # Extract strategy description
        desc_match = re.search(r'strategy_description["\s:]+["\']?(.*?)["\']?[,\}]', output)
        if desc_match:
            strategy.strategy_description = desc_match.group(1)

        return strategy

    def quick_classify_error(self, error_message: str) -> ErrorCategory:
        """
        Quickly classify an error without Claude Code.

        This is a fast fallback for simple cases.

        Args:
            error_message: The error message

        Returns:
            ErrorCategory
        """
        error_lower = error_message.lower()

        if any(x in error_lower for x in ["jacobian", "distort", "element", "mesh", "aspect"]):
            return ErrorCategory.MESH
        elif any(x in error_lower for x in ["converge", "iteration", "increment", "cutback"]):
            return ErrorCategory.CONVERGENCE
        elif any(x in error_lower for x in ["material", "elastic", "modulus", "poisson"]):
            return ErrorCategory.MATERIAL
        elif any(x in error_lower for x in ["boundary", "load", "constraint", "bc"]):
            return ErrorCategory.BOUNDARY_CONDITION
        elif any(x in error_lower for x in ["import", "syntax", "attribute", "type", "python"]):
            return ErrorCategory.IMPLEMENTATION
        elif any(x in error_lower for x in ["solver", "abaqus", "license"]):
            return ErrorCategory.SOLVER
        else:
            return ErrorCategory.OTHER

    async def investigate_error(
        self,
        error_message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> str:
        """
        Perform deep investigation of an error.

        This allows Claude Code to freely explore the codebase
        to understand the error.

        Args:
            error_message: The error message
            file_path: Optional file where error occurred
            line_number: Optional line number

        Returns:
            Investigation report as string
        """
        task = f"""
Investigate this error deeply. You have full access to read files and search the codebase.

## Error
```
{error_message}
```

{f'## Location: {file_path}:{line_number}' if file_path else ''}

## Investigation Tasks
1. Search for where this error might originate
2. Read relevant source files
3. Trace the execution path
4. Identify the root cause
5. Suggest how to fix it

Provide a detailed investigation report.
"""
        result = await self.provider.execute_task(
            task=task,
            output_format="text",
            max_turns=30,  # Allow more turns for deep investigation
        )

        return result.output
