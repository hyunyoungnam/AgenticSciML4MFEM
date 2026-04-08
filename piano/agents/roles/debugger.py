"""
Debugger Agent implementation.

The Debugger diagnoses errors from implementation or solver execution
and provides actionable fixes.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from piano.agents.base import BaseAgent, AgentContext, AgentRole
from piano.agents.prompts.debugger import DEBUGGER_PROMPTS


class ErrorCategory(Enum):
    """Categories of errors that can occur."""
    MESH = "mesh"
    CONVERGENCE = "convergence"
    MATERIAL = "material"
    BOUNDARY_CONDITION = "bc"
    IMPLEMENTATION = "implementation"
    SOLVER = "solver"
    OTHER = "other"


class ErrorSeverity(Enum):
    """Severity of the error."""
    RECOVERABLE = "recoverable"  # Can be fixed with parameter adjustment
    REQUIRES_ROLLBACK = "requires_rollback"  # Need to undo and try different approach
    FATAL = "fatal"  # Cannot continue with this solution


@dataclass
class DebugAnalysis:
    """Analysis from the Debugger agent."""
    category: ErrorCategory = ErrorCategory.OTHER
    severity: ErrorSeverity = ErrorSeverity.RECOVERABLE
    root_cause: str = ""
    primary_fix: str = ""
    alternative_fixes: List[str] = field(default_factory=list)
    prevention_advice: str = ""
    fix_code: str = ""
    suggested_delta_R: Optional[float] = None
    config_changes: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "root_cause": self.root_cause,
            "primary_fix": self.primary_fix,
            "alternative_fixes": self.alternative_fixes,
            "prevention_advice": self.prevention_advice,
            "fix_code": self.fix_code,
            "suggested_delta_R": self.suggested_delta_R,
            "config_changes": self.config_changes,
        }


@dataclass
class RetryStrategy:
    """Strategy for retrying after a failure."""
    should_retry: bool = True
    adjustments: Dict[str, Any] = field(default_factory=dict)
    strategy_description: str = ""
    abandon_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_retry": self.should_retry,
            "adjustments": self.adjustments,
            "strategy_description": self.strategy_description,
            "abandon_reason": self.abandon_reason,
        }


class DebuggerAgent(BaseAgent[DebugAnalysis]):
    """
    Debugger Agent that diagnoses errors and suggests fixes.

    Responsibilities:
    1. Classify and analyze errors
    2. Identify root causes
    3. Provide specific, actionable fixes
    4. Suggest retry strategies
    """

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.DEBUGGER,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return DEBUGGER_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "diagnose_error")

        if task == "diagnose_error":
            return self._build_diagnose_prompt(context, kwargs)
        elif task == "diagnose_mesh_error":
            return self._build_mesh_error_prompt(context, kwargs)
        elif task == "diagnose_convergence":
            return self._build_convergence_prompt(context, kwargs)
        elif task == "suggest_retry_strategy":
            return self._build_retry_prompt(context, kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _build_diagnose_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        return DEBUGGER_PROMPTS["diagnose_error"].format(
            error_type=kwargs.get("error_type", "unknown"),
            error_message=kwargs.get("error_message", ""),
            error_location=kwargs.get("error_location", "unknown"),
            solution_id=context.current_solution_id or "unknown",
            mutation=kwargs.get("mutation", ""),
            delta_R=kwargs.get("delta_R", "N/A"),
            logs=kwargs.get("logs", ""),
            model_info=str(context.model_info),
            fix_code="",  # Will be filled by LLM
        )

    def _build_mesh_error_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        return DEBUGGER_PROMPTS["diagnose_mesh_error"].format(
            error_message=kwargs.get("error_message", ""),
            min_jacobian=kwargs.get("min_jacobian", "N/A"),
            problem_elements=kwargs.get("problem_elements", []),
            delta_R=kwargs.get("delta_R", "N/A"),
            successful_delta_r=kwargs.get("successful_delta_r", "No previous successful runs"),
            config_changes="",  # Will be filled by LLM
        )

    def _build_convergence_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        return DEBUGGER_PROMPTS["diagnose_convergence"].format(
            error_message=kwargs.get("error_message", ""),
            convergence_history=kwargs.get("convergence_history", ""),
            solver_settings=kwargs.get("solver_settings", ""),
            load_info=kwargs.get("load_info", ""),
            suggested_initial="",
            suggested_min="",
            suggested_max="",
            suggested_max_iter="",
        )

    def _build_retry_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        return DEBUGGER_PROMPTS["suggest_retry_strategy"].format(
            solution_id=context.current_solution_id or "unknown",
            attempt_number=kwargs.get("attempt_number", 1),
            max_attempts=kwargs.get("max_attempts", 3),
            remaining_attempts=kwargs.get("max_attempts", 3) - kwargs.get("attempt_number", 1),
            error_history=kwargs.get("error_history", ""),
            previous_fixes=kwargs.get("previous_fixes", ""),
            adjustments="",
        )

    def parse_response(self, response: str) -> DebugAnalysis:
        """Parse the LLM response into a DebugAnalysis."""
        analysis = DebugAnalysis(raw_response=response)

        # Extract error category
        category_match = re.search(
            r'Category:\s*(\w+)',
            response, re.IGNORECASE
        )
        if category_match:
            cat_text = category_match.group(1).lower()
            for cat in ErrorCategory:
                if cat.value in cat_text or cat_text in cat.value:
                    analysis.category = cat
                    break

        # Extract severity
        severity_match = re.search(
            r'Severity:\s*(\w+)',
            response, re.IGNORECASE
        )
        if severity_match:
            sev_text = severity_match.group(1).lower()
            if "recover" in sev_text:
                analysis.severity = ErrorSeverity.RECOVERABLE
            elif "rollback" in sev_text:
                analysis.severity = ErrorSeverity.REQUIRES_ROLLBACK
            elif "fatal" in sev_text:
                analysis.severity = ErrorSeverity.FATAL

        # Extract root cause
        root_match = re.search(
            r'\*\*Root Cause(?:\s+Analysis)?\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if root_match:
            analysis.root_cause = root_match.group(1).strip()

        # Extract primary fix
        primary_match = re.search(
            r'\*\*Primary Fix\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if primary_match:
            analysis.primary_fix = primary_match.group(1).strip()

        # Extract alternative fixes
        alt_match = re.search(
            r'\*\*Alternative Fixes\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if alt_match:
            text = alt_match.group(1)
            analysis.alternative_fixes = [
                line.strip().lstrip('- ').lstrip('0123456789. ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract prevention advice
        prevent_match = re.search(
            r'\*\*Prevention\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if prevent_match:
            analysis.prevention_advice = prevent_match.group(1).strip()

        # Extract code fix
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            analysis.fix_code = code_match.group(1)

        # Extract suggested delta_R
        delta_r_match = re.search(
            r'(?:Safe delta_R|suggested delta_R)[:\s]+([+-]?[0-9.]+)',
            response, re.IGNORECASE
        )
        if delta_r_match:
            try:
                analysis.suggested_delta_R = float(delta_r_match.group(1))
            except ValueError:
                pass

        return analysis

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
        Diagnose a general error.

        Args:
            context: Agent context
            error_type: Type of error
            error_message: Error message
            mutation: Description of the mutation that caused the error
            delta_R: Delta R value used
            logs: Recent logs

        Returns:
            DebugAnalysis with diagnosis and fixes
        """
        return await self.execute(
            context,
            task="diagnose_error",
            error_type=error_type,
            error_message=error_message,
            mutation=mutation,
            delta_R=delta_R,
            logs=logs,
        )

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
        return await self.execute(
            context,
            task="diagnose_mesh_error",
            error_message=error_message,
            min_jacobian=min_jacobian,
            problem_elements=problem_elements,
            delta_R=delta_R,
            successful_delta_r=successful_delta_r or "No previous successful runs",
        )

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
        return await self.execute(
            context,
            task="diagnose_convergence",
            error_message=error_message,
            convergence_history=convergence_history,
            solver_settings=solver_settings,
            load_info=load_info,
        )

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

        Args:
            context: Agent context
            attempt_number: Current attempt number
            max_attempts: Maximum allowed attempts
            error_history: History of errors
            previous_fixes: Previously attempted fixes

        Returns:
            RetryStrategy
        """
        analysis = await self.execute(
            context,
            task="suggest_retry_strategy",
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            error_history=error_history,
            previous_fixes=previous_fixes,
        )

        # Convert analysis to retry strategy
        strategy = RetryStrategy()

        if analysis.severity == ErrorSeverity.FATAL:
            strategy.should_retry = False
            strategy.abandon_reason = analysis.root_cause
        else:
            strategy.should_retry = attempt_number < max_attempts
            strategy.strategy_description = analysis.primary_fix

            if analysis.suggested_delta_R is not None:
                strategy.adjustments["delta_R"] = analysis.suggested_delta_R

            strategy.adjustments.update(analysis.config_changes)

        return strategy

    def quick_classify_error(self, error_message: str) -> ErrorCategory:
        """
        Quickly classify an error without LLM.

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
