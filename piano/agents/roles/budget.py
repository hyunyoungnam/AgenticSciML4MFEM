"""
Budget Agent.

Runs at the end of each active learning iteration to decide whether to:
1. Continue — collect more FEM samples
2. Switch to HPO — run more hyperparameter optimization rounds instead
3. Stop — surrogate has converged (diminishing returns)

Replaces the fixed `max_samples` heuristic in AdaptiveOrchestrator.
The agent reasons about uncertainty drop rate, error trajectory, and
cost ratio (FEM simulation cost vs HPO round cost).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from piano.agents.base import AgentContext, AgentRole, BaseAgent

logger = logging.getLogger(__name__)


_BUDGET_SYSTEM = """You are a resource allocation strategist for an active learning loop that trains
neural operator surrogates for fracture mechanics FEM simulations.

Each FEM simulation costs ~5-30 minutes of compute. Each HPO round costs ~2 minutes.
The goal is to reach a target relative L2 error < 5% with minimum total FEM simulations.

Given the current iteration metrics, decide:
- **continue_fem**: Collect more FEM samples (uncertainty is still high and reducible by more data)
- **switch_hpo**: Stop FEM collection, run more HPO rounds instead (model architecture/physics is the bottleneck)
- **converged**: Stop everything — surrogate meets the quality target
- **increase_budget**: Temporarily increase samples_per_iteration by 50% (error spike or large uncertainty region)

Decision rules (use as guidelines, not rigid rules):
- If uncertainty dropped < 5% last iteration AND error is still high: switch_hpo
- If test error < convergence_threshold: converged
- If uncertainty is high AND error is improving: continue_fem
- If patience_count >= patience: converged (no improvement trend)

Output format (exactly):
DECISION: <continue_fem|switch_hpo|converged|increase_budget>
SAMPLES_NEXT: <integer, only if decision is continue_fem or increase_budget>
REASONING: <2-3 sentences with specific numbers>"""


_BUDGET_PROMPT = """Active learning iteration {iteration} summary:

Error trajectory (last {n_history} iterations):
{error_history}

Uncertainty trajectory:
{uncertainty_history}

Current state:
- Test error: {test_error:.6f}
- Convergence target: {convergence_threshold:.6f}
- Mean uncertainty: {mean_uncertainty:.6f}
- Uncertainty drop last iteration: {uncertainty_drop:.3f} ({uncertainty_drop_pct:.1f}%)
- Total samples collected: {n_samples}
- Max budget: {max_samples}
- Remaining budget: {remaining_budget}
- Patience: {patience_count}/{patience}
- HPO rounds run so far: {n_hpo_rounds}

Decide whether to continue FEM collection or switch strategy."""


@dataclass
class BudgetDecision:
    """Output of the Budget Agent."""
    decision: str = "continue_fem"
    samples_next: int = 10
    reasoning: str = ""
    raw_response: str = ""

    def should_stop(self) -> bool:
        return self.decision == "converged"

    def should_switch_hpo(self) -> bool:
        return self.decision == "switch_hpo"

    def should_increase_budget(self) -> bool:
        return self.decision == "increase_budget"


class BudgetAgent(BaseAgent[BudgetDecision]):
    """
    Budget allocation agent for active learning.

    Decides when to continue FEM sampling, when to switch to HPO-only mode,
    and when the surrogate has converged. Replaces the fixed `max_samples`
    heuristic with data-driven reasoning.

    The agent tracks uncertainty drop rate: if uncertainty dropped < 5% last
    iteration, more data is unlikely to help and HPO should be tried instead.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        convergence_threshold: float = 0.05,
        max_samples: int = 200,
        patience: int = 3,
        base_samples_per_iter: int = 10,
        **kwargs,
    ):
        super().__init__(role=AgentRole.ANALYST, model=model, temperature=0.2, **kwargs)
        self._convergence_threshold = convergence_threshold
        self._max_samples = max_samples
        self._patience = patience
        self._base_samples = base_samples_per_iter
        self._patience_count: int = 0
        self._prev_uncertainty: Optional[float] = None
        self._error_history: List[float] = []
        self._uncertainty_history: List[float] = []

    def get_system_prompt(self) -> str:
        return _BUDGET_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        s = kwargs["state"]
        n = min(len(self._error_history), 5)
        error_hist = "\n".join(
            f"  iter {i+1}: {e:.6f}"
            for i, e in enumerate(self._error_history[-n:])
        ) or "  (no history)"
        unc_hist = "\n".join(
            f"  iter {i+1}: {u:.6f}"
            for i, u in enumerate(self._uncertainty_history[-n:])
        ) or "  (no history)"

        unc_drop = 0.0
        unc_drop_pct = 0.0
        if self._prev_uncertainty and self._prev_uncertainty > 0:
            unc_drop = self._prev_uncertainty - s["mean_uncertainty"]
            unc_drop_pct = unc_drop / self._prev_uncertainty * 100

        return _BUDGET_PROMPT.format(
            iteration=s["iteration"],
            n_history=n,
            error_history=error_hist,
            uncertainty_history=unc_hist,
            test_error=s["test_error"],
            convergence_threshold=self._convergence_threshold,
            mean_uncertainty=s["mean_uncertainty"],
            uncertainty_drop=unc_drop,
            uncertainty_drop_pct=unc_drop_pct,
            n_samples=s["n_samples"],
            max_samples=self._max_samples,
            remaining_budget=max(0, self._max_samples - s["n_samples"]),
            patience_count=self._patience_count,
            patience=self._patience,
            n_hpo_rounds=s.get("n_hpo_rounds", 0),
        )

    def parse_response(self, response: str) -> BudgetDecision:
        result = BudgetDecision(raw_response=response)

        m = re.search(r"DECISION:\s*(\S+)", response, re.IGNORECASE)
        if m:
            result.decision = m.group(1).strip(".,").lower()

        m = re.search(r"SAMPLES_NEXT:\s*(\d+)", response, re.IGNORECASE)
        if m:
            result.samples_next = int(m.group(1))
        else:
            result.samples_next = self._base_samples

        if result.decision == "increase_budget":
            result.samples_next = max(result.samples_next, int(self._base_samples * 1.5))

        m = re.search(r"REASONING:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if m:
            result.reasoning = m.group(1).strip()

        return result

    async def decide(
        self,
        context: AgentContext,
        iteration: int,
        test_error: float,
        mean_uncertainty: float,
        n_samples: int,
        n_hpo_rounds: int = 0,
    ) -> BudgetDecision:
        """
        Decide whether to continue FEM collection or change strategy.

        Args:
            context: Agent context
            iteration: Current iteration index
            test_error: Current surrogate test L2 error
            mean_uncertainty: Mean ensemble uncertainty
            n_samples: Total samples collected so far
            n_hpo_rounds: HPO rounds run this session

        Returns:
            BudgetDecision
        """
        # Update history
        self._error_history.append(test_error)
        self._uncertainty_history.append(mean_uncertainty)

        # Update patience
        if len(self._error_history) > 1:
            if self._error_history[-1] >= self._error_history[-2] * 0.99:
                self._patience_count += 1
            else:
                self._patience_count = 0

        # Fast-path heuristics (skip LLM if trivially obvious)
        if test_error <= self._convergence_threshold:
            logger.info("BudgetAgent: converged (error below threshold)")
            decision = BudgetDecision(
                decision="converged",
                reasoning=f"Test error {test_error:.6f} < threshold {self._convergence_threshold:.6f}",
            )
            self._prev_uncertainty = mean_uncertainty
            return decision

        if n_samples >= self._max_samples:
            logger.info("BudgetAgent: budget exhausted")
            decision = BudgetDecision(
                decision="converged",
                reasoning=f"Budget exhausted ({n_samples}/{self._max_samples} samples)",
            )
            self._prev_uncertainty = mean_uncertainty
            return decision

        state = {
            "iteration": iteration,
            "test_error": test_error,
            "mean_uncertainty": mean_uncertainty,
            "n_samples": n_samples,
            "n_hpo_rounds": n_hpo_rounds,
        }
        decision = await self.execute(context, state=state)

        logger.info(
            f"BudgetAgent: {decision.decision} (samples_next={decision.samples_next}), "
            f"reasoning={decision.reasoning[:100]}"
        )
        self._prev_uncertainty = mean_uncertainty
        return decision

    def decide_sync(
        self,
        iteration: int,
        test_error: float,
        mean_uncertainty: float,
        n_samples: int,
        n_hpo_rounds: int = 0,
    ) -> BudgetDecision:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.decide(AgentContext(), iteration, test_error, mean_uncertainty,
                            n_samples, n_hpo_rounds)
            )
        finally:
            loop.close()

    def reset(self) -> None:
        """Reset state for a new training session."""
        self._patience_count = 0
        self._prev_uncertainty = None
        self._error_history.clear()
        self._uncertainty_history.clear()
