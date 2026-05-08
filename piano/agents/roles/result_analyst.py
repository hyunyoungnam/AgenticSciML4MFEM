"""
Result Analyst Agent.

Reads training curves and identifies loss patterns — purely observational.
No proposals, no hyperparameter suggestions. Participates in Round 1 (OBSERVATION)
of the multi-round debate and provides objective curve-reading for all agents.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from piano.agents.base import BaseAgent, AgentContext, AgentRole

if TYPE_CHECKING:
    from piano.agents.roles.hyperparameter_critic import TrainingHistory


_ANALYST_SYSTEM = """You are a training curve analyst for neural operators in physics-informed machine learning.

Your ONLY job is to describe what you observe in training data — do NOT suggest hyperparameter changes,
do NOT recommend actions, do NOT say "you should". Pure observation only.

## Loss Curve Patterns to Identify

- **converging**: both train and test loss steadily decreasing over time
- **overfitting**: train loss << test loss and the gap is widening
- **underfitting**: both losses high, model hasn't learned the mapping
- **oscillating**: loss bouncing up and down without a clear trend (unstable dynamics)
- **plateaued**: loss stopped improving (< 1% change across recent epochs)
- **diverging**: loss increasing — training instability or gradient explosion

## Per-Term PINO Notes

For each physics term (elasticity, crack), report: is it decreasing, plateaued, or high relative to data loss?
A term > 10% of data loss is dominating the signal. A term < 0.5% has negligible effect.

## Ensemble Notes

ensemble_std / mean_test_loss > 0.5 = high variance between members (inconsistent predictions).
ensemble_std / mean_test_loss < 0.1 = stable ensemble (members agree).

Use specific numbers from the data. Do not use vague language."""


_ANALYST_PROMPT = """{history_summary}

Describe ONLY what you observe. Use exact numbers. No suggestions.

OVERALL_PATTERN: <converging|plateaued|overfitting|underfitting|oscillating|diverging>
SEVERITY: <low|medium|high|critical>
OBSERVATION: <2-4 sentences with specific numbers describing loss trajectory and train/test gap>
PINO_STATUS: <per-term description with numbers, or "no PINO terms active">
ENSEMBLE_STATUS: <ensemble agreement description with ratio, or "no ensemble data">"""


@dataclass
class AnalystObservation:
    """Structured observation from the Result Analyst."""
    pattern: str = "unknown"
    severity: str = "medium"
    observation: str = ""
    pino_status: str = ""
    ensemble_status: str = ""
    raw_response: str = ""

    def to_debate_message(self) -> str:
        return (
            f"[ANALYST — Round 1 Observation]\n"
            f"Pattern: {self.pattern} (severity: {self.severity})\n"
            f"{self.observation}\n"
            f"PINO: {self.pino_status}\n"
            f"Ensemble: {self.ensemble_status}"
        )


class ResultAnalystAgent(BaseAgent[AnalystObservation]):
    """
    Result Analyst — observational only, no proposals.

    Reads loss curves and identifies patterns for Round 1 of the agent debate.
    Provides objective signal to downstream proposal agents (Architect, Physicist).
    """

    def __init__(self, model: str = "gpt-4-turbo", **kwargs):
        super().__init__(role=AgentRole.ANALYST, model=model, temperature=0.2, **kwargs)

    def get_system_prompt(self) -> str:
        return _ANALYST_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        history = kwargs["history"]
        return _ANALYST_PROMPT.format(history_summary=history.to_summary())

    def parse_response(self, response: str) -> AnalystObservation:
        obs = AnalystObservation(raw_response=response)

        m = re.search(r"OVERALL_PATTERN:\s*(\S+)", response)
        if m:
            obs.pattern = m.group(1).strip(".,").lower()

        m = re.search(r"SEVERITY:\s*(\S+)", response)
        if m:
            obs.severity = m.group(1).strip(".,").lower()

        m = re.search(r"OBSERVATION:\s*(.+?)(?=PINO_STATUS:|ENSEMBLE_STATUS:|$)", response, re.DOTALL)
        if m:
            obs.observation = m.group(1).strip()

        m = re.search(r"PINO_STATUS:\s*(.+?)(?=ENSEMBLE_STATUS:|$)", response, re.DOTALL)
        if m:
            obs.pino_status = m.group(1).strip()

        m = re.search(r"ENSEMBLE_STATUS:\s*(.+?)$", response, re.DOTALL)
        if m:
            obs.ensemble_status = m.group(1).strip()

        return obs

    async def observe(self, context: AgentContext, history: "TrainingHistory") -> AnalystObservation:
        return await self.execute(context, history=history)

    def observe_sync(self, history: "TrainingHistory") -> AnalystObservation:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.observe(AgentContext(), history))
        finally:
            loop.close()
