"""
Physicist Agent implementation.

The Physicist Agent specializes in physics-informed loss configuration,
understanding PDE constraints, singularities, and how to balance data-driven
learning with physics enforcement.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from piano.agents.base import BaseAgent, AgentContext, AgentRole
from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue


class PhysicsIssue(Enum):
    """Types of physics-related issues the physicist can diagnose."""
    WEAK_PHYSICS_ENFORCEMENT = auto()      # Physics loss too low, not constraining
    OVERLY_STRONG_PHYSICS = auto()         # Physics dominates, hurts data fit
    SINGULARITY_NOT_CAPTURED = auto()      # 1/sqrt(r) or similar not learned
    BOUNDARY_VIOLATION = auto()            # BC not satisfied
    EQUILIBRIUM_IMBALANCE = auto()         # Force balance not achieved
    ENERGY_INCONSISTENCY = auto()          # Strain energy errors
    PHYSICS_DATA_CONFLICT = auto()         # Physics and data losses fighting
    NONE = auto()


@dataclass
class PhysicsProposal:
    """A physics configuration proposal from the Physicist."""
    changes: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    physics_diagnosis: str = ""
    expected_impact: str = ""
    confidence: str = "medium"  # "low", "medium", "high"
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "changes": self.changes,
            "reasoning": self.reasoning,
            "physics_diagnosis": self.physics_diagnosis,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
        }


PHYSICIST_SYSTEM = """You are an expert computational physicist specializing in physics-informed neural networks for solid mechanics.

Your role is to tune physics loss configurations based on training diagnostics, ensuring the neural operator respects physical laws while learning from data.

## Physics-Informed Loss Components

The PINO (Physics-Informed Neural Operator) loss has two main terms:

### 1. Equilibrium Residual (pino_eq_weight)
- Enforces: div(sigma) = 0 (force balance at mesh nodes)
- Label-free: No ground truth needed
- Computed: Nodal force residual via B-matrix assembly
- Effect: Ensures predicted displacement field satisfies equilibrium

### 2. Energy-Norm Error (pino_weight)
- Enforces: Strain energy consistency with ground truth
- With labels: Requires ground truth displacement
- Computed: W(u_pred - u_true) / Volume
- Effect: Physics-weighted H1 seminorm, penalizes strain errors

### Material Parameters
- `pino_E`: Young's modulus for constitutive law (dimensionless, typically 1.0)
- `pino_nu`: Poisson's ratio (typically 0.3 for metals)

## Physics Considerations for Fracture Mechanics

### Crack Tip Singularity (1/sqrt(r))
- Near crack tip: stress ~ K_I / sqrt(2*pi*r)
- Neural networks struggle with singularities
- Higher physics weight can help enforce correct asymptotic behavior
- But too high can destabilize training

### Williams Expansion
- Mode I: u_x, u_y ~ sqrt(r) * f(theta)
- The sqrt(r) behavior is hard to learn from data alone
- Physics loss helps by enforcing equilibrium even in singular region

## Issue-to-Change Mapping

| Physics Issue | Recommended Changes |
|---------------|---------------------|
| WEAK_PHYSICS_ENFORCEMENT | Increase pino_weight and/or pino_eq_weight |
| OVERLY_STRONG_PHYSICS | Reduce pino_weight, let data loss dominate initially |
| SINGULARITY_NOT_CAPTURED | Increase pino_eq_weight, ensure equilibrium near tip |
| BOUNDARY_VIOLATION | Add or increase boundary loss (future: pino_bc_weight) |
| EQUILIBRIUM_IMBALANCE | Increase pino_eq_weight relative to pino_weight |
| ENERGY_INCONSISTENCY | Increase pino_weight, check material parameters |
| PHYSICS_DATA_CONFLICT | Reduce physics weights, train data-only first |

## Typical Weight Ranges

- `pino_weight`: 0.01 to 1.0 (energy norm)
- `pino_eq_weight`: 0.01 to 1.0 (equilibrium residual)
- Start low (0.01-0.1) and increase if physics is violated
- Ratio matters: eq_weight / pino_weight affects balance

Output your proposals in structured format."""


PHYSICIST_PROMPT = """## Current Physics Configuration
- pino_weight: {pino_weight}
- pino_eq_weight: {pino_eq_weight}
- pino_E: {pino_E}
- pino_nu: {pino_nu}

## Training Diagnostics
**Primary Issue**: {primary_issue}
**Severity**: {severity}
**Diagnosis**: {diagnosis}

## Loss History (sampled)
- Train losses: {train_losses}
- Test losses: {test_losses}
- PINO losses: {pino_losses}

## Problem Context
- Problem type: {problem_type}
- Dataset size: {dataset_size} samples
- Has singularity: {has_singularity}

## Previous Physics Configs Tried
{previous_configs}

## Your Task

Analyze the training from a PHYSICS perspective and propose changes to the physics loss configuration.

Consider:
1. Is the physics loss helping or hurting learning?
2. Is there evidence of physics violation (high PINO loss)?
3. For crack problems: is the singularity being captured?
4. Are the equilibrium and energy terms balanced appropriately?

Format your response as:
```
PHYSICS_DIAGNOSIS: [Your physics-specific diagnosis]

CHANGES:
- pino_weight: [value] (reason)
- pino_eq_weight: [value] (reason)
- pino_E: [value] (reason, usually keep at 1.0)
- pino_nu: [value] (reason, usually keep at 0.3)

REASONING: [Why these physics changes will help]
EXPECTED_IMPACT: [What improvement you expect]
CONFIDENCE: [low|medium|high]
```

Only include parameters you want to change.
"""


class PhysicistAgent(BaseAgent[PhysicsProposal]):
    """
    Physicist Agent for physics-informed loss configuration.

    Responsibilities:
    1. Analyze training from physics perspective
    2. Diagnose physics-specific issues (singularities, equilibrium, etc.)
    3. Propose physics loss weight adjustments
    4. Balance data-driven and physics-informed learning
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.PHYSICIST,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return PHYSICIST_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        current_config: Dict[str, Any] = kwargs.get("current_config", {})
        critique: CritiqueResult = kwargs.get("critique", CritiqueResult())
        training_history = kwargs.get("training_history", None)
        dataset_size: int = kwargs.get("dataset_size", 0)
        problem_type: str = kwargs.get("problem_type", "crack")
        has_singularity: bool = kwargs.get("has_singularity", True)
        previous_configs: List[Dict] = kwargs.get("previous_configs", [])

        # Extract loss history
        train_losses = "[]"
        test_losses = "[]"
        pino_losses = "[]"

        if training_history:
            train_losses = self._sample_losses(
                getattr(training_history, 'train_losses', [])
            )
            test_losses = self._sample_losses(
                getattr(training_history, 'test_losses', [])
            )
            pino_losses = self._sample_losses(
                getattr(training_history, 'pino_losses', [])
            )

        # Format previous configs
        prev_str = "None"
        if previous_configs:
            prev_lines = []
            for i, cfg in enumerate(previous_configs):
                physics_cfg = {
                    k: v for k, v in cfg.get("config", {}).items()
                    if k.startswith("pino")
                }
                result = cfg.get("result", "unknown")
                prev_lines.append(f"  Attempt {i+1}: {physics_cfg} -> {result}")
            prev_str = "\n".join(prev_lines) if prev_lines else "None"

        return PHYSICIST_PROMPT.format(
            pino_weight=current_config.get("pino_weight", 0.1),
            pino_eq_weight=current_config.get("pino_eq_weight", 0.1),
            pino_E=current_config.get("pino_E", 1.0),
            pino_nu=current_config.get("pino_nu", 0.3),
            primary_issue=critique.primary_issue.name,
            severity=critique.severity,
            diagnosis=critique.diagnosis,
            train_losses=train_losses,
            test_losses=test_losses,
            pino_losses=pino_losses,
            problem_type=problem_type,
            dataset_size=dataset_size,
            has_singularity=has_singularity,
            previous_configs=prev_str,
        )

    def _sample_losses(self, losses: List[float], max_samples: int = 10) -> str:
        """Sample losses for display."""
        if not losses:
            return "[]"
        if len(losses) <= max_samples:
            return str([round(l, 6) for l in losses])

        # Sample at intervals
        step = len(losses) // max_samples
        sampled = [round(losses[i * step], 6) for i in range(max_samples)]
        return str(sampled)

    def parse_response(self, response: str) -> PhysicsProposal:
        """Parse the LLM response into a PhysicsProposal."""
        proposal = PhysicsProposal(raw_response=response)

        # Extract physics diagnosis
        diag_match = re.search(
            r'PHYSICS_DIAGNOSIS:\s*(.*?)(?=CHANGES:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if diag_match:
            proposal.physics_diagnosis = diag_match.group(1).strip()

        # Extract changes
        changes_match = re.search(
            r'CHANGES:\s*(.*?)(?=REASONING:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if changes_match:
            changes_text = changes_match.group(1)
            proposal.changes = self._parse_changes(changes_text)

        # Extract reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.*?)(?=EXPECTED_IMPACT:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            proposal.reasoning = reasoning_match.group(1).strip()

        # Extract expected impact
        impact_match = re.search(
            r'EXPECTED_IMPACT:\s*(.*?)(?=CONFIDENCE:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if impact_match:
            proposal.expected_impact = impact_match.group(1).strip()

        # Extract confidence
        confidence_match = re.search(
            r'CONFIDENCE:\s*(low|medium|high)',
            response, re.IGNORECASE
        )
        if confidence_match:
            proposal.confidence = confidence_match.group(1).lower()

        return proposal

    def _parse_changes(self, text: str) -> Dict[str, Any]:
        """Parse physics parameter changes from text."""
        changes = {}

        # Physics parameters (all floats)
        params = ['pino_weight', 'pino_eq_weight', 'pino_E', 'pino_nu']

        for param in params:
            match = re.search(rf'{param}:\s*([0-9.e-]+)', text, re.IGNORECASE)
            if match:
                try:
                    changes[param] = float(match.group(1))
                except ValueError:
                    pass

        return changes

    def detect_physics_issues(
        self,
        training_history,
        current_config: Dict[str, Any],
    ) -> List[PhysicsIssue]:
        """
        Heuristic detection of physics issues (no LLM call).

        Used as a pre-filter to decide if physicist consultation is needed.
        """
        issues = []

        pino_losses = getattr(training_history, 'pino_losses', [])
        train_losses = getattr(training_history, 'train_losses', [])
        test_losses = getattr(training_history, 'test_losses', [])

        if not pino_losses or len(pino_losses) < 5:
            return issues

        # Check if PINO loss is not decreasing (physics not being learned)
        early_pino = sum(pino_losses[:5]) / 5
        late_pino = sum(pino_losses[-5:]) / 5

        if late_pino > 0.9 * early_pino:
            # PINO loss barely decreased
            issues.append(PhysicsIssue.WEAK_PHYSICS_ENFORCEMENT)

        # Check for physics-data conflict
        if train_losses and test_losses:
            train_improving = train_losses[-1] < train_losses[0] * 0.5
            pino_worsening = late_pino > early_pino * 1.2

            if train_improving and pino_worsening:
                issues.append(PhysicsIssue.PHYSICS_DATA_CONFLICT)

        # Check for overly strong physics (train loss stuck, pino very low)
        pino_weight = current_config.get("pino_weight", 0.1)
        if pino_weight > 0.5 and train_losses:
            if train_losses[-1] > 0.8 * train_losses[0]:
                issues.append(PhysicsIssue.OVERLY_STRONG_PHYSICS)

        return issues

    def should_consult(
        self,
        training_history,
        current_config: Dict[str, Any],
    ) -> bool:
        """
        Determine if the physicist should be consulted.

        Returns True if there are physics-specific issues to address.
        """
        issues = self.detect_physics_issues(training_history, current_config)
        return len(issues) > 0

    async def propose_physics_config(
        self,
        context: AgentContext,
        current_config: Dict[str, Any],
        critique: CritiqueResult,
        training_history=None,
        dataset_size: int = 0,
        problem_type: str = "crack",
        has_singularity: bool = True,
        previous_configs: Optional[List[Dict]] = None,
    ) -> PhysicsProposal:
        """
        Propose physics configuration changes based on training analysis.

        Args:
            context: Agent context
            current_config: Current model config dict
            critique: Critic's analysis results
            training_history: Training history with losses
            dataset_size: Number of training samples
            problem_type: Type of physics problem
            has_singularity: Whether problem has stress singularity
            previous_configs: History of previous configurations

        Returns:
            PhysicsProposal with recommended physics parameter changes
        """
        return await self.execute(
            context,
            current_config=current_config,
            critique=critique,
            training_history=training_history,
            dataset_size=dataset_size,
            problem_type=problem_type,
            has_singularity=has_singularity,
            previous_configs=previous_configs or [],
        )
