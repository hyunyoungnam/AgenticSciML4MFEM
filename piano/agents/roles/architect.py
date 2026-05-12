"""
Architect Agent implementation.

The Architect proposes new model configurations based on critic feedback,
considering architecture choices, optimizer settings, and training parameters.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from piano.agents.base import BaseAgent, AgentContext, AgentRole
from piano.agents.roles.hyperparameter_critic import CritiqueResult
from piano.surrogate.base import TransolverConfig


@dataclass
class ArchitectureProposal:
    """A configuration proposal from the Architect."""
    config: TransolverConfig = field(default_factory=TransolverConfig)
    changes: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    expected_impact: str = ""
    trade_offs: str = ""
    confidence: str = "medium"  # "low", "medium", "high"
    code_change_description: str = "none"
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "changes": self.changes,
            "reasoning": self.reasoning,
            "expected_impact": self.expected_impact,
            "trade_offs": self.trade_offs,
            "confidence": self.confidence,
            "code_change_description": self.code_change_description,
        }


ARCHITECT_SYSTEM = """You are an expert neural network architect specializing in neural operators for physics-informed machine learning.

Your role is to propose concrete model configurations based on training diagnostics.

## Model: Transolver

The Transolver is a Physics-Attention transformer-based neural operator that learns mappings from parameters to physical fields on unstructured meshes. Key hyperparameters:

### Transolver Hyperparameters
- `d_model`: Hidden dimension (32, 64, 128, 256). Higher = more capacity.
- `n_layers`: Transformer layers (2, 4, 6). More = deeper features, risk of overfitting.
- `n_heads`: Attention heads (2, 4, 8). Must divide d_model evenly.
- `slice_num`: Physics slices (4, 8, 16, 32). More = finer spatial resolution.
- `mlp_ratio`: FFN expansion (2.0, 4.0).
- `dropout`: Regularization (0.0, 0.1, 0.2, 0.3).
- `activation`: Activation function (gelu, relu, silu).

### Optimizer
- `optimizer_type`: adamw (default, good regularization), adam, sgd (needs tuning).
- `learning_rate`: (1e-4 to 1e-2). Lower = stable but slow, higher = fast but risky.
- `scheduler_type`: plateau (adaptive), cosine (smooth decay), none.

### Training
- `batch_size`: (8, 16, 32, 64). Larger = smoother gradients, more memory.
- `epochs`: Max epochs.
- `patience`: Early stopping patience.

## Design Principles

1. **Start Conservative**: Begin with moderate settings, adjust based on feedback.
2. **Address Root Cause**: Match changes to diagnosed issues.
3. **One Major Change**: Prefer changing one major aspect at a time for interpretability.
4. **Consider Data Size**: Small datasets need simpler models and more regularization.
5. **Flag Code Issues**: If the problem cannot be fixed with hyperparameters (e.g., wrong
   feature encoding, unnormalized loss term, missing physics term), set
   CODE_CHANGE_DESCRIPTION to a precise description of the required source-code change.
   An engineer agent with Claude Code CLI will implement it before retraining.

## Issue-to-Change Mapping

| Issue | Primary Changes |
|-------|-----------------|
| OVERFITTING | Increase dropout, reduce d_model/n_layers |
| UNDERFITTING | Increase d_model/n_layers, reduce dropout, increase lr |
| SLOW_CONVERGENCE | Increase lr, use cosine scheduler, reduce model size |
| UNSTABLE_TRAINING | Reduce lr, use plateau scheduler, increase batch_size |
| LOSS_PLATEAU | Change optimizer, adjust scheduler, modify architecture |
| GRADIENT_EXPLOSION | Reduce lr significantly, add gradient clipping via smaller batch |
| LEARNING_RATE_TOO_HIGH | Reduce lr by 2-5x |
| LEARNING_RATE_TOO_LOW | Increase lr by 2-5x |
| INSUFFICIENT_CAPACITY | Increase d_model, n_layers, or slice_num |
| EXCESSIVE_CAPACITY | Reduce d_model, n_layers, add dropout |

Output your proposals in structured format."""


ARCHITECT_PROMPT = """## Current Configuration
{current_config}

## Critic's Diagnosis
**Primary Issue**: {primary_issue}
**Severity**: {severity}
**Diagnosis**: {diagnosis}

**Recommendations from Critic**:
{recommendations}

## Training Context
- Dataset size: {dataset_size} samples
- Previous attempts: {n_attempts}
{previous_configs}

## Your Task

Propose a new TransolverConfig that addresses the critic's diagnosis.

Consider:
1. The severity of the issue
2. The dataset size (smaller datasets need simpler models)
3. Previous failed attempts (don't repeat them)
4. Trade-offs between capacity and generalization

Format your response as:
```
REASONING: [Why you're making these changes]

CHANGES:
- d_model: [value] (reason)
- n_layers: [value] (reason)
- n_heads: [value] (reason)
- slice_num: [value] (reason)
- dropout: [value] (reason)
- learning_rate: [value] (reason)
- optimizer_type: [adamw|adam|sgd] (reason)
- scheduler_type: [plateau|cosine|none] (reason)
- activation: [gelu|relu|silu] (reason)
- batch_size: [value] (reason)

EXPECTED_IMPACT: [What improvement you expect]
TRADE_OFFS: [What trade-offs this configuration makes]
CONFIDENCE: [low|medium|high]
CODE_CHANGE_DESCRIPTION: none | [If hyperparameter changes alone cannot fix the issue,
describe exactly what source-code change is needed — e.g. "The physics attention in
piano/surrogate/transolver.py should include log(r) relative to crack tip rather than raw
distance" or "The elasticity PINO loss in piano/physics/pino_loss.py should normalize
by domain area". Set to 'none' if config changes are sufficient.]
```
"""


class ArchitectAgent(BaseAgent[ArchitectureProposal]):
    """
    Architect Agent for proposing model configurations.

    Responsibilities:
    1. Interpret critic feedback
    2. Propose concrete TransolverConfig changes
    3. Explain trade-offs and expected impact
    4. Learn from previous failed attempts
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.ARCHITECT,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return ARCHITECT_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        current_config: TransolverConfig = kwargs.get(
            "current_config", TransolverConfig()
        )
        critique: CritiqueResult = kwargs.get("critique", CritiqueResult())
        dataset_size: int = kwargs.get("dataset_size", 0)
        previous_configs: List[Dict] = kwargs.get("previous_configs", [])
        debate_context: str = kwargs.get("debate_context", "")

        # Format current config
        config_dict = current_config.to_dict()
        config_str = "\n".join([f"  {k}: {v}" for k, v in config_dict.items()])

        # Format recommendations
        rec_str = "\n".join([f"- {r}" for r in critique.recommendations]) or "None"

        # Format previous configs
        prev_str = ""
        if previous_configs:
            prev_lines = ["Previous configurations tried:"]
            for i, cfg in enumerate(previous_configs):
                changes = cfg.get("changes", {})
                result = cfg.get("result", "unknown")
                prev_lines.append(f"  Attempt {i+1}: {changes} -> {result}")
            prev_str = "\n".join(prev_lines)

        prompt = ARCHITECT_PROMPT.format(
            current_config=config_str,
            primary_issue=critique.primary_issue.name,
            severity=critique.severity,
            diagnosis=critique.diagnosis,
            recommendations=rec_str,
            dataset_size=dataset_size,
            n_attempts=len(previous_configs),
            previous_configs=prev_str,
        )
        if debate_context:
            debate_section = (
                "\n## Debate Context (Agent Observations — Rounds 1-2)\n"
                + debate_context
                + "\nIMPORTANT: Your proposal MUST address the specific issues identified above.\n"
            )
            prompt = debate_section + "\n" + prompt
        return prompt

    def parse_response(self, response: str) -> ArchitectureProposal:
        """Parse the LLM response into an ArchitectureProposal."""
        proposal = ArchitectureProposal(raw_response=response)

        # Extract reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.*?)(?=CHANGES:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            proposal.reasoning = reasoning_match.group(1).strip()

        # Extract changes
        changes_match = re.search(
            r'CHANGES:\s*(.*?)(?=EXPECTED_IMPACT:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if changes_match:
            changes_text = changes_match.group(1)
            proposal.changes = self._parse_changes(changes_text)

        # Extract expected impact
        impact_match = re.search(
            r'EXPECTED_IMPACT:\s*(.*?)(?=TRADE_OFFS:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if impact_match:
            proposal.expected_impact = impact_match.group(1).strip()

        # Extract trade-offs
        tradeoffs_match = re.search(
            r'TRADE_OFFS:\s*(.*?)(?=CONFIDENCE:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if tradeoffs_match:
            proposal.trade_offs = tradeoffs_match.group(1).strip()

        # Extract confidence
        confidence_match = re.search(
            r'CONFIDENCE:\s*(low|medium|high)',
            response, re.IGNORECASE
        )
        if confidence_match:
            proposal.confidence = confidence_match.group(1).lower()

        # Extract code change description
        code_match = re.search(
            r'CODE_CHANGE_DESCRIPTION:\s*(.*?)(?=\n```|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if code_match:
            raw = code_match.group(1).strip()
            proposal.code_change_description = "none" if raw.lower() == "none" else raw

        return proposal

    def _parse_changes(self, text: str) -> Dict[str, Any]:
        """Parse change specifications from text."""
        changes = {}

        # Patterns for different parameter types
        patterns = {
            'int': ['d_model', 'n_layers', 'n_heads', 'slice_num', 'batch_size', 'epochs', 'patience'],
            'float': ['dropout', 'learning_rate', 'mlp_ratio'],
            'str': ['optimizer_type', 'scheduler_type', 'activation'],
        }

        for param in patterns['int']:
            match = re.search(rf'{param}:\s*(\d+)', text, re.IGNORECASE)
            if match:
                changes[param] = int(match.group(1))

        for param in patterns['float']:
            match = re.search(rf'{param}:\s*([0-9.e-]+)', text, re.IGNORECASE)
            if match:
                try:
                    changes[param] = float(match.group(1))
                except ValueError:
                    pass

        for param in patterns['str']:
            match = re.search(rf'{param}:\s*(\w+)', text, re.IGNORECASE)
            if match:
                changes[param] = match.group(1).lower()

        return changes

    def apply_changes(
        self,
        base_config: Any,
        changes: Dict[str, Any],
    ) -> TransolverConfig:
        """Apply proposed changes to the current TransolverConfig."""
        config_dict = base_config.to_dict() if hasattr(base_config, "to_dict") else {}
        _arch_keys = {"slice_num", "n_heads", "d_model", "n_layers", "mlp_ratio", "dropout",
                      "learning_rate", "batch_size", "epochs", "patience",
                      "optimizer_type", "scheduler_type", "activation"}
        for key, value in changes.items():
            if key in _arch_keys:
                config_dict[key] = value
        return TransolverConfig(
            slice_num=config_dict.get("slice_num", 32),
            n_heads=config_dict.get("n_heads", 8),
            d_model=config_dict.get("d_model", 256),
            n_layers=config_dict.get("n_layers", 6),
            mlp_ratio=config_dict.get("mlp_ratio", 4.0),
            dropout=config_dict.get("dropout", 0.0),
            learning_rate=config_dict.get("learning_rate", 1e-3),
            batch_size=config_dict.get("batch_size", 32),
            epochs=config_dict.get("epochs", 1000),
            patience=config_dict.get("patience", 100),
            output_dim=config_dict.get("output_dim", 1),
            energy=config_dict.get("energy", 0.0),
            equilibrium=config_dict.get("equilibrium", 0.0),
            stress_intensity=config_dict.get("stress_intensity", 0.0),
            traction_free=config_dict.get("traction_free", 0.0),
            near_tip=config_dict.get("near_tip", 0.0),
            j_integral=config_dict.get("j_integral", 0.0),
            optimizer_type=config_dict.get("optimizer_type", "adamw"),
            scheduler_type=config_dict.get("scheduler_type", "plateau"),
            activation=config_dict.get("activation", "gelu"),
        )

    async def propose_config(
        self,
        context: AgentContext,
        current_config: TransolverConfig,
        critique: CritiqueResult,
        dataset_size: int = 0,
        previous_configs: Optional[List[Dict]] = None,
        debate_context: str = "",
    ) -> ArchitectureProposal:
        """
        Propose a new configuration based on critique.

        Args:
            context: Agent context
            current_config: Current model configuration
            critique: Critic's analysis results
            dataset_size: Number of training samples
            previous_configs: History of previous configurations
            debate_context: Prior debate rounds (Rounds 1-2) for context

        Returns:
            ArchitectureProposal with new configuration
        """
        proposal = await self.execute(
            context,
            current_config=current_config,
            critique=critique,
            dataset_size=dataset_size,
            previous_configs=previous_configs or [],
            debate_context=debate_context,
        )

        proposal.config = self.apply_changes(current_config, proposal.changes)
        return proposal

    async def analyze(
        self,
        context: AgentContext,
        current_config: "TransolverConfig",
        debate_context: str,
    ) -> str:
        """Round 2: analyze root cause of training issues, no proposals."""
        if self._llm_provider is None:
            raise RuntimeError("LLM provider not set")
        config_str = "\n".join(f"  {k}: {v}" for k, v in current_config.to_dict().items())
        prompt = (
            "ROUND 2 (ANALYSIS) — Do NOT propose any hyperparameter values.\n\n"
            f"## Round 1 Observations\n{debate_context}\n\n"
            f"## Current Architecture\n{config_str}\n\n"
            "Analyze the ROOT CAUSE of training issues seen in Round 1:\n"
            "- Is this an architecture problem, optimizer problem, or data problem?\n"
            "- What specific mechanism explains the observed loss behavior?\n"
            "- Challenge any Round 1 observation you disagree with.\n\n"
            "4-6 sentences. No proposed values."
        )
        response = await self._llm_provider.generate(
            system_prompt=ARCHITECT_SYSTEM,
            user_prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
        )
        return f"[ARCHITECT — Round 2]\n{response.content}"

    def analyze_sync(self, current_config: "TransolverConfig", debate_context: str) -> str:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.analyze(AgentContext(), current_config, debate_context))
        finally:
            loop.close()
