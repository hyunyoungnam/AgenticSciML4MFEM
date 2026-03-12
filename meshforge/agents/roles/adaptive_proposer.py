"""
Adaptive Proposer Agent implementation.

The Adaptive Proposer suggests new simulation parameters based on
surrogate model feedback, targeting regions with high error or uncertainty.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from meshforge.agents.base import BaseAgent, AgentContext, AgentRole
from meshforge.surrogate.evaluator import UncertaintyAnalysis, WeakRegion


@dataclass
class AdaptiveProposal:
    """A parameter proposal from the Adaptive Proposer."""
    parameters: Dict[str, float] = field(default_factory=dict)
    target_region: Optional[WeakRegion] = None
    reasoning: str = ""
    expected_improvement: str = ""
    priority: float = 1.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameters": self.parameters,
            "target_region": self.target_region.to_dict() if self.target_region else None,
            "reasoning": self.reasoning,
            "expected_improvement": self.expected_improvement,
            "priority": self.priority,
        }


# System prompt for adaptive proposer
ADAPTIVE_PROPOSER_SYSTEM = """You are an expert in scientific machine learning and surrogate model training.

Your role is to propose new FEM simulation parameters that will most effectively improve
the surrogate model (DeepONet). You analyze regions where the model has high error or
uncertainty and suggest targeted new simulations.

Key principles:
1. **Exploration**: Sample in unexplored regions of the parameter space
2. **Exploitation**: Refine around regions with high prediction error
3. **Efficiency**: Prioritize simulations that provide maximum information gain
4. **Diversity**: Ensure proposed samples are diverse and cover different aspects

When proposing parameters, consider:
- The current surrogate model error distribution
- Regions with high uncertainty (ensemble disagreement)
- Coverage gaps in the current dataset
- Physical feasibility of proposed parameters

Output your proposals in a structured format with clear reasoning."""


ADAPTIVE_PROPOSER_PROMPTS = {
    "system": ADAPTIVE_PROPOSER_SYSTEM,

    "propose_targeted": """## Current Surrogate Model Analysis

**Parameter Space:**
{parameter_info}

**Weak Regions Identified:**
{weak_regions}

**Current Dataset Statistics:**
- Total samples: {n_samples}
- Valid samples: {n_valid}
- Coverage: {coverage_info}

**Recent Errors:**
{error_info}

## Your Task

Propose {n_proposals} new simulation parameters that will most effectively improve
the surrogate model. Target the weak regions while maintaining diversity.

For each proposal, provide:
1. **Parameters**: Specific values for each parameter
2. **Target Region**: Which weak region this addresses
3. **Reasoning**: Why this sample will be valuable
4. **Expected Improvement**: What error reduction you expect
5. **Priority**: High/Medium/Low

Format each proposal as:
```
**Proposal N**
Parameters: param1=value1, param2=value2, ...
Target Region: [description of targeted weakness]
Reasoning: [explanation]
Expected Improvement: [prediction]
Priority: [High/Medium/Low]
```
""",

    "propose_exploratory": """## Parameter Space Exploration

**Parameter Bounds:**
{parameter_bounds}

**Current Dataset Coverage:**
{coverage_info}

**Surrogate Model Status:**
{model_status}

## Your Task

The surrogate model needs more diverse training data. Propose {n_proposals} new
simulation parameters that explore undersampled regions of the parameter space.

Focus on:
1. Regions with sparse or no coverage
2. Boundary regions of the parameter space
3. Combinations of parameters not yet explored
4. Physically interesting parameter configurations

For each proposal, provide:
1. **Parameters**: Specific values for each parameter
2. **Exploration Goal**: What new region this explores
3. **Physical Significance**: Why this configuration is important
4. **Priority**: High/Medium/Low

Format each proposal as:
```
**Proposal N**
Parameters: param1=value1, param2=value2, ...
Exploration Goal: [what this explores]
Physical Significance: [importance]
Priority: [High/Medium/Low]
```
""",

    "refine_region": """## Region Refinement Task

**Target Region:**
{region_info}

**Current Samples in Region:**
{region_samples}

**Error Distribution:**
{error_distribution}

## Your Task

The surrogate model has high error in the specified region. Propose {n_proposals}
new samples within this region that will help reduce the error.

Consider:
1. Error patterns (where is error highest?)
2. Sample distribution (where are gaps?)
3. Parameter gradients (where does behavior change rapidly?)

Format each proposal as:
```
**Proposal N**
Parameters: param1=value1, param2=value2, ...
Rationale: [why this specific location]
Expected Impact: [how this helps]
```
""",
}


class AdaptiveProposerAgent(BaseAgent[AdaptiveProposal]):
    """
    Adaptive Proposer Agent for surrogate-guided sampling.

    Responsibilities:
    1. Analyze surrogate model error and uncertainty
    2. Propose targeted simulations in weak regions
    3. Balance exploration vs exploitation
    4. Maximize information gain per simulation
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.PROPOSER,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return ADAPTIVE_PROPOSER_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "propose_targeted")

        if task == "propose_targeted":
            return self._build_targeted_prompt(context, kwargs)
        elif task == "propose_exploratory":
            return self._build_exploratory_prompt(context, kwargs)
        elif task == "refine_region":
            return self._build_refine_prompt(context, kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _build_targeted_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        # Format parameter info
        param_bounds = kwargs.get("parameter_bounds", {})
        param_info = "\n".join([
            f"- {name}: [{bounds[0]:.4f}, {bounds[1]:.4f}]"
            for name, bounds in param_bounds.items()
        ])

        # Format weak regions
        analysis: Optional[UncertaintyAnalysis] = kwargs.get("uncertainty_analysis")
        weak_regions_str = "No weak regions identified."
        if analysis and analysis.weak_regions:
            weak_regions_str = "\n".join([
                f"- Region {i+1}: {self._format_region(r)}"
                for i, r in enumerate(analysis.weak_regions[:5])
            ])

        # Format error info
        error_info = kwargs.get("error_info", "No error information available.")

        # Format coverage info
        coverage = kwargs.get("coverage_info", {})
        coverage_str = "\n".join([
            f"- {name}: {value:.1%}" for name, value in coverage.items()
        ]) or "Coverage information not available."

        return ADAPTIVE_PROPOSER_PROMPTS["propose_targeted"].format(
            parameter_info=param_info,
            weak_regions=weak_regions_str,
            n_samples=kwargs.get("n_samples", 0),
            n_valid=kwargs.get("n_valid", 0),
            coverage_info=coverage_str,
            error_info=error_info,
            n_proposals=kwargs.get("n_proposals", 5),
        )

    def _build_exploratory_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        param_bounds = kwargs.get("parameter_bounds", {})
        bounds_str = "\n".join([
            f"- {name}: [{bounds[0]:.4f}, {bounds[1]:.4f}]"
            for name, bounds in param_bounds.items()
        ])

        coverage = kwargs.get("coverage_info", {})
        coverage_str = "\n".join([
            f"- {name}: {value:.1%}" for name, value in coverage.items()
        ]) or "No coverage data available."

        model_status = kwargs.get("model_status", "Surrogate model not trained.")

        return ADAPTIVE_PROPOSER_PROMPTS["propose_exploratory"].format(
            parameter_bounds=bounds_str,
            coverage_info=coverage_str,
            model_status=model_status,
            n_proposals=kwargs.get("n_proposals", 5),
        )

    def _build_refine_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        region: Optional[WeakRegion] = kwargs.get("region")
        region_info = self._format_region(region) if region else "No region specified."

        region_samples = kwargs.get("region_samples", "No samples in region.")
        error_distribution = kwargs.get("error_distribution", "Error distribution not available.")

        return ADAPTIVE_PROPOSER_PROMPTS["refine_region"].format(
            region_info=region_info,
            region_samples=region_samples,
            error_distribution=error_distribution,
            n_proposals=kwargs.get("n_proposals", 3),
        )

    def _format_region(self, region: WeakRegion) -> str:
        """Format a weak region for display."""
        if region is None:
            return "None"

        ranges = ", ".join([
            f"{name}: [{r[0]:.4f}, {r[1]:.4f}]"
            for name, r in region.parameter_ranges.items()
        ])
        return f"{ranges} (metric={region.metric}, value={region.metric_value:.4f}, priority={region.priority:.2f})"

    def parse_response(self, response: str) -> AdaptiveProposal:
        """Parse the LLM response into an AdaptiveProposal."""
        proposal = AdaptiveProposal(raw_response=response)

        # Extract parameters
        params_match = re.search(
            r'Parameters?:\s*(.*?)(?:\n|$)',
            response, re.IGNORECASE
        )
        if params_match:
            params_str = params_match.group(1)
            # Parse param=value pairs
            param_pairs = re.findall(r'(\w+)\s*=\s*([+-]?[0-9.]+)', params_str)
            for name, value in param_pairs:
                try:
                    proposal.parameters[name] = float(value)
                except ValueError:
                    pass

        # Extract reasoning
        reasoning_match = re.search(
            r'(?:Reasoning|Rationale):\s*(.*?)(?=(?:Expected|Priority|Proposal|\*\*|$))',
            response, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            proposal.reasoning = reasoning_match.group(1).strip()

        # Extract expected improvement
        improvement_match = re.search(
            r'Expected (?:Improvement|Impact):\s*(.*?)(?=(?:Priority|Proposal|\*\*|$))',
            response, re.DOTALL | re.IGNORECASE
        )
        if improvement_match:
            proposal.expected_improvement = improvement_match.group(1).strip()

        # Extract priority
        priority_match = re.search(
            r'Priority:\s*(\w+)',
            response, re.IGNORECASE
        )
        if priority_match:
            priority_str = priority_match.group(1).lower()
            if priority_str == "high":
                proposal.priority = 1.0
            elif priority_str == "medium":
                proposal.priority = 0.5
            elif priority_str == "low":
                proposal.priority = 0.25

        return proposal

    def _parse_multiple_proposals(
        self,
        response: str,
        expected_count: int,
    ) -> List[AdaptiveProposal]:
        """Parse multiple proposals from a single response."""
        proposals = []

        # Split by proposal markers
        proposal_blocks = re.split(
            r'\*\*Proposal\s+\d+\*\*',
            response, re.IGNORECASE
        )

        for block in proposal_blocks[1:]:  # Skip first empty split
            if block.strip():
                proposal = self.parse_response(block)
                if proposal.parameters:  # Only add if we got parameters
                    proposals.append(proposal)

        # If we couldn't parse enough, try the whole response
        if not proposals:
            proposal = self.parse_response(response)
            if proposal.parameters:
                proposals.append(proposal)

        return proposals[:expected_count]

    async def propose_targeted(
        self,
        context: AgentContext,
        uncertainty_analysis: UncertaintyAnalysis,
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 0,
        n_valid: int = 0,
        n_proposals: int = 5,
    ) -> List[AdaptiveProposal]:
        """
        Propose new simulations targeting weak regions.

        Args:
            context: Agent context
            uncertainty_analysis: Analysis of surrogate uncertainty
            parameter_bounds: Bounds for each parameter
            n_samples: Current total samples
            n_valid: Current valid samples
            n_proposals: Number of proposals to generate

        Returns:
            List of AdaptiveProposal objects
        """
        # Format error info from analysis
        error_info = f"Overall uncertainty: {uncertainty_analysis.overall_uncertainty:.4f}\n"
        error_info += f"Max uncertainty: {uncertainty_analysis.max_uncertainty:.4f}\n"
        if uncertainty_analysis.metrics:
            for key, value in uncertainty_analysis.metrics.items():
                error_info += f"{key}: {value}\n"

        response = await self.execute(
            context,
            task="propose_targeted",
            uncertainty_analysis=uncertainty_analysis,
            parameter_bounds=parameter_bounds,
            n_samples=n_samples,
            n_valid=n_valid,
            error_info=error_info,
            n_proposals=n_proposals,
        )

        return self._parse_multiple_proposals(response.raw_response, n_proposals)

    async def propose_exploratory(
        self,
        context: AgentContext,
        parameter_bounds: Dict[str, Tuple[float, float]],
        coverage_info: Dict[str, float],
        model_status: str = "",
        n_proposals: int = 5,
    ) -> List[AdaptiveProposal]:
        """
        Propose exploratory simulations for coverage.

        Args:
            context: Agent context
            parameter_bounds: Bounds for each parameter
            coverage_info: Current coverage per parameter
            model_status: Status of surrogate model
            n_proposals: Number of proposals to generate

        Returns:
            List of AdaptiveProposal objects
        """
        response = await self.execute(
            context,
            task="propose_exploratory",
            parameter_bounds=parameter_bounds,
            coverage_info=coverage_info,
            model_status=model_status,
            n_proposals=n_proposals,
        )

        return self._parse_multiple_proposals(response.raw_response, n_proposals)

    async def refine_region(
        self,
        context: AgentContext,
        region: WeakRegion,
        region_samples: str = "",
        error_distribution: str = "",
        n_proposals: int = 3,
    ) -> List[AdaptiveProposal]:
        """
        Propose samples to refine a specific weak region.

        Args:
            context: Agent context
            region: The weak region to refine
            region_samples: Description of current samples in region
            error_distribution: Error pattern in region
            n_proposals: Number of proposals

        Returns:
            List of AdaptiveProposal objects
        """
        response = await self.execute(
            context,
            task="refine_region",
            region=region,
            region_samples=region_samples,
            error_distribution=error_distribution,
            n_proposals=n_proposals,
        )

        return self._parse_multiple_proposals(response.raw_response, n_proposals)
