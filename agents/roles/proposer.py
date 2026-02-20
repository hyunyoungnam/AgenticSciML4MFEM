"""
Proposer Agent implementation.

The Proposer suggests mutations to the base model, guided by knowledge,
failure patterns, and evaluation guidelines.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent, AgentContext, AgentRole, AgentMessage, MessageType
from agents.prompts.proposer import PROPOSER_PROMPTS
from evolution.solution import SolutionGenome


@dataclass
class MutationProposal:
    """A mutation proposal from the Proposer agent."""
    mutation_type: str = "morphing"  # morphing, material, boundary_condition, combined
    delta_R: Optional[float] = None
    material_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bc_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reasoning: str = ""
    expected_outcome: str = ""
    risk_assessment: str = ""
    confidence: str = "medium"  # high, medium, low
    raw_response: str = ""

    def to_genome(self) -> SolutionGenome:
        """Convert proposal to a SolutionGenome."""
        return SolutionGenome(
            delta_R=self.delta_R or 0.0,
            material_changes=self.material_changes,
            boundary_condition_changes=self.bc_changes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mutation_type": self.mutation_type,
            "delta_R": self.delta_R,
            "material_changes": self.material_changes,
            "bc_changes": self.bc_changes,
            "reasoning": self.reasoning,
            "expected_outcome": self.expected_outcome,
            "risk_assessment": self.risk_assessment,
            "confidence": self.confidence,
        }


class ProposerAgent(BaseAgent[MutationProposal]):
    """
    Proposer Agent that suggests mutations to FEA models.

    Responsibilities:
    1. Propose intelligent mutations based on knowledge and history
    2. Balance exploration and exploitation
    3. Respond to Critic feedback and refine proposals
    4. Synthesize debate into final proposals
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
        return PROPOSER_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "propose_mutation")

        if task == "propose_mutation":
            return self._build_propose_prompt(context, kwargs)
        elif task == "propose_initial":
            return self._build_initial_prompt(context, kwargs)
        elif task == "refine_proposal":
            return self._build_refine_prompt(context, kwargs)
        elif task == "synthesize_debate":
            return self._build_synthesize_prompt(context, kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _build_propose_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        # Get parent mutations summary
        parent_mutations = kwargs.get("parent_mutations", "Base model (no mutations)")

        # Note: Static knowledge base removed - LLMs have FEA domain knowledge built-in
        knowledge_str = "Use your built-in FEA domain knowledge for best practices."

        # Format failure history (runtime-specific, still useful)
        failure_str = ""
        for failure in context.failure_history[-5:]:  # Last 5 failures
            failure_str += f"- delta_R={failure.get('delta_R', 'N/A')}: {failure.get('error', 'Unknown error')}\n"
        if not failure_str:
            failure_str = "No previous failures recorded."

        # Format guidelines
        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return PROPOSER_PROMPTS["propose_mutation"].format(
            model_info=str(context.model_info),
            solution_id=context.current_solution_id or "new",
            generation=kwargs.get("generation", 0),
            parent_mutations=parent_mutations,
            guidelines=guidelines,
            knowledge_context=knowledge_str,
            failure_history=failure_str,
        )

    def _build_initial_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        # Note: Static knowledge base removed - LLMs have FEA domain knowledge built-in
        knowledge_str = "Use your built-in FEA domain knowledge for best practices."

        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return PROPOSER_PROMPTS["propose_initial"].format(
            model_info=str(context.model_info),
            guidelines=guidelines,
            knowledge_context=knowledge_str,
            population_size=kwargs.get("population_size", 5),
        )

    def _build_refine_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return PROPOSER_PROMPTS["refine_proposal"].format(
            original_proposal=kwargs.get("original_proposal", ""),
            critic_objections=kwargs.get("critic_objections", ""),
            critic_reasoning=kwargs.get("critic_reasoning", ""),
            guidelines=guidelines,
        )

    def _build_synthesize_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return PROPOSER_PROMPTS["synthesize_debate"].format(
            debate_history=kwargs.get("debate_history", ""),
            round1_proposal=kwargs.get("round1_proposal", ""),
            all_objections=kwargs.get("all_objections", ""),
            guidelines=guidelines,
        )

    def parse_response(self, response: str) -> MutationProposal:
        """Parse the LLM response into a MutationProposal."""
        proposal = MutationProposal(raw_response=response)

        # Extract mutation type
        type_match = re.search(
            r'\*\*Mutation Type\*\*:\s*(\w+)',
            response, re.IGNORECASE
        )
        if type_match:
            proposal.mutation_type = type_match.group(1).lower()

        # Extract delta_R (handles markdown formatting like **delta_R**: 0.5)
        delta_r_match = re.search(
            r'\*{0,2}delta_R\*{0,2}[:\s]+([+-]?[0-9.]+)',
            response, re.IGNORECASE
        )
        if delta_r_match:
            try:
                proposal.delta_R = float(delta_r_match.group(1))
            except ValueError:
                pass

        # Extract reasoning
        reasoning_match = re.search(
            r'\*\*Reasoning\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            proposal.reasoning = reasoning_match.group(1).strip()

        # Extract expected outcome
        outcome_match = re.search(
            r'\*\*Expected Outcome\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if outcome_match:
            proposal.expected_outcome = outcome_match.group(1).strip()

        # Extract risk assessment
        risk_match = re.search(
            r'\*\*Risk Assessment\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if risk_match:
            proposal.risk_assessment = risk_match.group(1).strip()

        # Extract confidence level
        conf_match = re.search(
            r'\*\*Confidence(?:\s+Level)?\*\*:\s*(\w+)',
            response, re.IGNORECASE
        )
        if conf_match:
            proposal.confidence = conf_match.group(1).lower()

        # Try to extract material changes
        mat_match = re.search(
            r'material_changes:\s*(\{.*?\})',
            response, re.DOTALL | re.IGNORECASE
        )
        if mat_match:
            try:
                import json
                proposal.material_changes = json.loads(mat_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        return proposal

    async def propose_mutation(
        self,
        context: AgentContext,
        generation: int = 0,
        parent_mutations: str = "",
    ) -> MutationProposal:
        """
        Propose a single mutation.

        Args:
            context: Agent context
            generation: Current generation number
            parent_mutations: Description of parent's mutations

        Returns:
            MutationProposal
        """
        return await self.execute(
            context,
            task="propose_mutation",
            generation=generation,
            parent_mutations=parent_mutations,
        )

    async def propose_initial_population(
        self,
        context: AgentContext,
        population_size: int = 5,
    ) -> List[MutationProposal]:
        """
        Propose initial set of mutations for generation 0.

        Args:
            context: Agent context
            population_size: Number of proposals to generate

        Returns:
            List of MutationProposals
        """
        response = await self.execute(
            context,
            task="propose_initial",
            population_size=population_size,
        )

        # Parse multiple proposals from the response
        proposals = self._parse_multiple_proposals(response.raw_response, population_size)
        return proposals

    def _parse_multiple_proposals(
        self,
        response: str,
        expected_count: int,
    ) -> List[MutationProposal]:
        """Parse multiple proposals from a single response."""
        proposals = []

        # Split by proposal markers
        proposal_blocks = re.split(
            r'\*\*Proposal\s+\d+\*\*:?',
            response, re.IGNORECASE
        )

        for block in proposal_blocks[1:]:  # Skip first empty split
            if block.strip():
                proposal = self.parse_response(block)
                proposals.append(proposal)

        # If we couldn't parse enough, create the response as a single proposal
        if len(proposals) < expected_count and not proposals:
            proposals.append(self.parse_response(response))

        return proposals[:expected_count]

    async def refine_proposal(
        self,
        context: AgentContext,
        original_proposal: str,
        critic_objections: str,
        critic_reasoning: str,
    ) -> MutationProposal:
        """
        Refine a proposal based on Critic feedback.

        Args:
            context: Agent context
            original_proposal: The original proposal text
            critic_objections: Specific objections from the Critic
            critic_reasoning: Critic's full reasoning

        Returns:
            Refined MutationProposal
        """
        return await self.execute(
            context,
            task="refine_proposal",
            original_proposal=original_proposal,
            critic_objections=critic_objections,
            critic_reasoning=critic_reasoning,
        )

    async def synthesize_debate(
        self,
        context: AgentContext,
        debate_history: str,
        round1_proposal: str,
        all_objections: str,
    ) -> MutationProposal:
        """
        Synthesize debate into final proposal (Round 3).

        Args:
            context: Agent context
            debate_history: Full debate history
            round1_proposal: Original Round 1 proposal
            all_objections: All Critic objections

        Returns:
            Final MutationProposal
        """
        return await self.execute(
            context,
            task="synthesize_debate",
            debate_history=debate_history,
            round1_proposal=round1_proposal,
            all_objections=all_objections,
        )
