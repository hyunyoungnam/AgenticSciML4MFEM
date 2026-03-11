"""
Critic Agent implementation.

The Critic performs pre-flight validation and challenges proposals
through structured debate to ensure solution quality.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from inpforge.agents.base import BaseAgent, AgentContext, AgentRole, AgentMessage, MessageType
from inpforge.agents.prompts.critic import CRITIC_PROMPTS


class CriticVote(Enum):
    """Critic's vote on a proposal."""
    APPROVE = "approve"
    REJECT = "reject"
    LEAN_APPROVE = "lean_approve"
    LEAN_REJECT = "lean_reject"
    NEEDS_REVISION = "needs_revision"


@dataclass
class Critique:
    """A critique from the Critic agent."""
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    mesh_quality_risk: str = "medium"  # low, medium, high
    convergence_risk: str = "medium"
    physical_plausibility: str = "acceptable"  # acceptable, questionable, unacceptable
    vote: CriticVote = CriticVote.NEEDS_REVISION
    confidence: str = "medium"
    implementation_notes: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strengths": self.strengths,
            "concerns": self.concerns,
            "objections": self.objections,
            "recommendations": self.recommendations,
            "mesh_quality_risk": self.mesh_quality_risk,
            "convergence_risk": self.convergence_risk,
            "physical_plausibility": self.physical_plausibility,
            "vote": self.vote.value,
            "confidence": self.confidence,
            "implementation_notes": self.implementation_notes,
        }

    @property
    def is_approved(self) -> bool:
        """Check if the vote is an approval."""
        return self.vote in (CriticVote.APPROVE, CriticVote.LEAN_APPROVE)


class CriticAgent(BaseAgent[Critique]):
    """
    Critic Agent that validates proposals and provides feedback.

    Responsibilities:
    1. Challenge mutation proposals to identify issues
    2. Verify against guidelines and physical constraints
    3. Predict failure modes
    4. Provide constructive feedback
    """

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.CRITIC,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return CRITIC_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "critique_proposal")

        if task == "critique_proposal":
            return self._build_critique_prompt(context, kwargs)
        elif task == "respond_to_refinement":
            return self._build_response_prompt(context, kwargs)
        elif task == "final_vote":
            return self._build_final_vote_prompt(context, kwargs)
        elif task == "quick_validate":
            return self._build_quick_validate_prompt(context, kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _build_critique_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        # Note: Static knowledge base removed - LLMs have FEA domain knowledge built-in
        knowledge_str = "Use your built-in FEA domain knowledge for validation."

        # Format failure history (runtime-specific, still useful)
        failure_str = ""
        for failure in context.failure_history[-5:]:
            failure_str += f"- delta_R={failure.get('delta_R', 'N/A')}: {failure.get('error', 'Unknown')}\n"
        if not failure_str:
            failure_str = "No previous failures recorded."

        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return CRITIC_PROMPTS["critique_proposal"].format(
            proposal=kwargs.get("proposal", ""),
            guidelines=guidelines,
            model_info=str(context.model_info),
            knowledge_context=knowledge_str,
            failure_history=failure_str,
        )

    def _build_response_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return CRITIC_PROMPTS["respond_to_refinement"].format(
            original_proposal=kwargs.get("original_proposal", ""),
            previous_critique=kwargs.get("previous_critique", ""),
            refined_proposal=kwargs.get("refined_proposal", ""),
            guidelines=guidelines,
        )

    def _build_final_vote_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return CRITIC_PROMPTS["final_vote"].format(
            final_proposal=kwargs.get("final_proposal", ""),
            debate_summary=kwargs.get("debate_summary", ""),
            guidelines=guidelines,
            model_info=str(context.model_info),
        )

    def _build_quick_validate_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        guidelines = context.evaluation_criteria.get("guidelines", "No guidelines available.")

        return CRITIC_PROMPTS["quick_validate"].format(
            proposal=kwargs.get("proposal", ""),
            guidelines=guidelines,
        )

    def parse_response(self, response: str) -> Critique:
        """Parse the LLM response into a Critique."""
        critique = Critique(raw_response=response)

        # Extract strengths
        strengths_match = re.search(
            r'\*\*Strengths\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if strengths_match:
            text = strengths_match.group(1)
            critique.strengths = [
                line.strip().lstrip('- ').lstrip('* ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract concerns
        concerns_match = re.search(
            r'\*\*Concerns\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if concerns_match:
            text = concerns_match.group(1)
            critique.concerns = [
                line.strip().lstrip('- ').lstrip('* ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract objections
        objections_match = re.search(
            r'\*\*(?:Specific )?Objections\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if objections_match:
            text = objections_match.group(1)
            critique.objections = [
                line.strip().lstrip('- ').lstrip('* ').lstrip('0123456789. ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract recommendations
        rec_match = re.search(
            r'\*\*Recommendations\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if rec_match:
            text = rec_match.group(1)
            critique.recommendations = [
                line.strip().lstrip('- ').lstrip('* ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract risk levels
        mesh_risk_match = re.search(
            r'Mesh Quality Risk:\s*(\w+)',
            response, re.IGNORECASE
        )
        if mesh_risk_match:
            critique.mesh_quality_risk = mesh_risk_match.group(1).lower()

        conv_risk_match = re.search(
            r'Convergence Risk:\s*(\w+)',
            response, re.IGNORECASE
        )
        if conv_risk_match:
            critique.convergence_risk = conv_risk_match.group(1).lower()

        phys_match = re.search(
            r'Physical Plausibility:\s*(\w+)',
            response, re.IGNORECASE
        )
        if phys_match:
            critique.physical_plausibility = phys_match.group(1).lower()

        # Extract vote
        vote = self._extract_vote(response)
        critique.vote = vote

        # Extract confidence
        conf_match = re.search(
            r'\*\*Confidence\*\*:\s*(\w+)',
            response, re.IGNORECASE
        )
        if conf_match:
            critique.confidence = conf_match.group(1).lower()

        # Extract implementation notes
        impl_match = re.search(
            r'\*\*(?:If Approved - )?Implementation Notes\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if impl_match:
            critique.implementation_notes = impl_match.group(1).strip()

        return critique

    def _extract_vote(self, response: str) -> CriticVote:
        """Extract the vote from the response."""
        response_upper = response.upper()

        # Check for final vote first
        final_match = re.search(
            r'\*\*FINAL VOTE\*\*:\s*(\w+)',
            response, re.IGNORECASE
        )
        if final_match:
            vote_text = final_match.group(1).upper()
            if vote_text == "APPROVE":
                return CriticVote.APPROVE
            elif vote_text == "REJECT":
                return CriticVote.REJECT

        # Check for preliminary/updated vote
        prelim_match = re.search(
            r'\*\*(?:Preliminary|Updated)\s*Vote\*\*:\s*(\w+)',
            response, re.IGNORECASE
        )
        if prelim_match:
            vote_text = prelim_match.group(1).upper()
            if "APPROVE" in vote_text:
                if "LEAN" in vote_text:
                    return CriticVote.LEAN_APPROVE
                return CriticVote.APPROVE
            elif "REJECT" in vote_text:
                if "LEAN" in vote_text:
                    return CriticVote.LEAN_REJECT
                return CriticVote.REJECT
            elif "REVISION" in vote_text:
                return CriticVote.NEEDS_REVISION

        # Fallback: look for key phrases
        if "LEAN_APPROVE" in response_upper or "LEAN APPROVE" in response_upper:
            return CriticVote.LEAN_APPROVE
        elif "LEAN_REJECT" in response_upper or "LEAN REJECT" in response_upper:
            return CriticVote.LEAN_REJECT
        elif "NEEDS_REVISION" in response_upper or "NEEDS REVISION" in response_upper:
            return CriticVote.NEEDS_REVISION
        elif "APPROVE" in response_upper:
            return CriticVote.APPROVE if "REJECT" not in response_upper else CriticVote.LEAN_APPROVE
        elif "REJECT" in response_upper:
            return CriticVote.REJECT

        return CriticVote.NEEDS_REVISION

    async def critique_proposal(
        self,
        context: AgentContext,
        proposal: str,
    ) -> Critique:
        """
        Critique a mutation proposal.

        Args:
            context: Agent context
            proposal: The proposal to critique

        Returns:
            Critique with feedback and vote
        """
        return await self.execute(context, task="critique_proposal", proposal=proposal)

    async def respond_to_refinement(
        self,
        context: AgentContext,
        original_proposal: str,
        previous_critique: str,
        refined_proposal: str,
    ) -> Critique:
        """
        Respond to a refined proposal.

        Args:
            context: Agent context
            original_proposal: The original proposal
            previous_critique: Previous critique text
            refined_proposal: The refined proposal

        Returns:
            Updated Critique
        """
        return await self.execute(
            context,
            task="respond_to_refinement",
            original_proposal=original_proposal,
            previous_critique=previous_critique,
            refined_proposal=refined_proposal,
        )

    async def final_vote(
        self,
        context: AgentContext,
        final_proposal: str,
        debate_summary: str,
    ) -> Critique:
        """
        Provide final vote on a proposal (Round 4).

        Args:
            context: Agent context
            final_proposal: The final proposal
            debate_summary: Summary of the debate

        Returns:
            Final Critique with APPROVE/REJECT vote
        """
        return await self.execute(
            context,
            task="final_vote",
            final_proposal=final_proposal,
            debate_summary=debate_summary,
        )

    async def quick_validate(
        self,
        context: AgentContext,
        proposal: str,
    ) -> bool:
        """
        Quick validation check against basic constraints.

        Args:
            context: Agent context
            proposal: Proposal to validate

        Returns:
            True if proposal passes basic validation
        """
        critique = await self.execute(
            context,
            task="quick_validate",
            proposal=proposal,
        )
        return critique.vote in (CriticVote.APPROVE, CriticVote.LEAN_APPROVE)
