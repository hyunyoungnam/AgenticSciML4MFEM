"""
Debate Controller for structured Proposer-Critic debates.

Orchestrates N-round debates between the Proposer and Critic agents
to refine mutation proposals before implementation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from inpforge.agents.base import AgentContext
from inpforge.agents.roles.proposer import ProposerAgent, MutationProposal
from inpforge.agents.roles.critic import CriticAgent, Critique, CriticVote
from inpforge.evolution.solution import DebateRound


class DebateOutcome(Enum):
    """Outcome of a debate."""
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"  # Max rounds reached without resolution


@dataclass
class DebateResult:
    """Result of a complete debate."""
    outcome: DebateOutcome
    final_proposal: Optional[MutationProposal] = None
    final_critique: Optional[Critique] = None
    rounds: List[DebateRound] = field(default_factory=list)
    total_rounds: int = 0
    consensus_score: float = 0.0  # 0-1, how much agreement was reached
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_approved(self) -> bool:
        """Check if the proposal was approved."""
        return self.outcome == DebateOutcome.APPROVED

    def get_summary(self) -> str:
        """Get a summary of the debate."""
        lines = [
            f"Debate Outcome: {self.outcome.value}",
            f"Total Rounds: {self.total_rounds}",
            f"Consensus Score: {self.consensus_score:.2f}",
        ]

        if self.final_proposal and self.final_proposal.delta_R is not None:
            lines.append(f"Final delta_R: {self.final_proposal.delta_R}")

        if self.final_critique:
            lines.append(f"Final Vote: {self.final_critique.vote.value}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome.value,
            "final_proposal": self.final_proposal.to_dict() if self.final_proposal else None,
            "final_critique": self.final_critique.to_dict() if self.final_critique else None,
            "rounds": [r.to_dict() for r in self.rounds],
            "total_rounds": self.total_rounds,
            "consensus_score": self.consensus_score,
            "metadata": self.metadata,
        }


class DebateController:
    """
    Controller for structured Proposer-Critic debates.

    Orchestrates a multi-round debate:
    - Round 1-2: Proposer presents, Critic challenges, Proposer refines
    - Round 3: Proposer synthesizes feedback
    - Round 4: Final proposal, Critic votes APPROVE/REJECT

    The debate can end early if consensus is reached.
    """

    def __init__(
        self,
        proposer: ProposerAgent,
        critic: CriticAgent,
        num_rounds: int = 4,
        consensus_threshold: float = 0.7,
    ):
        """
        Initialize the debate controller.

        Args:
            proposer: ProposerAgent instance
            critic: CriticAgent instance
            num_rounds: Maximum number of debate rounds
            consensus_threshold: Threshold for early consensus (0-1)
        """
        self.proposer = proposer
        self.critic = critic
        self.num_rounds = num_rounds
        self.consensus_threshold = consensus_threshold

    async def run_debate(
        self,
        context: AgentContext,
        initial_proposal: Optional[MutationProposal] = None,
        generation: int = 0,
        parent_mutations: str = "",
    ) -> DebateResult:
        """
        Run a complete debate on a mutation proposal.

        Args:
            context: Shared agent context
            initial_proposal: Optional pre-generated proposal
            generation: Current generation number
            parent_mutations: Description of parent mutations

        Returns:
            DebateResult with outcome, final proposal, and debate history
        """
        rounds: List[DebateRound] = []
        current_proposal = initial_proposal
        all_objections: List[str] = []
        proposal_text = ""

        # Round 1: Initial proposal and critique
        if current_proposal is None:
            current_proposal = await self.proposer.propose_mutation(
                context,
                generation=generation,
                parent_mutations=parent_mutations,
            )

        proposal_text = current_proposal.raw_response
        round1_proposal = proposal_text

        critique = await self.critic.critique_proposal(context, proposal_text)
        all_objections.extend(critique.objections)

        rounds.append(DebateRound(
            round_number=1,
            proposer_message=proposal_text,
            critic_message=critique.raw_response,
            proposer_reasoning=current_proposal.reasoning,
            critic_objections=critique.objections,
        ))

        # Check for early consensus
        if critique.vote == CriticVote.APPROVE:
            return self._create_result(
                DebateOutcome.APPROVED,
                current_proposal,
                critique,
                rounds,
                consensus_score=1.0,
            )

        if critique.vote == CriticVote.REJECT and critique.confidence == "high":
            # Strong rejection - no point continuing
            return self._create_result(
                DebateOutcome.REJECTED,
                current_proposal,
                critique,
                rounds,
                consensus_score=0.0,
            )

        # Round 2: Proposer refines based on critique
        previous_critique = critique.raw_response
        refined_proposal = await self.proposer.refine_proposal(
            context,
            original_proposal=proposal_text,
            critic_objections="\n".join(critique.objections),
            critic_reasoning=previous_critique,
        )
        current_proposal = refined_proposal
        proposal_text = refined_proposal.raw_response

        # Critic responds to refinement
        critique = await self.critic.respond_to_refinement(
            context,
            original_proposal=round1_proposal,
            previous_critique=previous_critique,
            refined_proposal=proposal_text,
        )
        all_objections.extend(critique.objections)

        rounds.append(DebateRound(
            round_number=2,
            proposer_message=proposal_text,
            critic_message=critique.raw_response,
            proposer_reasoning=current_proposal.reasoning,
            critic_objections=critique.objections,
        ))

        # Check for consensus
        if critique.vote == CriticVote.APPROVE:
            return self._create_result(
                DebateOutcome.APPROVED,
                current_proposal,
                critique,
                rounds,
                consensus_score=0.9,
            )

        # Round 3: Proposer synthesizes all feedback
        debate_history = self._format_debate_history(rounds)
        synthesized_proposal = await self.proposer.synthesize_debate(
            context,
            debate_history=debate_history,
            round1_proposal=round1_proposal,
            all_objections="\n".join(all_objections),
        )
        current_proposal = synthesized_proposal
        proposal_text = synthesized_proposal.raw_response

        # Critic provides updated assessment
        critique = await self.critic.respond_to_refinement(
            context,
            original_proposal=round1_proposal,
            previous_critique=critique.raw_response,
            refined_proposal=proposal_text,
        )

        rounds.append(DebateRound(
            round_number=3,
            proposer_message=proposal_text,
            critic_message=critique.raw_response,
            proposer_reasoning=synthesized_proposal.reasoning,
            critic_objections=critique.objections,
        ))

        # Round 4: Final vote
        debate_summary = self._create_debate_summary(rounds)
        final_critique = await self.critic.final_vote(
            context,
            final_proposal=proposal_text,
            debate_summary=debate_summary,
        )

        rounds.append(DebateRound(
            round_number=4,
            proposer_message=proposal_text,
            critic_message=final_critique.raw_response,
            proposer_reasoning=current_proposal.reasoning,
            critic_objections=final_critique.objections,
        ))

        # Determine outcome
        if final_critique.vote == CriticVote.APPROVE:
            outcome = DebateOutcome.APPROVED
            consensus_score = 0.8 + (0.2 if final_critique.confidence == "high" else 0.0)
        elif final_critique.vote == CriticVote.REJECT:
            outcome = DebateOutcome.REJECTED
            consensus_score = 0.0
        else:
            # Still undecided after all rounds
            outcome = DebateOutcome.TIMEOUT
            consensus_score = self._calculate_consensus(final_critique)

        return self._create_result(
            outcome,
            current_proposal,
            final_critique,
            rounds,
            consensus_score=consensus_score,
        )

    def _format_debate_history(self, rounds: List[DebateRound]) -> str:
        """Format debate history for context."""
        lines = []
        for r in rounds:
            lines.append(f"## Round {r.round_number}")
            lines.append(f"### Proposer")
            lines.append(r.proposer_message[:500] + "..." if len(r.proposer_message) > 500 else r.proposer_message)
            lines.append(f"### Critic")
            lines.append(r.critic_message[:500] + "..." if len(r.critic_message) > 500 else r.critic_message)
            lines.append("")
        return "\n".join(lines)

    def _create_debate_summary(self, rounds: List[DebateRound]) -> str:
        """Create a concise summary of the debate."""
        lines = ["## Debate Summary", ""]

        for r in rounds:
            lines.append(f"**Round {r.round_number}:**")
            if r.proposer_reasoning:
                lines.append(f"- Proposer reasoning: {r.proposer_reasoning[:200]}...")
            if r.critic_objections:
                lines.append(f"- Critic objections: {', '.join(r.critic_objections[:3])}")
            lines.append("")

        return "\n".join(lines)

    def _calculate_consensus(self, critique: Critique) -> float:
        """Calculate consensus score from critique."""
        score = 0.5  # Start neutral

        if critique.vote == CriticVote.LEAN_APPROVE:
            score = 0.6
        elif critique.vote == CriticVote.LEAN_REJECT:
            score = 0.4
        elif critique.vote == CriticVote.NEEDS_REVISION:
            score = 0.5

        # Adjust based on risk levels
        if critique.mesh_quality_risk == "low":
            score += 0.1
        elif critique.mesh_quality_risk == "high":
            score -= 0.1

        if critique.convergence_risk == "low":
            score += 0.1
        elif critique.convergence_risk == "high":
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _create_result(
        self,
        outcome: DebateOutcome,
        proposal: MutationProposal,
        critique: Critique,
        rounds: List[DebateRound],
        consensus_score: float,
    ) -> DebateResult:
        """Create a DebateResult."""
        return DebateResult(
            outcome=outcome,
            final_proposal=proposal,
            final_critique=critique,
            rounds=rounds,
            total_rounds=len(rounds),
            consensus_score=consensus_score,
            metadata={
                "proposer_model": self.proposer.model,
                "critic_model": self.critic.model,
            },
        )

    async def quick_debate(
        self,
        context: AgentContext,
        proposal: MutationProposal,
    ) -> DebateResult:
        """
        Run a quick 2-round debate for less critical proposals.

        Args:
            context: Agent context
            proposal: Proposal to debate

        Returns:
            DebateResult
        """
        rounds: List[DebateRound] = []
        proposal_text = proposal.raw_response

        # Round 1: Critique
        critique = await self.critic.critique_proposal(context, proposal_text)
        rounds.append(DebateRound(
            round_number=1,
            proposer_message=proposal_text,
            critic_message=critique.raw_response,
            proposer_reasoning=proposal.reasoning,
            critic_objections=critique.objections,
        ))

        if critique.vote == CriticVote.APPROVE:
            return self._create_result(
                DebateOutcome.APPROVED,
                proposal,
                critique,
                rounds,
                consensus_score=1.0,
            )

        if critique.vote == CriticVote.REJECT:
            return self._create_result(
                DebateOutcome.REJECTED,
                proposal,
                critique,
                rounds,
                consensus_score=0.0,
            )

        # Round 2: Refine and final vote
        refined = await self.proposer.refine_proposal(
            context,
            original_proposal=proposal_text,
            critic_objections="\n".join(critique.objections),
            critic_reasoning=critique.raw_response,
        )

        final_critique = await self.critic.final_vote(
            context,
            final_proposal=refined.raw_response,
            debate_summary=f"Initial objections: {critique.objections}",
        )

        rounds.append(DebateRound(
            round_number=2,
            proposer_message=refined.raw_response,
            critic_message=final_critique.raw_response,
            proposer_reasoning=refined.reasoning,
            critic_objections=final_critique.objections,
        ))

        outcome = (
            DebateOutcome.APPROVED
            if final_critique.vote == CriticVote.APPROVE
            else DebateOutcome.REJECTED
        )

        return self._create_result(
            outcome,
            refined,
            final_critique,
            rounds,
            consensus_score=0.7 if outcome == DebateOutcome.APPROVED else 0.3,
        )
