"""
Multi-round Agent Debate Orchestrator.

Implements the 4-round structured debate from the AgenticSciML paper (Jiang & Karniadakis):

  Round 1 OBSERVATION  — Analyst + Critic describe training state (no proposals)
  Round 2 ANALYSIS     — Architect + Physicist reason about root causes (no proposals)
  Round 3 SYNTHESIS    — Architect + Physicist propose concrete changes (N candidates)
  Round 4 FINALIZATION — Critic validates each candidate before they are applied

Ensemble branching (n_candidates > 1) generates N independent proposals in Round 3.
Each candidate is then brief-trained by AgenticSurrogateTrainer to pick the best one.
This fixes two failure modes of single-round single-proposal debate:
  1. Bad proposals get applied immediately without validation against real training
  2. Stuck-in-loop: identical proposals repeat when greedy search plateaus
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent, TrainingHistory
    from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.result_analyst import ResultAnalystAgent
    from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent, KBEntry
    from piano.surrogate.base import TransolverConfig

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """Output of one 4-round debate candidate."""
    arch_proposal: Any  # ArchitectureProposal
    physics_changes: Dict[str, Any]
    debate_log: List[str] = field(default_factory=list)
    validation_text: str = ""


class DebateOrchestrator:
    """
    4-round multi-agent debate for HPO with optional ensemble branching.

    Rounds 1–2 run once (shared context). Round 3 runs n_candidates times to
    generate diverse proposals. Round 4 validates each candidate independently.
    """

    def __init__(
        self,
        analyst: "ResultAnalystAgent",
        critic: "HyperparameterCriticAgent",
        architect: "ArchitectAgent",
        physicist: "PhysicistAgent",
        knowledge_retriever: Optional["KnowledgeRetrieverAgent"] = None,
    ):
        self.analyst = analyst
        self.critic = critic
        self.architect = architect
        self.physicist = physicist
        self.knowledge_retriever = knowledge_retriever

    def run_debate_sync(
        self,
        history: "TrainingHistory",
        current_config: "TransolverConfig",
        dataset_size: int,
        config_history: List[Dict[str, Any]],
        problem_type: str = "crack",
        has_singularity: bool = True,
    ) -> DebateResult:
        """Run a single-candidate debate and return one DebateResult."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                self._run_debate(
                    history, current_config, dataset_size,
                    config_history, problem_type, has_singularity,
                    n_candidates=1,
                )
            )
            return results[0]
        finally:
            loop.close()

    def run_ensemble_debates_sync(
        self,
        history: "TrainingHistory",
        current_config: "TransolverConfig",
        dataset_size: int,
        config_history: List[Dict[str, Any]],
        problem_type: str = "crack",
        has_singularity: bool = True,
        n_candidates: int = 3,
    ) -> List[DebateResult]:
        """Run Rounds 1-2 once, then generate n_candidates proposals in Round 3."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._run_debate(
                    history, current_config, dataset_size,
                    config_history, problem_type, has_singularity,
                    n_candidates=n_candidates,
                )
            )
        finally:
            loop.close()

    async def _run_debate(
        self,
        history: "TrainingHistory",
        current_config: "TransolverConfig",
        dataset_size: int,
        config_history: List[Dict[str, Any]],
        problem_type: str,
        has_singularity: bool,
        n_candidates: int = 1,
    ) -> List[DebateResult]:
        from piano.agents.base import AgentContext

        context = AgentContext()
        config_dict = current_config.to_dict()
        shared_log: List[str] = []

        # ── Pre-Round: Knowledge Retrieval ──────────────────────────────────
        if self.knowledge_retriever is not None:
            kb_entry = self.knowledge_retriever.retrieve(
                diagnosis="",  # filled after Round 1 Critic
                analyst_pattern="",
                pino_status="",
                config_history=config_history,
            )
            if kb_entry:
                context.knowledge_context.append({
                    "source": "knowledge_retriever",
                    "text": kb_entry.to_context_string(),
                })
                shared_log.append(kb_entry.to_context_string())
                logger.info(f"  KB entry injected: {kb_entry.method_name}")

        # ── Round 1: OBSERVATION ────────────────────────────────────────────
        logger.info("Debate Round 1: Observation")

        analyst_obs = await self.analyst.observe(context, history)
        analyst_text = analyst_obs.to_debate_message()
        shared_log.append(analyst_text)
        logger.info(f"  Analyst: {analyst_obs.pattern} (severity={analyst_obs.severity})")

        critic_obs_text = await self.critic.observe(context, history, config_dict)
        shared_log.append(critic_obs_text)

        critique = await self.critic.analyze_training(
            context=context,
            training_history=history,
            config=config_dict,
            previous_attempts=[
                {"summary": f"Round {i+1}: test_loss={h.get('result', '?')}, changes={h.get('changes', {})}"}
                for i, h in enumerate(config_history)
            ],
        )
        logger.info(f"  Critic: {critique.primary_issue.name} (severity={critique.severity})")

        # Re-retrieve KB with Critic diagnosis for better relevance
        if self.knowledge_retriever is not None and not context.knowledge_context:
            kb_entry = self.knowledge_retriever.retrieve(
                diagnosis=critique.primary_issue.name,
                analyst_pattern=analyst_obs.observation,
                pino_status=analyst_obs.pino_status,
                config_history=config_history,
            )
            if kb_entry:
                context.knowledge_context.append({
                    "source": "knowledge_retriever",
                    "text": kb_entry.to_context_string(),
                })
                shared_log.append(kb_entry.to_context_string())
                logger.info(f"  KB entry (post-Critic): {kb_entry.method_name}")

        r1_context = "\n\n".join(shared_log)

        # ── Round 2: ANALYSIS ───────────────────────────────────────────────
        logger.info("Debate Round 2: Analysis")

        arch_analysis = await self.architect.analyze(context, current_config, r1_context)
        shared_log.append(arch_analysis)
        logger.info(f"  Architect analysis: {arch_analysis[:120].replace(chr(10), ' ')}...")

        phys_analysis = await self.physicist.analyze(context, config_dict, history, r1_context)
        shared_log.append(phys_analysis)
        logger.info(f"  Physicist analysis: {phys_analysis[:120].replace(chr(10), ' ')}...")

        r12_context = "\n\n".join(shared_log)

        # ── Round 3: SYNTHESIS — n_candidates independent proposals ────────
        logger.info(f"Debate Round 3: Synthesis ({n_candidates} candidate(s))")

        candidates: List[DebateResult] = []
        for cand_idx in range(n_candidates):
            cand_log = shared_log[:]  # each candidate gets its own copy of the shared log

            arch_proposal = await self.architect.propose_config(
                context=context,
                current_config=current_config,
                critique=critique,
                dataset_size=dataset_size,
                previous_configs=config_history,
                debate_context=r12_context,
            )
            cand_log.append(f"[ARCHITECT — Round 3 Candidate {cand_idx+1}]\n{arch_proposal.raw_response}")
            logger.info(f"  Architect candidate {cand_idx+1}: {arch_proposal.changes}")

            phys_proposal = await self.physicist.propose_physics_config(
                context=context,
                current_config=config_dict,
                critique=critique,
                training_history=history,
                dataset_size=dataset_size,
                problem_type=problem_type,
                has_singularity=has_singularity,
                previous_configs=config_history,
                debate_context=r12_context,
            )
            cand_log.append(f"[PHYSICIST — Round 3 Candidate {cand_idx+1}]\n{phys_proposal.raw_response}")
            logger.info(f"  Physicist candidate {cand_idx+1}: {phys_proposal.changes}")

            # ── Round 4: FINALIZATION — validate this candidate ─────────────
            logger.info(f"Debate Round 4: Finalization (candidate {cand_idx+1})")
            arch_summary = (
                f"Architecture changes: {arch_proposal.changes}\n"
                f"Reasoning: {arch_proposal.reasoning[:300]}"
            )
            phys_summary = (
                f"Physics changes: {phys_proposal.changes}\n"
                f"Diagnosis: {phys_proposal.physics_diagnosis[:300]}"
            )
            validation_text = await self.critic.validate_proposals(
                arch_summary=arch_summary,
                phys_summary=phys_summary,
                debate_context="\n\n".join(cand_log),
            )
            logger.info(f"  Validation {cand_idx+1}: {validation_text[:120].replace(chr(10), ' ')}")

            candidates.append(DebateResult(
                arch_proposal=arch_proposal,
                physics_changes=phys_proposal.changes,
                debate_log=cand_log,
                validation_text=validation_text,
            ))

        return candidates
