"""
Agent role implementations for agentic SciML.

Core agents for the HPO debate loop:
- ResultAnalystAgent: Observes training curves (Round 1 — no proposals)
- HyperparameterCriticAgent: Diagnoses training issues + validates proposals (Rounds 1 & 4)
- ArchitectAgent: Analyzes causes + proposes architecture/optimizer changes (Rounds 2 & 3)
- PhysicistAgent: Analyzes physics + proposes loss configuration changes (Rounds 2 & 3)
- EngineerAgent: Implements code-level changes via Claude Code CLI
- DebuggerAgent: Diagnoses code failures and provides targeted fix descriptions

Knowledge and data agents:
- KnowledgeRetrieverAgent: Surfaces relevant KB entries before each debate round
- DataAnalystAgent: Pre-training dataset analysis (Phase 0)

Candidate selection:
- SelectorEnsembleAgent: 3-LLM majority vote for candidate selection (replaces brief-training)

Active learning agents:
- AdaptiveProposerAgent: LLM-based sample proposal targeting weak regions
- MeshStrategyAgent: r/h-refinement strategy for MFEM meshes
- BudgetAgent: Decides when to collect more data vs switch to HPO vs stop
"""

# Result Analyst Agent (Round 1: observation only)
from piano.agents.roles.result_analyst import ResultAnalystAgent, AnalystObservation

# Hyperparameter Critic Agent (Rounds 1 & 4)
from piano.agents.roles.hyperparameter_critic import (
    HyperparameterCriticAgent,
    CritiqueResult,
    TrainingHistory,
    TrainingIssue,
)

# Architect Agent (Rounds 2 & 3)
from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal

# Physicist Agent (Rounds 2 & 3)
from piano.agents.roles.physicist import PhysicistAgent, PhysicsProposal, PhysicsIssue

# Adaptive Proposer Agent (for active learning)
from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal

# Engineer Agent (code-level changes via Claude Code CLI)
from piano.agents.roles.engineer import EngineerAgent, EngineerResult

# Debugger Agent (diagnoses code failures for EngineerAgent)
from piano.agents.roles.debugger import DebuggerAgent, DebugResult

# Knowledge Retriever Agent + KB
from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent, KBEntry

# Data Analyst Agent (pre-training dataset analysis)
from piano.agents.roles.data_analyst import DataAnalystAgent, DataAnalysis

# Selector Ensemble Agent (3-LLM voting for candidate selection)
from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent, SelectionResult, VoteResult

# Mesh Strategy Agent (r/h-refinement decisions for MFEM)
from piano.agents.roles.mesh_strategy import MeshStrategyAgent, MeshStrategyDecision

# Budget Agent (active learning stopping criterion)
from piano.agents.roles.budget import BudgetAgent, BudgetDecision

__all__ = [
    # Result Analyst
    "ResultAnalystAgent",
    "AnalystObservation",
    # Hyperparameter Critic
    "HyperparameterCriticAgent",
    "CritiqueResult",
    "TrainingHistory",
    "TrainingIssue",
    # Architect
    "ArchitectAgent",
    "ArchitectureProposal",
    # Physicist
    "PhysicistAgent",
    "PhysicsProposal",
    "PhysicsIssue",
    # Adaptive Proposer
    "AdaptiveProposerAgent",
    "AdaptiveProposal",
    # Engineer
    "EngineerAgent",
    "EngineerResult",
    # Debugger
    "DebuggerAgent",
    "DebugResult",
    # Knowledge Retriever
    "KnowledgeRetrieverAgent",
    "KBEntry",
    # Data Analyst
    "DataAnalystAgent",
    "DataAnalysis",
    # Selector Ensemble
    "SelectorEnsembleAgent",
    "SelectionResult",
    "VoteResult",
    # Mesh Strategy
    "MeshStrategyAgent",
    "MeshStrategyDecision",
    # Budget
    "BudgetAgent",
    "BudgetDecision",
]
