"""
Agent role implementations for agentic SciML.

Core agents for the HPO loop:
- HyperparameterCriticAgent: Diagnoses training issues
- ArchitectAgent: Proposes architecture/optimizer changes
- PhysicistAgent: Proposes physics loss configuration changes

Additional agents:
- AdaptiveProposerAgent: For active learning sample selection
"""

# Hyperparameter Critic Agent
from piano.agents.roles.hyperparameter_critic import (
    HyperparameterCriticAgent,
    CritiqueResult,
    TrainingHistory,
    TrainingIssue,
)

# Architect Agent
from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal

# Physicist Agent
from piano.agents.roles.physicist import PhysicistAgent, PhysicsProposal, PhysicsIssue

# Adaptive Proposer Agent (for active learning)
from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal

__all__ = [
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
]
