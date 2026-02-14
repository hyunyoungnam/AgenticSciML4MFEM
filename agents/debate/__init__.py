"""
Multi-round debate orchestration.

Implements structured debate between Proposer and Critic agents
for quality-controlled mutation proposals.
"""

from agents.debate.controller import DebateController, DebateRound, DebateResult

__all__ = [
    "DebateController",
    "DebateRound",
    "DebateResult",
]
