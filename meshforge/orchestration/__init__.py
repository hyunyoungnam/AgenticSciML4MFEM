"""
Main workflow orchestration for AgenticSciML.

Coordinates the four-phase workflow: Analysis, Knowledge Funnel,
Proposer-Critic Debate, and Engineer-Debugger execution.
"""

from meshforge.orchestration.orchestrator import AgenticOrchestrator
from meshforge.orchestration.phases import (
    Phase1AnalysisController,
    Phase2KnowledgeController,
    Phase3DebateController,
    Phase4ExecutionController,
)

__all__ = [
    "AgenticOrchestrator",
    "Phase1AnalysisController",
    "Phase2KnowledgeController",
    "Phase3DebateController",
    "Phase4ExecutionController",
]
