"""
Workflow orchestration for adaptive surrogate model training.

Provides surrogate-guided adaptive learning for efficient FEM data generation.
"""

from piano.orchestration.adaptive import (
    AdaptiveConfig,
    AdaptiveOrchestrator,
    AdaptiveResult,
)
from piano.orchestration.debate import DebateOrchestrator, DebateResult

__all__ = [
    "AdaptiveConfig",
    "AdaptiveOrchestrator",
    "AdaptiveResult",
    "DebateOrchestrator",
    "DebateResult",
]
