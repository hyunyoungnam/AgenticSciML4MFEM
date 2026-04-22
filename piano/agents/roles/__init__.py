"""
Agent role implementations for adaptive surrogate-guided learning.
"""

# Hyperparameter optimization agents (import first - no dependencies on broken modules)
from piano.agents.roles.hyperparameter_critic import (
    HyperparameterCriticAgent,
    CritiqueResult,
    TrainingHistory,
    TrainingIssue,
)
from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal

# Adaptive learning agents
from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal

# Core agents
from piano.agents.roles.evaluator import EvaluatorAgent

# NOTE: Some agents have import issues (missing proposer module)
# They are imported lazily to avoid breaking the module
try:
    from piano.agents.roles.engineer import EngineerAgent
except ImportError:
    EngineerAgent = None  # type: ignore

try:
    from piano.agents.roles.debugger import DebuggerAgent
except ImportError:
    DebuggerAgent = None  # type: ignore

try:
    from piano.agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
except ImportError:
    ClaudeCodeEngineer = None  # type: ignore
    ClaudeCodeEngineerConfig = None  # type: ignore

try:
    from piano.agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig
except ImportError:
    ClaudeCodeDebugger = None  # type: ignore
    ClaudeCodeDebuggerConfig = None  # type: ignore

__all__ = [
    # Hyperparameter optimization agents
    "HyperparameterCriticAgent",
    "CritiqueResult",
    "TrainingHistory",
    "TrainingIssue",
    "ArchitectAgent",
    "ArchitectureProposal",
    # Adaptive learning agents
    "AdaptiveProposerAgent",
    "AdaptiveProposal",
    # Core agents
    "EvaluatorAgent",
    "EngineerAgent",
    "DebuggerAgent",
    # Claude Code-powered agents
    "ClaudeCodeEngineer",
    "ClaudeCodeEngineerConfig",
    "ClaudeCodeDebugger",
    "ClaudeCodeDebuggerConfig",
]
