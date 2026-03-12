"""
Agent role implementations for adaptive surrogate-guided learning.
"""

# Adaptive learning agents
from meshforge.agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal

# Core agents
from meshforge.agents.roles.evaluator import EvaluatorAgent
from meshforge.agents.roles.engineer import EngineerAgent
from meshforge.agents.roles.debugger import DebuggerAgent

# Claude Code-powered agents
from meshforge.agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
from meshforge.agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig

__all__ = [
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
