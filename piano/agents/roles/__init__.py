"""
Agent role implementations for adaptive surrogate-guided learning.
"""

# Adaptive learning agents
from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal

# Core agents
from piano.agents.roles.evaluator import EvaluatorAgent
from piano.agents.roles.engineer import EngineerAgent
from piano.agents.roles.debugger import DebuggerAgent

# Claude Code-powered agents
from piano.agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
from piano.agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig

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
