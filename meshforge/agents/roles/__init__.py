"""
Agent role implementations.

Each role represents a specialized agent with specific responsibilities
in the multi-agent collaboration system.

Includes both hand-coded agents and Claude Code-powered agents.
"""

from meshforge.agents.roles.evaluator import EvaluatorAgent
from meshforge.agents.roles.proposer import ProposerAgent
from meshforge.agents.roles.critic import CriticAgent
from meshforge.agents.roles.engineer import EngineerAgent
from meshforge.agents.roles.debugger import DebuggerAgent
from meshforge.agents.roles.result_analyst import ResultAnalystAgent
from meshforge.agents.roles.retriever import RetrieverAgent
from meshforge.agents.roles.selector import SelectorAgent

# Claude Code-powered agents
from meshforge.agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
from meshforge.agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig

__all__ = [
    # Hand-coded agents
    "EvaluatorAgent",
    "ProposerAgent",
    "CriticAgent",
    "EngineerAgent",
    "DebuggerAgent",
    "ResultAnalystAgent",
    "RetrieverAgent",
    "SelectorAgent",
    # Claude Code-powered agents
    "ClaudeCodeEngineer",
    "ClaudeCodeEngineerConfig",
    "ClaudeCodeDebugger",
    "ClaudeCodeDebuggerConfig",
]
