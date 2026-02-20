"""
Agent role implementations.

Each role represents a specialized agent with specific responsibilities
in the multi-agent collaboration system.

Includes both hand-coded agents and Claude Code-powered agents.
"""

from agents.roles.evaluator import EvaluatorAgent
from agents.roles.proposer import ProposerAgent
from agents.roles.critic import CriticAgent
from agents.roles.engineer import EngineerAgent
from agents.roles.debugger import DebuggerAgent
from agents.roles.result_analyst import ResultAnalystAgent
from agents.roles.retriever import RetrieverAgent
from agents.roles.selector import SelectorAgent

# Claude Code-powered agents
from agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
from agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig

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
