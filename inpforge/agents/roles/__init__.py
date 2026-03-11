"""
Agent role implementations.

Each role represents a specialized agent with specific responsibilities
in the multi-agent collaboration system.

Includes both hand-coded agents and Claude Code-powered agents.
"""

from inpforge.agents.roles.evaluator import EvaluatorAgent
from inpforge.agents.roles.proposer import ProposerAgent
from inpforge.agents.roles.critic import CriticAgent
from inpforge.agents.roles.engineer import EngineerAgent
from inpforge.agents.roles.debugger import DebuggerAgent
from inpforge.agents.roles.result_analyst import ResultAnalystAgent
from inpforge.agents.roles.retriever import RetrieverAgent
from inpforge.agents.roles.selector import SelectorAgent

# Claude Code-powered agents
from inpforge.agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
from inpforge.agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig

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
