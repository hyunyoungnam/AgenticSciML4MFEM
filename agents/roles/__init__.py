"""
Agent role implementations.

Each role represents a specialized agent with specific responsibilities
in the multi-agent collaboration system.
"""

from agents.roles.evaluator import EvaluatorAgent
from agents.roles.proposer import ProposerAgent
from agents.roles.critic import CriticAgent
from agents.roles.engineer import EngineerAgent
from agents.roles.debugger import DebuggerAgent
from agents.roles.result_analyst import ResultAnalystAgent
from agents.roles.retriever import RetrieverAgent
from agents.roles.selector import SelectorAgent

__all__ = [
    "EvaluatorAgent",
    "ProposerAgent",
    "CriticAgent",
    "EngineerAgent",
    "DebuggerAgent",
    "ResultAnalystAgent",
    "RetrieverAgent",
    "SelectorAgent",
]
