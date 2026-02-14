"""
Agent prompt templates.

Contains structured prompts for each agent role in the system.
"""

from agents.prompts.evaluator import EVALUATOR_PROMPTS
from agents.prompts.proposer import PROPOSER_PROMPTS
from agents.prompts.critic import CRITIC_PROMPTS
from agents.prompts.engineer import ENGINEER_PROMPTS
from agents.prompts.debugger import DEBUGGER_PROMPTS

__all__ = [
    "EVALUATOR_PROMPTS",
    "PROPOSER_PROMPTS",
    "CRITIC_PROMPTS",
    "ENGINEER_PROMPTS",
    "DEBUGGER_PROMPTS",
]
