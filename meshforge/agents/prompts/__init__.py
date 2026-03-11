"""
Agent prompt templates.

Contains structured prompts for each agent role in the system.
"""

from meshforge.agents.prompts.evaluator import EVALUATOR_PROMPTS
from meshforge.agents.prompts.proposer import PROPOSER_PROMPTS
from meshforge.agents.prompts.critic import CRITIC_PROMPTS
from meshforge.agents.prompts.engineer import ENGINEER_PROMPTS
from meshforge.agents.prompts.debugger import DEBUGGER_PROMPTS

__all__ = [
    "EVALUATOR_PROMPTS",
    "PROPOSER_PROMPTS",
    "CRITIC_PROMPTS",
    "ENGINEER_PROMPTS",
    "DEBUGGER_PROMPTS",
]
