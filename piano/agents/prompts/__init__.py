"""
Agent prompt templates.

Contains structured prompts for each agent role in the system.
"""

from piano.agents.prompts.evaluator import EVALUATOR_PROMPTS
from piano.agents.prompts.engineer import ENGINEER_PROMPTS
from piano.agents.prompts.debugger import DEBUGGER_PROMPTS

__all__ = [
    "EVALUATOR_PROMPTS",
    "ENGINEER_PROMPTS",
    "DEBUGGER_PROMPTS",
]
