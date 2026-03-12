"""
Knowledge management for AgenticSciML.

Note: Static FEA knowledge base is deprecated. Modern LLMs like Claude
and GPT-4 have comprehensive FEA domain knowledge built-in.
"""

from meshforge.knowledge.base import KnowledgeEntry, KnowledgeBase

__all__ = [
    "KnowledgeEntry",
    "KnowledgeBase",
]
