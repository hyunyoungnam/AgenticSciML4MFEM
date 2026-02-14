"""
Evolutionary tree search for solution exploration.

Manages solution populations, parent selection, and
generation-based evolution of Abaqus model variants.
"""

from evolution.solution import Solution, SolutionGenome, SolutionMetrics, SolutionStatus
from evolution.tree import SolutionTree
from evolution.selection import ParentSelector, TournamentSelector, EnsembleSelector

__all__ = [
    "Solution",
    "SolutionGenome",
    "SolutionMetrics",
    "SolutionStatus",
    "SolutionTree",
    "ParentSelector",
    "TournamentSelector",
    "EnsembleSelector",
]
