"""
Evolutionary tree search for solution exploration.

Manages solution populations, parent selection, and
generation-based evolution of Abaqus model variants.
"""

from meshforge.evolution.solution import Solution, SolutionGenome, SolutionMetrics, SolutionStatus
from meshforge.evolution.tree import SolutionTree
from meshforge.evolution.selection import ParentSelector, TournamentSelector, EnsembleSelector

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
