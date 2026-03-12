"""
meshforge: Automated MFEM mesh generation for AI/ML training datasets.

This library provides multi-agent AI systems to automatically generate diverse,
validated MFEM mesh files through intelligent mesh morphing. It eliminates
the tedious manual work of creating FEA datasets for machine learning research.

Key Features:
- Adaptive learning: Surrogate-guided data generation with DeepONet
- MFEM integration: Native support for MFEM mesh format
- FEM simulation: Built-in solvers for elasticity and heat transfer

Example (Adaptive Learning):
    >>> from meshforge import AdaptiveOrchestrator, AdaptiveConfig
    >>> config = AdaptiveConfig(
    ...     base_mesh_path="model.mesh",
    ...     output_dir="./output",
    ...     parameter_bounds={"delta_R": (-0.5, 0.5)},
    ... )
    >>> orchestrator = AdaptiveOrchestrator(config)
    >>> result = orchestrator.run()

Example (Basic Morphing):
    >>> from meshforge import MFEMManager, apply_morphing
    >>> manager = MFEMManager("model.mesh")
    >>> apply_morphing(manager, config_path="morphing.md", delta_r=0.5)
    >>> manager.save("output.mesh")
"""

__version__ = "0.2.0"
__author__ = "H.-Y. Nam, Q. Jiang"

# Core API - MFEM mesh management
from meshforge.mesh.base import MeshManager
from meshforge.mesh.mfem_manager import MFEMManager
from meshforge.morphing import run_morphing as apply_morphing, MorphingContext
from meshforge.schema import HeavyData, LightData, Nodes, Elements

# Solver API
from meshforge.solvers.base import (
    SolverInterface,
    PhysicsType,
    PhysicsConfig,
    MaterialProperties,
    SolverResult,
)
from meshforge.solvers.mfem_solver import MFEMSolver

# Evaluation API
from meshforge.evaluation.pipeline import EvaluationPipeline, EvaluationResult

# Adaptive Learning API (new)
from meshforge.orchestration.adaptive import (
    AdaptiveOrchestrator,
    AdaptiveConfig,
    AdaptiveResult,
)

# Dataset API (new)
from meshforge.data.dataset import FEMDataset, FEMSample, DatasetConfig

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes - Mesh management
    "MeshManager",
    "MFEMManager",
    "MorphingContext",
    "HeavyData",
    "LightData",
    "Nodes",
    "Elements",
    # Solver classes
    "SolverInterface",
    "MFEMSolver",
    "PhysicsType",
    "PhysicsConfig",
    "MaterialProperties",
    "SolverResult",
    # Evaluation
    "EvaluationPipeline",
    "EvaluationResult",
    # Adaptive Learning (new)
    "AdaptiveOrchestrator",
    "AdaptiveConfig",
    "AdaptiveResult",
    # Dataset (new)
    "FEMDataset",
    "FEMSample",
    "DatasetConfig",
    # Functions
    "apply_morphing",
]
