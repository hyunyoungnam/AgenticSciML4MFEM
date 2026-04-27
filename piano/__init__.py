"""
PIANO: Physics-Informed Agentic Neural Operator

A self-improving surrogate framework for computational mechanics that combines:
- Transolver neural operator for learning FEM field predictions
- PINO loss for physics-informed training (equilibrium + energy norm)
- 3-agent HPO system for autonomous hyperparameter optimization:
  - CriticAgent: Diagnoses training issues
  - ArchitectAgent: Proposes architecture/optimizer changes
  - PhysicistAgent: Proposes physics loss configuration

Example (Agentic Training):
    >>> from piano.surrogate.agentic_trainer import (
    ...     AgenticSurrogateTrainer, AgenticTrainingConfig
    ... )
    >>> config = AgenticTrainingConfig(
    ...     max_hpo_rounds=3,
    ...     use_physicist=True,
    ...     problem_type="crack",
    ... )
    >>> trainer = AgenticSurrogateTrainer(config, llm_provider)
    >>> result = trainer.train(params, coords, outputs)
"""

__version__ = "0.3.0"
__author__ = "H.-Y. Nam, Q. Jiang"

# Core API - MFEM mesh management
from piano.mesh.base import MeshManager
from piano.mesh.mfem_manager import MFEMManager

# Solver API
from piano.solvers.base import (
    SolverInterface,
    PhysicsType,
    PhysicsConfig,
    MaterialProperties,
    SolverResult,
)
from piano.solvers.mfem_solver import MFEMSolver

# Evaluation API
from piano.evaluation.pipeline import EvaluationPipeline, EvaluationResult

# Adaptive Learning API
from piano.orchestration.adaptive import (
    AdaptiveOrchestrator,
    AdaptiveConfig,
    AdaptiveResult,
)

# Surrogate API
from piano.surrogate.base import TransolverConfig
from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
from piano.surrogate.agentic_trainer import (
    AgenticSurrogateTrainer,
    AgenticTrainingConfig,
    AgenticTrainingResult,
)

# Dataset API
from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig

# Geometry API
from piano.geometry import (
    CrackGeometry,
    EdgeCrack,
    CenterCrack,
    CrackMeshGenerator,
    generate_crack_mesh,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes - Mesh management
    "MeshManager",
    "MFEMManager",
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
    # Adaptive Learning
    "AdaptiveOrchestrator",
    "AdaptiveConfig",
    "AdaptiveResult",
    # Surrogate
    "TransolverConfig",
    "SurrogateTrainer",
    "TrainingConfig",
    "AgenticSurrogateTrainer",
    "AgenticTrainingConfig",
    "AgenticTrainingResult",
    # Dataset
    "FEMDataset",
    "FEMSample",
    "DatasetConfig",
    # Geometry
    "CrackGeometry",
    "EdgeCrack",
    "CenterCrack",
    "CrackMeshGenerator",
    "generate_crack_mesh",
]
