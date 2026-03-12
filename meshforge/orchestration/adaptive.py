"""
Adaptive learning orchestrator.

Implements the adaptive learning loop:
1. Generate initial FEM simulations
2. Train surrogate model (DeepONet)
3. Identify weak regions (high error/uncertainty)
4. Generate targeted new simulations
5. Repeat until convergence
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.dataset import DatasetConfig, FEMDataset, FEMSample
from ..data.loader import SimulationResultLoader
from ..mesh.mfem_manager import MFEMManager
from ..morphing import MorphingEngine
from ..solvers.base import (
    BoundaryCondition,
    BoundaryConditionType,
    MaterialProperties,
    PhysicsConfig,
    PhysicsType,
)
from ..solvers.mfem_solver import MFEMSolver
from ..surrogate.base import SurrogateConfig
from ..surrogate.deeponet import DeepONetEnsemble
from ..surrogate.evaluator import SurrogateEvaluator, UncertaintyAnalysis, WeakRegion
from ..surrogate.trainer import SurrogateTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive learning.

    Attributes:
        base_mesh_path: Path to base mesh file
        output_dir: Directory for outputs
        parameter_names: Names of varying parameters
        parameter_bounds: Bounds for each parameter
        initial_samples: Number of initial samples to generate
        samples_per_iteration: New samples per iteration
        max_iterations: Maximum adaptive iterations
        convergence_threshold: Error threshold for convergence
        surrogate_config: Configuration for surrogate model
        physics_config: Physics configuration for simulations
    """
    base_mesh_path: Path
    output_dir: Path
    parameter_names: List[str] = field(default_factory=lambda: ["delta_R"])
    parameter_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"delta_R": (-0.5, 0.5)}
    )
    initial_samples: int = 20
    samples_per_iteration: int = 10
    max_iterations: int = 10
    convergence_threshold: float = 0.05
    surrogate_config: SurrogateConfig = field(default_factory=SurrogateConfig)
    physics_config: Optional[PhysicsConfig] = None
    use_ensemble: bool = True
    n_ensemble: int = 5
    random_seed: int = 42

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.base_mesh_path = Path(self.base_mesh_path)


@dataclass
class AdaptiveResult:
    """
    Result of adaptive learning run.

    Attributes:
        success: Whether learning converged
        n_iterations: Number of iterations completed
        final_error: Final surrogate error
        total_samples: Total samples generated
        dataset_path: Path to final dataset
        surrogate_path: Path to trained surrogate
        history: Training history per iteration
    """
    success: bool
    n_iterations: int = 0
    final_error: float = float('inf')
    total_samples: int = 0
    dataset_path: Optional[Path] = None
    surrogate_path: Optional[Path] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


class AdaptiveOrchestrator:
    """
    Orchestrates adaptive learning for surrogate model training.

    Workflow:
    1. Generate initial samples with uniform parameter sampling
    2. Run FEM simulations to get ground truth
    3. Train surrogate model on current dataset
    4. Evaluate surrogate, identify weak regions
    5. Generate new samples in weak regions
    6. Repeat until convergence or max iterations
    """

    def __init__(self, config: AdaptiveConfig):
        """
        Initialize orchestrator.

        Args:
            config: Adaptive learning configuration
        """
        self.config = config
        self.dataset: Optional[FEMDataset] = None
        self.surrogate: Optional[DeepONetEnsemble] = None
        self.trainer: Optional[SurrogateTrainer] = None
        self.evaluator: Optional[SurrogateEvaluator] = None

        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir = self.config.output_dir / "meshes"
        self.meshes_dir.mkdir(exist_ok=True)
        self.dataset_dir = self.config.output_dir / "dataset"
        self.dataset_dir.mkdir(exist_ok=True)
        self.surrogate_dir = self.config.output_dir / "surrogate"
        self.surrogate_dir.mkdir(exist_ok=True)

        # Initialize dataset
        dataset_config = DatasetConfig(
            name="adaptive_fem_dataset",
            parameter_names=config.parameter_names,
            parameter_bounds=config.parameter_bounds,
            output_fields=["displacement", "von_mises"],
            coordinate_dim=2,  # Will be updated from mesh
            storage_dir=self.dataset_dir,
        )
        self.dataset = FEMDataset(dataset_config)

        # Setup RNG
        np.random.seed(config.random_seed)

        logger.info(f"Initialized AdaptiveOrchestrator with config: {config}")

    def run(
        self,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> AdaptiveResult:
        """
        Run the adaptive learning loop.

        Args:
            callback: Optional callback(iteration, metrics) for progress reporting

        Returns:
            AdaptiveResult with final metrics and paths
        """
        history = []

        try:
            # Phase 1: Generate initial samples
            logger.info("Phase 1: Generating initial samples")
            initial_params = self._generate_initial_parameters()
            self._run_simulations(initial_params)

            # Main adaptive loop
            for iteration in range(self.config.max_iterations):
                logger.info(f"=== Iteration {iteration + 1}/{self.config.max_iterations} ===")

                # Phase 2: Train surrogate
                logger.info("Training surrogate model")
                train_result = self._train_surrogate()

                if not train_result.success:
                    logger.error(f"Surrogate training failed: {train_result.error_message}")
                    return AdaptiveResult(
                        success=False,
                        error_message=train_result.error_message,
                        n_iterations=iteration + 1,
                        total_samples=len(self.dataset),
                    )

                # Phase 3: Evaluate and identify weak regions
                logger.info("Evaluating surrogate model")
                analysis = self._evaluate_surrogate()

                # Record iteration metrics
                iteration_metrics = {
                    "iteration": iteration + 1,
                    "n_samples": len(self.dataset),
                    "train_loss": train_result.train_loss,
                    "test_loss": train_result.test_loss,
                    "n_weak_regions": len(analysis.weak_regions),
                    "overall_uncertainty": analysis.overall_uncertainty,
                    "test_error": train_result.metrics.get("rmse", float('inf')),
                }
                history.append(iteration_metrics)

                if callback:
                    callback(iteration + 1, iteration_metrics)

                logger.info(f"Iteration metrics: {iteration_metrics}")

                # Check convergence
                test_error = train_result.metrics.get("relative_l2", float('inf'))
                if test_error < self.config.convergence_threshold:
                    logger.info(f"Converged! Error {test_error:.4f} < {self.config.convergence_threshold}")
                    break

                # Phase 4: Generate new samples in weak regions
                if analysis.weak_regions:
                    logger.info(f"Generating {self.config.samples_per_iteration} new samples in weak regions")
                    new_params = self._suggest_new_parameters(analysis)
                    self._run_simulations(new_params)
                else:
                    logger.info("No weak regions identified, generating random samples")
                    random_params = self._generate_random_parameters(
                        self.config.samples_per_iteration
                    )
                    self._run_simulations(random_params)

            # Save final dataset and surrogate
            dataset_path = self.dataset.save()
            surrogate_path = self.surrogate_dir / "final_model"
            if self.surrogate:
                self.surrogate.save(surrogate_path)

            return AdaptiveResult(
                success=True,
                n_iterations=len(history),
                final_error=history[-1]["test_error"] if history else float('inf'),
                total_samples=len(self.dataset),
                dataset_path=dataset_path,
                surrogate_path=surrogate_path,
                history=history,
            )

        except Exception as e:
            logger.exception("Adaptive learning failed")
            return AdaptiveResult(
                success=False,
                error_message=str(e),
                n_iterations=len(history),
                total_samples=len(self.dataset) if self.dataset else 0,
                history=history,
            )

    def _generate_initial_parameters(self) -> List[Dict[str, float]]:
        """Generate initial parameter samples using Latin Hypercube Sampling."""
        n_samples = self.config.initial_samples
        n_params = len(self.config.parameter_names)

        # Latin Hypercube Sampling for better coverage
        samples = []
        for i, name in enumerate(self.config.parameter_names):
            min_val, max_val = self.config.parameter_bounds[name]
            # Create stratified samples
            edges = np.linspace(min_val, max_val, n_samples + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            samples.append(points)

        samples = np.array(samples).T  # Shape: (n_samples, n_params)

        # Convert to list of dicts
        param_list = []
        for i in range(n_samples):
            params = {
                name: float(samples[i, j])
                for j, name in enumerate(self.config.parameter_names)
            }
            param_list.append(params)

        return param_list

    def _generate_random_parameters(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate random parameter samples."""
        param_list = []
        for _ in range(n_samples):
            params = {}
            for name in self.config.parameter_names:
                min_val, max_val = self.config.parameter_bounds[name]
                params[name] = np.random.uniform(min_val, max_val)
            param_list.append(params)
        return param_list

    def _suggest_new_parameters(
        self,
        analysis: UncertaintyAnalysis
    ) -> List[Dict[str, float]]:
        """Suggest new parameters based on uncertainty analysis."""
        if self.evaluator is None:
            return self._generate_random_parameters(self.config.samples_per_iteration)

        return self.evaluator.suggest_samples(
            analysis,
            budget=self.config.samples_per_iteration
        )

    def _run_simulations(self, param_list: List[Dict[str, float]]) -> None:
        """
        Run FEM simulations for given parameters.

        Args:
            param_list: List of parameter dictionaries
        """
        result_loader = SimulationResultLoader()

        for i, params in enumerate(param_list):
            logger.info(f"Running simulation {i+1}/{len(param_list)}: {params}")

            try:
                sample = self._run_single_simulation(params, result_loader)
                self.dataset.add_sample(sample)
                logger.info(f"Sample {sample.sample_id} added (valid={sample.is_valid})")

            except Exception as e:
                logger.warning(f"Simulation failed for params {params}: {e}")
                # Add failed sample
                sample = FEMSample(
                    sample_id=f"failed_{i}_{np.random.randint(10000)}",
                    parameters=params,
                    coordinates=np.array([]),
                    is_valid=False,
                    metadata={"error": str(e)},
                )
                self.dataset.add_sample(sample)

    def _run_single_simulation(
        self,
        params: Dict[str, float],
        result_loader: SimulationResultLoader
    ) -> FEMSample:
        """
        Run a single FEM simulation.

        Args:
            params: Parameter dictionary
            result_loader: Loader for converting results

        Returns:
            FEMSample with simulation results
        """
        # Load base mesh
        mesh_manager = MFEMManager(self.config.base_mesh_path)

        # Apply morphing if delta_R is specified
        if "delta_R" in params:
            delta_R = params["delta_R"]
            morphed_coords = self._apply_morphing(mesh_manager, delta_R)
            mesh_manager.update_nodes(morphed_coords)

        # Setup physics
        physics = self.config.physics_config
        if physics is None:
            # Default physics
            physics = PhysicsConfig(
                physics_type=PhysicsType.LINEAR_ELASTICITY,
                material=MaterialProperties(
                    E=params.get("E", 200e9),
                    nu=params.get("nu", 0.3),
                ),
                boundary_conditions=[
                    BoundaryCondition(
                        bc_type=BoundaryConditionType.DISPLACEMENT,
                        boundary_id=1,
                        value=0.0,
                    ),
                    BoundaryCondition(
                        bc_type=BoundaryConditionType.TRACTION,
                        boundary_id=2,
                        value=np.array([0.0, -1e6]),
                    ),
                ],
            )

        # Run solver
        solver = MFEMSolver(order=1)
        solver.setup(mesh_manager, physics)

        sample_id = f"sim_{len(self.dataset)}_{np.random.randint(10000)}"
        output_dir = self.meshes_dir / sample_id
        output_dir.mkdir(exist_ok=True)

        result = solver.solve(output_dir)

        # Convert to FEMSample
        return result_loader.from_solver_result(
            solver_result=result,
            mesh_manager=mesh_manager,
            parameters=params,
            sample_id=sample_id,
        )

    def _apply_morphing(
        self,
        mesh_manager: MFEMManager,
        delta_R: float
    ) -> np.ndarray:
        """
        Apply mesh morphing for hole radius change.

        Args:
            mesh_manager: Mesh manager
            delta_R: Change in hole radius

        Returns:
            Morphed coordinates
        """
        coords = mesh_manager.get_nodes()

        # Simple radial morphing from center
        # Assumes a hole at origin - adjust as needed
        center = np.array([0.0, 0.0])
        if coords.shape[1] == 3:
            center = np.array([0.0, 0.0, 0.0])

        # Calculate distances from center
        distances = np.linalg.norm(coords - center, axis=1)

        # Apply morphing - nodes closer to center move more
        R0 = np.min(distances[distances > 0])  # Initial hole radius estimate
        R_outer = np.max(distances)

        morphed = coords.copy()
        for i in range(len(coords)):
            r = distances[i]
            if r > 0:
                # Linear interpolation of displacement
                factor = (R_outer - r) / (R_outer - R0) if R_outer > R0 else 0
                direction = (coords[i] - center) / r
                morphed[i] = coords[i] + delta_R * factor * direction

        return morphed

    def _train_surrogate(self) -> TrainingResult:
        """Train surrogate model on current dataset."""
        training_config = TrainingConfig(
            surrogate_config=self.config.surrogate_config,
            use_ensemble=self.config.use_ensemble,
            n_ensemble=self.config.n_ensemble,
            normalize_inputs=True,
            normalize_outputs=True,
            train_test_split=0.2,
            random_seed=self.config.random_seed,
            save_dir=self.surrogate_dir,
        )

        self.trainer = SurrogateTrainer(training_config)

        # Prepare training data
        try:
            parameters, coordinates, outputs = self.dataset.prepare_training_data(
                output_field="displacement",
                valid_only=True
            )
        except ValueError as e:
            return TrainingResult(success=False, error_message=str(e))

        # Train
        result = self.trainer.train(parameters, coordinates, outputs)

        if result.success:
            self.surrogate = self.trainer.model

        return result

    def _evaluate_surrogate(self) -> UncertaintyAnalysis:
        """Evaluate surrogate model and identify weak regions."""
        if self.surrogate is None:
            return UncertaintyAnalysis()

        # Initialize evaluator
        self.evaluator = SurrogateEvaluator(
            model=self.surrogate,
            parameter_names=self.config.parameter_names,
            parameter_bounds=self.config.parameter_bounds,
        )

        # Get validation data
        try:
            parameters, coordinates, outputs = self.dataset.prepare_training_data(
                output_field="displacement",
                valid_only=True
            )
        except ValueError:
            return UncertaintyAnalysis()

        # Analyze errors on existing data
        analysis = self.evaluator.analyze_errors(
            parameters=parameters,
            coordinates=coordinates,
            true_outputs=outputs,
            error_threshold=self.config.convergence_threshold,
        )

        return analysis

    def get_dataset(self) -> FEMDataset:
        """Get the current dataset."""
        return self.dataset

    def get_surrogate(self):
        """Get the trained surrogate model."""
        return self.surrogate
