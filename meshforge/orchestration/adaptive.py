"""
Adaptive learning orchestrator.

Implements an autonomous active learning loop for efficient FEM dataset generation:
1. Generate initial FEM simulations (Latin Hypercube Sampling)
2. Train surrogate model (FNO/Transolver - implementation pending)
3. Evaluate surrogate and identify high-error/high-uncertainty regions
4. Select new samples using acquisition functions (informative sampling)
5. Repeat until convergence criteria are met

The key innovation is "informative sampling" - using acquisition functions
to prioritize samples that maximize information gain about the underlying
physics, rather than sampling uniformly.

Reference:
    Settles (2009): "Active Learning Literature Survey"
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..data.dataset import DatasetConfig, FEMDataset, FEMSample
from ..data.loader import SimulationResultLoader
from ..mesh.mfem_manager import MFEMManager
from ..morphing import (
    TMOPAdaptivity,
    AdaptivityConfig,
    is_tmop_available,
)
from ..solvers.base import (
    BoundaryCondition,
    BoundaryConditionType,
    MaterialProperties,
    PhysicsConfig,
    PhysicsType,
)
from ..solvers.mfem_solver import MFEMSolver
from ..surrogate.base import SurrogateConfig, SurrogateModel
from ..surrogate.evaluator import SurrogateEvaluator, UncertaintyAnalysis, WeakRegion
from ..surrogate.trainer import SurrogateTrainer, TrainingConfig, TrainingResult
from ..surrogate.acquisition import (
    AcquisitionFunction,
    AcquisitionType,
    get_acquisition_function,
)
from .metrics import ActiveLearningMetrics, ConvergenceMonitor

logger = logging.getLogger(__name__)


class StoppingCriterion(Enum):
    """Reasons for stopping the active learning loop."""
    CONVERGED = auto()  # Error below threshold
    PATIENCE_EXHAUSTED = auto()  # No improvement for N iterations
    BUDGET_EXHAUSTED = auto()  # Maximum samples reached
    MAX_ITERATIONS = auto()  # Maximum iterations reached
    LOW_UNCERTAINTY = auto()  # Uncertainty below threshold
    DIMINISHING_RETURNS = auto()  # Efficiency dropped below threshold
    USER_INTERRUPTED = auto()  # User requested stop


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive active learning.

    Essential parameters only - derived values computed at runtime.

    Attributes:
        base_mesh_path: Path to base mesh file
        output_dir: Directory for outputs
        parameter_bounds: Bounds for each parameter {name: (min, max)}
        initial_samples: Number of initial LHS samples
        max_samples: Hard budget limit on total samples
        convergence_threshold: Error threshold for convergence
        patience: Iterations without improvement before stopping
        n_ensemble: Number of ensemble models for uncertainty
        physics_config: Optional physics configuration for simulations
        acquisition_strategy: Acquisition function ("uncertainty", "ei", "qbc")
    """
    base_mesh_path: Path
    output_dir: Path
    parameter_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"delta_R": (-0.5, 0.5)}
    )
    initial_samples: int = 20
    max_samples: int = 200
    convergence_threshold: float = 0.05
    patience: int = 3
    n_ensemble: int = 5
    physics_config: Optional[PhysicsConfig] = None
    acquisition_strategy: str = "uncertainty"
    random_seed: int = 42

    # Derived at runtime
    @property
    def parameter_names(self) -> List[str]:
        """Derive parameter names from bounds keys."""
        return list(self.parameter_bounds.keys())

    @property
    def samples_per_iteration(self) -> int:
        """Base samples per iteration (dynamically adjusted)."""
        return 10

    @property
    def max_iterations(self) -> int:
        """Derive max iterations from budget."""
        return (self.max_samples - self.initial_samples) // self.samples_per_iteration + 1

    @property
    def surrogate_config(self) -> SurrogateConfig:
        """Default surrogate configuration."""
        return SurrogateConfig()

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.base_mesh_path = Path(self.base_mesh_path)


@dataclass
class AdaptiveResult:
    """
    Result of adaptive active learning run.

    Attributes:
        success: Whether learning converged successfully
        n_iterations: Number of iterations completed
        final_error: Final surrogate error
        total_samples: Total samples generated
        dataset_path: Path to final dataset
        surrogate_path: Path to trained surrogate
        history: Training history per iteration
        stopping_criterion: Why the loop stopped
        sample_efficiency: Error reduction per sample
        metrics_path: Path to saved metrics
    """
    success: bool
    n_iterations: int = 0
    final_error: float = float('inf')
    initial_error: float = float('inf')
    total_samples: int = 0
    dataset_path: Optional[Path] = None
    surrogate_path: Optional[Path] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    stopping_criterion: Optional[StoppingCriterion] = None
    sample_efficiency: float = 0.0
    error_reduction_percent: float = 0.0
    metrics_path: Optional[Path] = None


class AdaptiveOrchestrator:
    """
    Orchestrates autonomous active learning for surrogate model training.

    Implements an informative sampling strategy using acquisition functions
    to prioritize FEM simulations that maximize model improvement.

    Workflow:
    1. Generate initial samples with Latin Hypercube Sampling
    2. Run FEM simulations to get ground truth
    3. Train surrogate model (FNO/Transolver - implementation pending)
    4. Evaluate surrogate and compute acquisition scores
    5. Select new samples using acquisition function (informative sampling)
    6. Repeat until convergence criteria are met

    The key difference from uniform sampling is that new samples are
    selected to maximize information gain, focusing computational
    resources on regions where the model is weak.
    """

    def __init__(self, config: AdaptiveConfig):
        """
        Initialize orchestrator.

        Args:
            config: Adaptive learning configuration
        """
        self.config = config
        self.dataset: Optional[FEMDataset] = None
        self.surrogate: Optional[SurrogateModel] = None
        self.trainer: Optional[SurrogateTrainer] = None
        self.evaluator: Optional[SurrogateEvaluator] = None

        # Active learning components
        self.metrics = ActiveLearningMetrics()
        self.convergence_monitor = ConvergenceMonitor(
            target_error=config.convergence_threshold,
            patience=config.patience,
            min_improvement=0.01,  # Hardcoded sensible default
            max_samples=config.max_samples,
        )
        self._acquisition_fn: Optional[AcquisitionFunction] = None
        self._coordinates: Optional[np.ndarray] = None
        self._best_error: float = float('inf')
        self._no_improvement_count: int = 0

        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir = self.config.output_dir / "meshes"
        self.meshes_dir.mkdir(exist_ok=True)
        self.dataset_dir = self.config.output_dir / "dataset"
        self.dataset_dir.mkdir(exist_ok=True)
        self.surrogate_dir = self.config.output_dir / "surrogate"
        self.surrogate_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.config.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

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

        # Initialize acquisition function
        self._acquisition_fn = get_acquisition_function(config.acquisition_strategy)

        self._adaptivity = TMOPAdaptivity() if is_tmop_available() else None

        # Setup RNG
        np.random.seed(config.random_seed)

        logger.info(f"Initialized AdaptiveOrchestrator")
        logger.info(f"  Acquisition: {config.acquisition_strategy}")
        logger.info(f"  Convergence: {config.convergence_threshold}")
        logger.info(f"  Patience: {config.patience}")
        logger.info(f"  Max samples: {config.max_samples}")

    def run(
        self,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> AdaptiveResult:
        """
        Run the autonomous active learning loop.

        This method implements the core active learning algorithm:
        1. Initialize with diverse samples (LHS)
        2. Train surrogate and evaluate
        3. Use acquisition function to select informative samples
        4. Repeat until convergence criteria met

        Args:
            callback: Optional callback(iteration, metrics) for progress reporting

        Returns:
            AdaptiveResult with final metrics, paths, and efficiency statistics
        """
        history = []
        stopping_criterion = None
        initial_error = float('inf')

        try:
            # Phase 1: Generate initial samples using Latin Hypercube Sampling
            logger.info("="*60)
            logger.info("PHASE 1: Initial Sampling (Latin Hypercube)")
            logger.info("="*60)
            initial_params = self._generate_initial_parameters()
            self._run_simulations(initial_params)

            # Main active learning loop
            for iteration in range(self.config.max_iterations):
                logger.info("")
                logger.info("="*60)
                logger.info(f"ITERATION {iteration + 1}/{self.config.max_iterations}")
                logger.info("="*60)

                # Phase 2: Train surrogate
                logger.info("Phase 2: Training surrogate model...")
                train_result = self._train_surrogate()

                if not train_result.success:
                    logger.error(f"Surrogate training failed: {train_result.error_message}")
                    return AdaptiveResult(
                        success=False,
                        error_message=train_result.error_message,
                        n_iterations=iteration + 1,
                        total_samples=len(self.dataset),
                        stopping_criterion=StoppingCriterion.USER_INTERRUPTED,
                    )

                # Phase 3: Evaluate surrogate and compute uncertainty
                logger.info("Phase 3: Evaluating surrogate model...")
                analysis = self._evaluate_surrogate()
                uncertainty_stats = self._compute_uncertainty_stats()

                # Get current error metrics
                test_error = train_result.metrics.get("relative_l2", float('inf'))
                if iteration == 0:
                    initial_error = test_error

                # Log metrics
                iteration_metrics = {
                    "iteration": iteration + 1,
                    "n_samples": len(self.dataset),
                    "train_loss": train_result.train_loss,
                    "test_loss": train_result.test_loss,
                    "test_error": test_error,
                    "n_weak_regions": len(analysis.weak_regions),
                    "mean_uncertainty": uncertainty_stats.get("mean_uncertainty", 0),
                    "max_uncertainty": uncertainty_stats.get("max_uncertainty", 0),
                }
                history.append(iteration_metrics)

                # Update metrics tracker
                self.metrics.log_iteration(
                    iteration=iteration + 1,
                    n_samples_total=len(self.dataset),
                    n_samples_new=self.config.samples_per_iteration if iteration > 0 else self.config.initial_samples,
                    train_error=train_result.train_loss,
                    test_error=test_error,
                    mean_uncertainty=uncertainty_stats.get("mean_uncertainty", 0),
                    max_uncertainty=uncertainty_stats.get("max_uncertainty", 0),
                    n_weak_regions=len(analysis.weak_regions),
                )

                if callback:
                    callback(iteration + 1, iteration_metrics)

                logger.info(f"  Test error: {test_error:.6f}")
                logger.info(f"  Mean uncertainty: {uncertainty_stats.get('mean_uncertainty', 0):.6f}")
                logger.info(f"  Weak regions: {len(analysis.weak_regions)}")

                # Check convergence criteria
                stopping_criterion = self._check_convergence(
                    test_error=test_error,
                    n_samples=len(self.dataset),
                    n_weak_regions=len(analysis.weak_regions),
                    mean_uncertainty=uncertainty_stats.get("mean_uncertainty", 0),
                    iteration=iteration,
                )

                if stopping_criterion is not None:
                    logger.info(f"Stopping: {stopping_criterion.name}")
                    break

                # Phase 4: Select new samples using acquisition function
                logger.info("Phase 4: Selecting informative samples...")
                n_new_samples = self._compute_adaptive_budget(test_error)
                new_params = self._select_informative_samples(n_new_samples)

                logger.info(f"  Selected {len(new_params)} new samples using {self.config.acquisition_strategy}")

                # Phase 5: Run simulations for new samples
                logger.info("Phase 5: Running FEM simulations...")
                self._run_simulations(new_params)

                # Phase 6: Adapt base mesh toward high-uncertainty spatial regions
                if self._adaptivity is not None and self.surrogate is not None and new_params:
                    logger.info("Phase 6: Applying r-adaptivity to base mesh...")
                    node_uncertainty = self._get_node_uncertainty(new_params[0])
                    adapted_manager = MFEMManager(str(self.config.base_mesh_path))
                    adapted_coords, _ = self._apply_r_adaptivity(adapted_manager, node_uncertainty)
                    adapted_manager.update_nodes(adapted_coords)
                    adapted_manager.save(self.config.base_mesh_path)
                    self._coordinates = adapted_coords
                    logger.info(f"  Base mesh adapted ({len(adapted_coords)} nodes)")

            # Final stopping criterion if loop completed
            if stopping_criterion is None:
                stopping_criterion = StoppingCriterion.MAX_ITERATIONS

            # Save final artifacts
            logger.info("")
            logger.info("="*60)
            logger.info("SAVING RESULTS")
            logger.info("="*60)

            dataset_path = self.dataset.save()
            surrogate_path = self.surrogate_dir / "final_model"
            if self.surrogate:
                self.surrogate.save(surrogate_path)

            # Save metrics
            metrics_path = self.metrics_dir / "active_learning_metrics.json"
            self.metrics.save(metrics_path)

            # Compute final statistics
            final_error = history[-1]["test_error"] if history else float('inf')
            error_reduction = initial_error - final_error
            sample_efficiency = self.metrics.compute_efficiency()
            error_reduction_percent = (error_reduction / initial_error * 100) if initial_error > 0 else 0

            logger.info(f"Final error: {final_error:.6f}")
            logger.info(f"Error reduction: {error_reduction:.6f} ({error_reduction_percent:.1f}%)")
            logger.info(f"Sample efficiency: {sample_efficiency:.6f}")
            logger.info(f"Total samples: {len(self.dataset)}")
            logger.info(self.metrics.summary())

            return AdaptiveResult(
                success=True,
                n_iterations=len(history),
                final_error=final_error,
                initial_error=initial_error,
                total_samples=len(self.dataset),
                dataset_path=dataset_path,
                surrogate_path=surrogate_path,
                history=history,
                stopping_criterion=stopping_criterion,
                sample_efficiency=sample_efficiency,
                error_reduction_percent=error_reduction_percent,
                metrics_path=metrics_path,
            )

        except Exception as e:
            logger.exception("Active learning failed")
            return AdaptiveResult(
                success=False,
                error_message=str(e),
                n_iterations=len(history),
                total_samples=len(self.dataset) if self.dataset else 0,
                history=history,
                stopping_criterion=StoppingCriterion.USER_INTERRUPTED,
            )

    def _check_convergence(
        self,
        test_error: float,
        n_samples: int,
        n_weak_regions: int,
        mean_uncertainty: float,
        iteration: int
    ) -> Optional[StoppingCriterion]:
        """
        Check all convergence criteria.

        Returns StoppingCriterion if should stop, None otherwise.
        """
        min_improvement = 0.01  # Hardcoded sensible default

        # Check target error
        if test_error <= self.config.convergence_threshold:
            return StoppingCriterion.CONVERGED

        # Check sample budget
        if n_samples >= self.config.max_samples:
            return StoppingCriterion.BUDGET_EXHAUSTED

        # Check improvement (patience)
        improvement = self._best_error - test_error
        if improvement > min_improvement:
            self._best_error = test_error
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.config.patience:
            return StoppingCriterion.PATIENCE_EXHAUSTED

        # Check uncertainty (use convergence_threshold as uncertainty threshold)
        if mean_uncertainty < self.config.convergence_threshold:
            return StoppingCriterion.LOW_UNCERTAINTY

        # Check diminishing returns
        if self.metrics.detect_diminishing_returns(
            window=self.config.patience,
            threshold=min_improvement
        ):
            return StoppingCriterion.DIMINISHING_RETURNS

        return None

    def _compute_adaptive_budget(self, current_error: float) -> int:
        """
        Dynamically compute number of samples for next iteration.

        Uses more samples when error is high, fewer as we converge.
        """
        base_samples = self.config.samples_per_iteration

        # Scale budget based on how far from convergence
        error_ratio = current_error / self.config.convergence_threshold

        if error_ratio > 5:
            # Far from convergence - use more samples
            scale = 1.5
        elif error_ratio > 2:
            # Moderate distance - standard budget
            scale = 1.0
        elif error_ratio > 1:
            # Close to convergence - use fewer, more targeted samples
            scale = 0.7
        else:
            # Very close - minimal samples
            scale = 0.5

        budget = int(base_samples * scale)
        budget = max(3, min(budget, base_samples * 2))

        return budget

    def _select_informative_samples(
        self,
        n_samples: int
    ) -> List[Dict[str, float]]:
        """
        Select new samples using acquisition-based active learning.

        This is the core active learning step - using the acquisition
        function to identify the most informative parameter configurations.
        """
        if self.evaluator is None or self._coordinates is None:
            return self._generate_random_parameters(n_samples)

        # Get existing samples to avoid
        existing_samples = None
        try:
            params, _, _ = self.dataset.prepare_training_data(
                output_field="displacement",
                valid_only=True
            )
            existing_samples = params
        except ValueError:
            pass

        # Use acquisition-based selection (hardcoded sensible defaults)
        new_params = self.evaluator.suggest_samples_active(
            budget=n_samples,
            coordinates=self._coordinates,
            acquisition_type=self.config.acquisition_strategy,
            n_candidates=1000,
            diversity_weight=0.1,
            existing_samples=existing_samples,
        )

        return new_params

    def _compute_uncertainty_stats(self) -> Dict[str, float]:
        """Compute uncertainty statistics across parameter space."""
        if self.evaluator is None or self._coordinates is None:
            return {"mean_uncertainty": 0.0, "max_uncertainty": 0.0}

        return self.evaluator.estimate_remaining_uncertainty(
            coordinates=self._coordinates,
            n_probe_samples=100
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

    def _apply_r_adaptivity(
        self,
        mesh_manager: MFEMManager,
        error_field: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply r-adaptivity based on surrogate model error field.

        Uses TMOP to redistribute mesh nodes, clustering them in regions
        where the surrogate model has high prediction error.

        Args:
            mesh_manager: Mesh manager with current mesh
            error_field: (N,) array of error values at each node.
                        Higher values → nodes will cluster there.

        Returns:
            Tuple of (adapted coordinates, quality metrics dict)
        """
        if self._adaptivity is None:
            logger.warning("R-adaptivity not available, skipping mesh adaptation")
            return mesh_manager.get_nodes(), {}

        result = self._adaptivity.adapt(mesh_manager, error_field)

        if result.success:
            logger.info(
                f"R-adaptivity: {result.iterations} iterations, "
                f"quality {result.quality_before.get('min_quality', 0):.3f} → "
                f"{result.quality_after.get('min_quality', 0):.3f}"
            )
            return result.coords_adapted, result.quality_after
        else:
            logger.warning(f"R-adaptivity failed: {result.error_message}")
            return mesh_manager.get_nodes(), {}

    def adapt_mesh_to_error(
        self,
        error_field: np.ndarray,
        mesh_path: Optional[Path] = None,
    ) -> Tuple[MFEMManager, Dict[str, float]]:
        """
        Adapt mesh based on surrogate model error field.

        Public method to perform r-adaptivity on a mesh given an error field.
        Nodes are redistributed to cluster in high-error regions.

        Args:
            error_field: (N,) array of pointwise error values.
                        Typically from surrogate.compute_pointwise_error()
            mesh_path: Optional path to mesh file. If None, uses base mesh.

        Returns:
            Tuple of (adapted MFEMManager, quality metrics)

        Example:
            # Get error from surrogate
            coords = mesh_manager.get_nodes()
            predictions = surrogate.predict(params, coords)
            error_field = np.abs(predictions - ground_truth)

            # Adapt mesh
            adapted_manager, quality = orchestrator.adapt_mesh_to_error(error_field)
        """
        mesh_path = mesh_path or self.config.base_mesh_path
        manager = MFEMManager(str(mesh_path))

        coords, quality = self._apply_r_adaptivity(manager, error_field)

        return manager, quality

    def _get_node_uncertainty(self, params: Dict[str, float]) -> np.ndarray:
        """
        Compute scalar uncertainty at each mesh node for one parameter config.

        Collapses ensemble uncertainty (N_nodes, output_dim) → (N_nodes,) by
        averaging across output dimensions.  Used to drive r-adaptivity.

        Args:
            params: Parameter dictionary for the configuration to probe

        Returns:
            (N_nodes,) array of uncertainty values
        """
        param_array = np.array([[params[name] for name in self.config.parameter_names]])
        result = self.surrogate.predict(param_array, self._coordinates)
        if result.uncertainty is None:
            return np.zeros(len(self._coordinates))
        return np.mean(result.uncertainty, axis=-1).flatten()

    def _train_surrogate(self) -> TrainingResult:
        """Train surrogate model on current dataset."""
        training_config = TrainingConfig(
            surrogate_config=self.config.surrogate_config,
            use_ensemble=self.config.n_ensemble > 1,
            n_ensemble=self.config.n_ensemble,
            normalize_inputs=True,
            normalize_outputs=True,
            train_test_split=0.2,
            random_seed=self.config.random_seed,
            save_dir=self.surrogate_dir,
        )

        self.trainer = SurrogateTrainer(training_config)

        try:
            parameters, coordinates_list, outputs_list = self.dataset.prepare_training_data(
                output_field="displacement",
                valid_only=True,
            )
        except ValueError as e:
            return TrainingResult(success=False, error_message=str(e))

        # Use the current base mesh as the representative coordinate set for
        # acquisition function probing and uncertainty estimation.  Loading it
        # here (rather than from the dataset) means it automatically reflects
        # any r-adaptivity applied at the end of the previous iteration.
        base_manager = MFEMManager(str(self.config.base_mesh_path))
        self._coordinates = base_manager.get_nodes()

        result = self.trainer.train(parameters, coordinates_list, outputs_list)

        if result.success:
            self.surrogate = self.trainer.model

        return result

    def _evaluate_surrogate(self) -> UncertaintyAnalysis:
        """Evaluate surrogate model and identify weak regions using ensemble uncertainty."""
        if self.surrogate is None:
            return UncertaintyAnalysis()

        if self._coordinates is None:
            return UncertaintyAnalysis()

        # Initialize evaluator
        self.evaluator = SurrogateEvaluator(
            model=self.surrogate,
            parameter_names=self.config.parameter_names,
            parameter_bounds=self.config.parameter_bounds,
        )

        # Probe ensemble uncertainty across parameter space — no ground truth needed
        return self.evaluator.analyze_uncertainty(
            coordinates=self._coordinates,
            n_probe_samples=100,
            uncertainty_threshold=self.config.convergence_threshold,
        )

    def get_dataset(self) -> FEMDataset:
        """Get the current dataset."""
        return self.dataset

    def get_surrogate(self):
        """Get the trained surrogate model."""
        return self.surrogate
