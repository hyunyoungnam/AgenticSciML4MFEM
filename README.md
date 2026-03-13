# meshforge

**meshforge** is an adaptive active learning framework for efficient FEM dataset generation. It uses DeepONet surrogate models with **informative sampling** to intelligently guide data collection, focusing computational resources on regions where the surrogate model is weak.

Generate high-quality training data for scientific machine learning with minimal FEM simulations through autonomous active learning.

---

## Table of Contents

- [Key Features](#key-features)
- [Core Concept: Informative Sampling](#core-concept-informative-sampling)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Active Learning Loop](#active-learning-loop)
- [Acquisition Functions](#acquisition-functions)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)

---

## Key Features

- **Active Learning Loop**: Autonomous iteration between data generation and model training with intelligent stopping criteria
- **Informative Sampling**: Acquisition functions (Uncertainty, Expected Improvement, Query-by-Committee) prioritize samples that maximize model improvement
- **DeepONet Integration**: Ensemble-based operator learning with epistemic uncertainty quantification
- **Adaptive Budget**: Dynamically adjusts sampling rate based on convergence progress
- **MFEM Native**: Works directly with MFEM mesh format
- **Efficient**: Reduces FEM simulations by 50-80% compared to uniform sampling
- **Convergence Monitoring**: Multiple stopping criteria (error threshold, patience, budget, diminishing returns)

---

## Core Concept: Informative Sampling

Traditional approaches sample the parameter space uniformly. **Informative sampling** uses the surrogate model's uncertainty to identify where new simulations would be most valuable:

```
Traditional Sampling:          Active Learning:
┌───────────────────┐         ┌───────────────────┐
│ • • • • • • • • • │         │ •       •     •   │
│ • • • • • • • • • │         │           ••••••  │ ← High uncertainty region
│ • • • • • • • • • │   vs    │ •    ••••••••••  │   gets more samples
│ • • • • • • • • • │         │       •••••••     │
│ • • • • • • • • • │         │ •   •       •     │
└───────────────────┘         └───────────────────┘
  100 samples                   40 samples, same accuracy
```

The acquisition function quantifies "informativeness" - regions with high ensemble disagreement, high expected improvement, or high uncertainty receive more samples.

---

## Quick Start

```python
from meshforge import AdaptiveOrchestrator, AdaptiveConfig

# Configure active learning
config = AdaptiveConfig(
    base_mesh_path="meshes/plate_with_hole.mesh",
    output_dir="./output",
    parameter_names=["delta_R", "E", "nu"],
    parameter_bounds={
        "delta_R": (-0.3, 0.3),
        "E": (150e9, 250e9),
        "nu": (0.25, 0.35),
    },

    # Initial exploration
    initial_samples=20,
    samples_per_iteration=10,
    max_iterations=15,

    # Active learning settings
    acquisition_strategy="uncertainty",  # or "ei", "qbc", "ucb", "hybrid"
    convergence_threshold=0.05,
    convergence_patience=3,
    max_samples=200,

    # Ensemble for uncertainty quantification
    use_ensemble=True,
    n_ensemble=5,
)

# Run autonomous active learning
orchestrator = AdaptiveOrchestrator(config)
result = orchestrator.run()

# Results
print(f"Converged: {result.stopping_criterion.name}")
print(f"Total samples: {result.total_samples}")
print(f"Error reduction: {result.error_reduction_percent:.1f}%")
print(f"Sample efficiency: {result.sample_efficiency:.6f}")
```

---

## Installation

### From Source

```bash
git clone https://github.com/your-repo/meshforge.git
cd meshforge
pip install -e ".[all]"
```

### Dependencies

```bash
# Core + surrogate model support
pip install -e ".[surrogate]"

# With MFEM solver
pip install -e ".[mfem]"

# Full installation
pip install -e ".[all]"
```

### Prerequisites

- Python 3.9+
- DeepXDE with TensorFlow or PyTorch backend
- (Optional) PyMFEM for FEM simulations
- scipy (for acquisition functions)
- (Optional) scikit-learn (for error hotspot clustering)

---

## Active Learning Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ACTIVE LEARNING WORKFLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. INITIAL SAMPLING (Latin Hypercube)                               │
│     └─ Generate diverse initial samples for good coverage            │
│                          ↓                                            │
│  2. FEM SIMULATIONS                                                  │
│     └─ Run MFEM solver for each parameter configuration              │
│                          ↓                                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  ACTIVE LEARNING LOOP                                           │ │
│  │                                                                   │ │
│  │  3. TRAIN DeepONet ENSEMBLE                                      │ │
│  │     └─ N models with bootstrap sampling for uncertainty          │ │
│  │                          ↓                                        │ │
│  │  4. EVALUATE & COMPUTE ACQUISITION SCORES                        │ │
│  │     ├─ Predict on candidate parameter configurations             │ │
│  │     ├─ Compute uncertainty (ensemble disagreement)               │ │
│  │     └─ Score candidates with acquisition function                │ │
│  │                          ↓                                        │ │
│  │  5. SELECT INFORMATIVE SAMPLES                                   │ │
│  │     ├─ Rank candidates by acquisition score                      │ │
│  │     ├─ Apply diversity filter to avoid clustering                │ │
│  │     └─ Select top-k most informative configurations              │ │
│  │                          ↓                                        │ │
│  │  6. CHECK CONVERGENCE CRITERIA                                   │ │
│  │     ├─ Error < threshold? → CONVERGED                            │ │
│  │     ├─ No improvement for N iterations? → PATIENCE_EXHAUSTED     │ │
│  │     ├─ Samples >= budget? → BUDGET_EXHAUSTED                     │ │
│  │     ├─ Uncertainty < threshold? → LOW_UNCERTAINTY                │ │
│  │     └─ Efficiency dropping? → DIMINISHING_RETURNS                │ │
│  │                          ↓                                        │ │
│  │  7. RUN NEW SIMULATIONS & LOOP                                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                          ↓                                            │
│  8. SAVE: Dataset, Trained Surrogate, Metrics                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Acquisition Functions

Acquisition functions determine which samples are most "informative". Each balances exploration (sampling uncertain regions) and exploitation (sampling where errors are expected to be high).

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `uncertainty` | Sample where model uncertainty is highest | General use, exploration-focused |
| `ei` | Expected Improvement over current best | Balancing exploration/exploitation |
| `qbc` | Query-by-Committee (ensemble disagreement) | When ensemble diversity matters |
| `ucb` | Upper Confidence Bound (optimistic) | When you want aggressive exploration |
| `hybrid` | Weighted combination of strategies | Complex problems needing multiple signals |

### Usage

```python
# Pure uncertainty sampling
config = AdaptiveConfig(
    acquisition_strategy="uncertainty",
    ...
)

# Expected Improvement (Bayesian optimization style)
config = AdaptiveConfig(
    acquisition_strategy="ei",
    ...
)

# Query-by-Committee (ensemble disagreement)
config = AdaptiveConfig(
    acquisition_strategy="qbc",
    ...
)

# Custom hybrid
from meshforge.surrogate.acquisition import HybridAcquisition, UncertaintySampling, ExpectedImprovement

hybrid = HybridAcquisition([
    (UncertaintySampling(), 0.6),
    (ExpectedImprovement(), 0.4),
])
```

---

## Usage Examples

### Basic Active Learning

```python
from meshforge import AdaptiveOrchestrator, AdaptiveConfig

config = AdaptiveConfig(
    base_mesh_path="plate.mesh",
    output_dir="./results",
    parameter_bounds={"delta_R": (-0.5, 0.5)},
    acquisition_strategy="uncertainty",
    initial_samples=20,
    max_iterations=10,
    convergence_threshold=0.05,
)

result = AdaptiveOrchestrator(config).run()
print(f"Stopping reason: {result.stopping_criterion.name}")
```

### With Progress Callback

```python
def on_iteration(iteration, metrics):
    print(f"Iteration {iteration}: error={metrics['test_error']:.4f}, "
          f"samples={metrics['n_samples']}, uncertainty={metrics['mean_uncertainty']:.4f}")

result = orchestrator.run(callback=on_iteration)
```

### Analyzing Learning Efficiency

```python
from meshforge.orchestration.metrics import ActiveLearningMetrics

# Load saved metrics
metrics = ActiveLearningMetrics.load("./results/metrics/active_learning_metrics.json")

# Get summary
print(metrics.summary())

# Check efficiency
print(f"Sample efficiency: {metrics.compute_efficiency():.6f}")
print(f"Optimal stopping point: iteration {metrics.find_optimal_stopping_point()}")

# Plot error history
samples, train_errors, test_errors = metrics.get_error_history()
```

### Custom Physics Configuration

```python
from meshforge import AdaptiveConfig, PhysicsConfig, PhysicsType, MaterialProperties

physics = PhysicsConfig(
    physics_type=PhysicsType.LINEAR_ELASTICITY,
    material=MaterialProperties(E=200e9, nu=0.3),
)

config = AdaptiveConfig(
    base_mesh_path="plate.mesh",
    output_dir="./results",
    physics_config=physics,
    parameter_bounds={"delta_R": (-0.3, 0.3)},
    acquisition_strategy="ei",
)
```

### Using the Trained Surrogate

```python
import numpy as np

# After training
surrogate = orchestrator.get_surrogate()

# Predict with uncertainty
new_params = np.array([[0.1, 200e9, 0.3]])  # delta_R, E, nu
coordinates = dataset[0].coordinates

prediction = surrogate.predict(new_params, coordinates)
print(f"Predicted displacement: {prediction.values}")
print(f"Uncertainty (std): {prediction.uncertainty}")

# Get detailed epistemic uncertainty
detailed = surrogate.predict_with_epistemic_uncertainty(new_params, coordinates)
print(f"95% CI: [{detailed.metadata['lower_95']}, {detailed.metadata['upper_95']}]")
```

### Spatial Error Analysis

```python
from meshforge.surrogate.error_analysis import SpatialErrorAnalyzer

analyzer = SpatialErrorAnalyzer(surrogate, coordinates)

# Analyze where errors concentrate spatially
analysis = analyzer.analyze(params, true_values, parameter_names=["delta_R", "E", "nu"])

print(f"Global mean error: {analysis.global_stats['mean_error']:.6f}")
print(f"Number of hotspots: {len(analysis.hotspots)}")

for hotspot in analysis.hotspots[:3]:
    print(f"  Hotspot at {hotspot.center}: error={hotspot.mean_error:.4f}")
```

### Direct Acquisition Function Usage

```python
from meshforge.surrogate.acquisition import get_acquisition_function
import numpy as np

# Create acquisition function
acq_fn = get_acquisition_function("ucb", kappa=2.0)

# Generate candidates
candidates = np.random.uniform(
    low=[b[0] for b in param_bounds.values()],
    high=[b[1] for b in param_bounds.values()],
    size=(1000, len(param_bounds))
)

# Score candidates
scores = acq_fn.compute(candidates, surrogate, coordinates)

# Select top 10 with diversity
result = acq_fn.select_batch(candidates, surrogate, coordinates, batch_size=10, diversity_weight=0.2)
best_indices = result.best_indices
```

---

## Architecture

### Module Structure

```
meshforge/
├── __init__.py              # Public API
├── cli.py                   # Command-line interface
│
├── surrogate/               # Surrogate models & active learning
│   ├── base.py             # SurrogateModel interface
│   ├── deeponet.py         # DeepONet + Ensemble with uncertainty
│   ├── trainer.py          # Training workflow
│   ├── evaluator.py        # Uncertainty analysis + acquisition sampling
│   ├── acquisition.py      # Acquisition functions (NEW)
│   └── error_analysis.py   # Spatial error decomposition (NEW)
│
├── orchestration/           # Workflow control
│   ├── adaptive.py         # AdaptiveOrchestrator (active learning loop)
│   └── metrics.py          # Learning efficiency tracking (NEW)
│
├── data/                    # Dataset management
│   ├── dataset.py          # FEMSample, FEMDataset
│   └── loader.py           # Data loaders
│
├── mesh/                    # Mesh handling
│   ├── base.py             # MeshManager interface
│   └── mfem_manager.py     # MFEM implementation
│
├── solvers/                 # FEM solvers
│   ├── base.py             # SolverInterface
│   └── mfem_solver.py      # MFEM solver
│
├── morphing.py              # IDW mesh morphing
├── evaluation/              # Mesh quality metrics
└── agents/                  # LLM-based agents (optional)
```

### Data Flow

```
                    ┌──────────────────────────────┐
                    │     ACTIVE LEARNING LOOP      │
                    └──────────────────────────────┘
                                  │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
    ▼                            ▼                            ▼
┌─────────┐              ┌──────────────┐             ┌───────────────┐
│ Dataset │◄────────────│ FEM Solver   │◄────────────│ Acquisition   │
│         │              │              │             │ Function      │
└────┬────┘              └──────────────┘             └───────┬───────┘
     │                                                        │
     │ Training Data                              Informative │
     ▼                                              Samples   │
┌──────────────┐                                              │
│ DeepONet     │                                              │
│ Ensemble     │──────────────────────────────────────────────┘
│              │  Uncertainty Estimates
└──────────────┘
```

---

## Configuration

### AdaptiveConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_mesh_path` | Path | required | Path to base MFEM mesh |
| `output_dir` | Path | required | Output directory |
| `parameter_names` | List[str] | `["delta_R"]` | Parameter names |
| `parameter_bounds` | Dict | required | Bounds for each parameter |
| `initial_samples` | int | 20 | Initial LHS samples |
| `samples_per_iteration` | int | 10 | New samples per iteration |
| `max_iterations` | int | 10 | Maximum adaptive iterations |
| `convergence_threshold` | float | 0.05 | Error threshold for convergence |
| `use_ensemble` | bool | True | Use ensemble for uncertainty |
| `n_ensemble` | int | 5 | Number of ensemble models |

### Active Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `acquisition_strategy` | str | `"uncertainty"` | Acquisition function (`"uncertainty"`, `"ei"`, `"qbc"`, `"ucb"`, `"hybrid"`) |
| `adaptive_budget` | bool | True | Dynamically adjust samples per iteration |
| `convergence_patience` | int | 3 | Iterations without improvement before stopping |
| `min_improvement` | float | 0.01 | Minimum improvement to reset patience |
| `max_samples` | int | 500 | Hard budget limit |
| `diversity_weight` | float | 0.1 | Weight for diversity in selection (0-1) |
| `n_candidates` | int | 1000 | Candidates to consider per iteration |
| `uncertainty_threshold` | float | 0.05 | Stop when uncertainty drops below |

### SurrogateConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branch_layers` | List[int] | `[128, 128, 128]` | Branch network layers |
| `trunk_layers` | List[int] | `[128, 128, 128]` | Trunk network layers |
| `activation` | str | `"tanh"` | Activation function |
| `learning_rate` | float | 1e-3 | Learning rate |
| `epochs` | int | 10000 | Training epochs |
| `patience` | int | 1000 | Early stopping patience |

---

## API Reference

### Core Classes

```python
# Orchestration
from meshforge import AdaptiveOrchestrator, AdaptiveConfig, AdaptiveResult

# Dataset
from meshforge import FEMDataset, FEMSample, DatasetConfig

# Mesh
from meshforge import MFEMManager, MeshManager

# Solver
from meshforge import MFEMSolver, PhysicsConfig, PhysicsType, MaterialProperties

# Surrogate (import from submodule)
from meshforge.surrogate import DeepONetSurrogate, DeepONetEnsemble, SurrogateTrainer, SurrogateEvaluator

# Active Learning
from meshforge.surrogate.acquisition import (
    AcquisitionFunction,
    UncertaintySampling,
    ExpectedImprovement,
    QueryByCommittee,
    UpperConfidenceBound,
    HybridAcquisition,
    get_acquisition_function,
)
from meshforge.surrogate.error_analysis import SpatialErrorAnalyzer, ErrorDecomposer
from meshforge.orchestration.metrics import ActiveLearningMetrics, ConvergenceMonitor
```

### Key Methods

```python
# AdaptiveOrchestrator
orchestrator.run(callback=None) → AdaptiveResult
orchestrator.get_dataset() → FEMDataset
orchestrator.get_surrogate() → SurrogateModel

# SurrogateEvaluator (with acquisition)
evaluator.suggest_samples_active(budget, coordinates, acquisition_type, ...) → List[Dict]
evaluator.compute_acquisition_scores(candidates, coordinates, acquisition_type) → np.ndarray
evaluator.estimate_remaining_uncertainty(coordinates) → Dict[str, float]

# DeepONetEnsemble
model.predict(branch_input, trunk_input) → PredictionResult
model.predict_with_epistemic_uncertainty(...) → PredictionResult
model.get_ensemble_predictions(...) → np.ndarray
model.compute_disagreement(...) → np.ndarray

# ActiveLearningMetrics
metrics.log_iteration(...)
metrics.compute_efficiency() → float
metrics.detect_diminishing_returns() → bool
metrics.find_optimal_stopping_point() → int
metrics.summary() → str
```

---

## Stopping Criteria

The active learning loop stops when any of these criteria are met:

| Criterion | Condition | Configuration |
|-----------|-----------|---------------|
| `CONVERGED` | Error < threshold | `convergence_threshold` |
| `PATIENCE_EXHAUSTED` | No improvement for N iterations | `convergence_patience`, `min_improvement` |
| `BUDGET_EXHAUSTED` | Total samples >= limit | `max_samples` |
| `MAX_ITERATIONS` | Iterations >= limit | `max_iterations` |
| `LOW_UNCERTAINTY` | Mean uncertainty < threshold | `uncertainty_threshold` |
| `DIMINISHING_RETURNS` | Efficiency dropping consistently | `convergence_patience`, `min_improvement` |

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Authors

- H.-Y. Nam
- Q. Jiang

## References

- Lu et al. (2021): "Learning nonlinear operators via DeepONet", Nature Machine Intelligence
- Settles (2009): "Active Learning Literature Survey"
- [MFEM](https://mfem.org/): Modular Finite Element Methods library
- [DeepXDE](https://deepxde.readthedocs.io/): Deep learning library for scientific computing
