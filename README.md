# meshforge

**meshforge** is an adaptive learning framework for efficient FEM dataset generation. It uses DeepONet surrogate models to intelligently guide data collection, focusing computational resources on regions where the surrogate model is weak.

Generate high-quality training data for scientific machine learning with minimal FEM simulations.

---

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)

---

## Key Features

- **Adaptive Sampling**: Automatically identifies regions in parameter space where the surrogate model has high error or uncertainty
- **DeepONet Integration**: Uses DeepXDE for operator learning (DeepONet) surrogate models
- **MFEM Native**: Works directly with MFEM mesh format - no Abaqus INP conversion needed
- **Efficient**: Reduces the number of expensive FEM simulations by 50-80% compared to uniform sampling
- **Extensible**: Modular design for custom physics, mesh types, and surrogate architectures

---

## Quick Start

```python
from meshforge import AdaptiveOrchestrator, AdaptiveConfig

# Configure adaptive learning
config = AdaptiveConfig(
    base_mesh_path="meshes/plate_with_hole.mesh",
    output_dir="./output",
    parameter_names=["delta_R", "E", "nu"],
    parameter_bounds={
        "delta_R": (-0.3, 0.3),
        "E": (150e9, 250e9),
        "nu": (0.25, 0.35),
    },
    initial_samples=20,
    samples_per_iteration=10,
    max_iterations=10,
    convergence_threshold=0.05,
)

# Run adaptive learning
orchestrator = AdaptiveOrchestrator(config)
result = orchestrator.run()

# Access results
print(f"Total samples generated: {result.total_samples}")
print(f"Final surrogate error: {result.final_error:.4f}")

# Get trained surrogate for inference
surrogate = orchestrator.get_surrogate()
dataset = orchestrator.get_dataset()
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

---

## How It Works

### Adaptive Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    Adaptive Learning Workflow                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Generate Initial Samples (Latin Hypercube Sampling)        │
│         ↓                                                       │
│  2. Run FEM Simulations → Store in Dataset                     │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ADAPTIVE LOOP                                           │   │
│  │                                                          │   │
│  │  3. Train DeepONet Surrogate (ensemble for uncertainty)  │   │
│  │         ↓                                                │   │
│  │  4. Evaluate Surrogate → Identify Weak Regions           │   │
│  │         ↓                                                │   │
│  │  5. Generate New Samples in Weak Regions                 │   │
│  │         ↓                                                │   │
│  │  6. Run FEM Simulations → Update Dataset                 │   │
│  │         ↓                                                │   │
│  │  7. Check Convergence (error < threshold?)               │   │
│  │         ↓                                                │   │
│  │      NO → Continue loop    YES → Exit                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  8. Save Dataset + Trained Surrogate Model                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Surrogate Model (DeepONet)**: Neural network that learns the mapping from input parameters to FEM solution fields. Trained on simulation data.

2. **Uncertainty Quantification**: Ensemble of DeepONet models provides uncertainty estimates. High uncertainty indicates regions needing more data.

3. **Weak Region Detection**: Analyzes surrogate errors and uncertainty to identify parameter regions where predictions are poor.

4. **Targeted Sampling**: New simulations are concentrated in weak regions, maximizing information gain per simulation.

---

## Usage Examples

### Basic Adaptive Learning

```python
from meshforge import AdaptiveOrchestrator, AdaptiveConfig

config = AdaptiveConfig(
    base_mesh_path="plate.mesh",
    output_dir="./results",
    parameter_bounds={"delta_R": (-0.5, 0.5)},
    initial_samples=20,
    max_iterations=5,
)

result = AdaptiveOrchestrator(config).run()
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
)
```

### Using the Trained Surrogate

```python
import numpy as np

# After training
surrogate = orchestrator.get_surrogate()

# Predict on new parameters
new_params = np.array([[0.1, 200e9, 0.3]])  # delta_R, E, nu
coordinates = dataset[0].coordinates  # Use mesh coordinates

prediction = surrogate.predict(new_params, coordinates)
print(f"Predicted displacement: {prediction.values}")
print(f"Uncertainty: {prediction.uncertainty}")
```

### Working with the Dataset

```python
from meshforge import FEMDataset

# Load existing dataset
dataset = FEMDataset.load("./results/dataset")

# Get statistics
stats = dataset.compute_statistics()
print(f"Samples: {stats.n_samples}, Valid: {stats.n_valid}")

# Prepare for training
params, coords, outputs = dataset.prepare_training_data(
    output_field="displacement"
)
```

### Standalone Mesh Morphing

```python
from meshforge import MFEMManager, apply_morphing

# Load mesh
manager = MFEMManager("plate.mesh")

# Apply morphing (change hole radius)
apply_morphing(manager, config_path="morphing.md", delta_r=0.3)

# Save
manager.save("plate_morphed.mesh")
```

---

## Architecture

### Module Structure

```
meshforge/
├── __init__.py              # Public API
├── cli.py                   # Command-line interface
│
├── surrogate/               # DeepONet surrogate models
│   ├── base.py             # SurrogateModel interface
│   ├── deeponet.py         # DeepONet + Ensemble
│   ├── trainer.py          # Training workflow
│   └── evaluator.py        # Uncertainty analysis
│
├── data/                    # Dataset management
│   ├── dataset.py          # FEMSample, FEMDataset
│   └── loader.py           # Data loaders
│
├── orchestration/           # Workflow control
│   └── adaptive.py         # AdaptiveOrchestrator
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
Parameters → Mesh Morphing → FEM Solver → Solution Fields
     ↓                                          ↓
  Dataset ←─────────────────────────────────────┘
     ↓
  DeepONet Training
     ↓
  Uncertainty Analysis → New Parameters (loop)
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
| `convergence_threshold` | float | 0.05 | Error threshold |
| `use_ensemble` | bool | True | Use ensemble for uncertainty |
| `n_ensemble` | int | 5 | Number of ensemble models |

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
from meshforge.surrogate import DeepONetSurrogate, SurrogateTrainer, SurrogateEvaluator
```

### Key Methods

```python
# AdaptiveOrchestrator
orchestrator.run() → AdaptiveResult
orchestrator.get_dataset() → FEMDataset
orchestrator.get_surrogate() → SurrogateModel

# FEMDataset
dataset.add_sample(sample)
dataset.prepare_training_data(output_field) → (params, coords, outputs)
dataset.compute_statistics() → DatasetStatistics
dataset.save(path)
FEMDataset.load(path) → FEMDataset

# SurrogateModel
model.build(input_dim, coord_dim, num_sensors)
model.train(branch_inputs, trunk_inputs, outputs) → history
model.predict(branch_input, trunk_input) → PredictionResult
```

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Authors

- H.-Y. Nam
- Q. Jiang

## References

- Lu et al. (2021): "Learning nonlinear operators via DeepONet", Nature Machine Intelligence
- [MFEM](https://mfem.org/): Modular Finite Element Methods library
- [DeepXDE](https://deepxde.readthedocs.io/): Deep learning library for scientific computing
