# R-Adaptivity Configuration

Error-driven mesh adaptation using MFEM's TMOP (Target-Matrix Optimization Paradigm).
Nodes are redistributed to cluster in regions where the surrogate model has high prediction error.

---

## Configuration (YAML)

```yaml
problem_type: r_adaptivity

# R-adaptivity settings
adaptivity:
  # Target element size scaling
  # High-error regions get size_scale_min (small elements, attract nodes)
  # Low-error regions get size_scale_max (large elements, repel nodes)
  size_scale_min: 0.3
  size_scale_max: 2.0

  # Error field processing
  error_threshold: 0.1
  smoothing_iterations: 2

  # TMOP settings
  target_type: ideal_shape_equal_size
  quality_metric: shape_and_size
  barrier_type: shifted  # "shifted" or "pseudo"

  # Newton solver
  max_iterations: 200
  tolerance: 1.0e-8
  min_det_threshold: 0.001  # For barrier activation
  verbosity: 0

  # Boundary handling
  fix_boundary: true
```

---

## How R-Adaptivity Works

1. **Error Field Input**: The surrogate model (DeepONet) provides pointwise prediction errors at each mesh node.

2. **Target Size Computation**:
   - High error → small target element size → attracts nodes
   - Low error → large target element size → repels nodes

3. **TMOP Optimization**: MFEM's variational mesh optimizer solves for optimal node positions that:
   - Match the spatially-varying target sizes
   - Preserve element quality
   - Prevent element inversion (via barrier functions)

4. **Result**: Mesh has higher resolution in regions where the surrogate model needs improvement.

---

## Parameter Reference

### Size Scaling
- **size_scale_min**: Minimum target element size scaling (0.1-0.5 typical)
- **size_scale_max**: Maximum target element size scaling (1.5-3.0 typical)
- **error_threshold**: Errors below this are considered "low error"

### TMOP Settings
- **target_type**: How to construct target elements
  - `ideal_shape_equal_size`: Equilateral elements matching original size
  - `given_shape_and_size`: Preserve original element shapes
- **quality_metric**: What to optimize
  - `shape`: Element shape only
  - `size`: Element size only
  - `shape_and_size`: Combined (recommended)
- **barrier_type**: Barrier function for preventing inversion
  - `shifted`: More aggressive, recommended
  - `pseudo`: Smoother

### Solver Settings
- **max_iterations**: Maximum Newton iterations
- **tolerance**: Convergence tolerance
- **min_det_threshold**: Jacobian determinant threshold for barrier activation

---

## Usage Example

```python
from piano.morphing import TMOPAdaptivity, AdaptivityConfig
from piano.mesh.mfem_manager import MFEMManager

# Load mesh
manager = MFEMManager("mesh.mesh")
coords = manager.get_nodes()

# Get error field from surrogate model
predictions = surrogate.predict(params, coords)
error_field = np.abs(predictions - ground_truth)

# Configure adaptivity
config = AdaptivityConfig(
    size_scale_min=0.3,
    size_scale_max=2.0,
    max_iterations=200,
)

# Adapt mesh
adaptivity = TMOPAdaptivity(config)
result = adaptivity.adapt(manager, error_field)

if result.success:
    print(f"Quality improved: {result.quality_before['min_quality']:.3f} → "
          f"{result.quality_after['min_quality']:.3f}")
    # manager now has adapted mesh
    manager.save("adapted_mesh.mesh")
```

---

## Requirements

- PyMFEM with TMOP support (`pip install mfem`)
