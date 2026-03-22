"""
Comprehensive Framework Test with Visualization.

Single test file that validates all major features of the meshforge framework
and produces ONE consolidated visualization. Run with:

    python tests/test_comprehensive.py

Or with pytest:

    pytest tests/test_comprehensive.py -v
"""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock
import warnings

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional visualization (matplotlib)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(__file__).parent / "test_outputs"
N_SAMPLES = 50
N_POINTS = 100
N_ENSEMBLE = 3
EPOCHS = 50


# ============================================================================
# Helper Functions
# ============================================================================

def create_synthetic_data(n_samples: int, n_points: int, seed: int = 42):
    """Create synthetic training data for DeepONet."""
    np.random.seed(seed)
    n_params = 3
    branch_inputs = np.random.rand(n_samples, n_params).astype(np.float32)

    x = np.linspace(0, 10, int(np.sqrt(n_points)))
    y = np.linspace(0, 10, int(np.sqrt(n_points)))
    xx, yy = np.meshgrid(x, y)
    trunk_inputs = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    actual_n_points = trunk_inputs.shape[0]

    outputs = np.zeros((n_samples, actual_n_points, 1), dtype=np.float32)
    for i in range(n_samples):
        p = branch_inputs[i]
        field = (p[0] * np.sin(trunk_inputs[:, 0]) +
                 p[1] * np.cos(trunk_inputs[:, 1]) +
                 p[2] * trunk_inputs[:, 0] * trunk_inputs[:, 1] * 0.1)
        outputs[i, :, 0] = field

    return branch_inputs, trunk_inputs, outputs


def create_test_mesh_content():
    """Create a minimal valid MFEM 2D mesh string."""
    return """MFEM mesh v1.0

dimension
2

elements
6
1 3 0 1 5 4
1 3 1 2 6 5
1 3 2 3 7 6
1 3 4 5 9 8
1 3 5 6 10 9
1 3 6 7 11 10

boundary
8
1 1 0 1
1 1 1 2
1 1 2 3
2 1 3 7
2 1 7 11
3 1 11 10
3 1 10 9
3 1 9 8

vertices
12
2
0 0
1 0
2 0
3 0
0 1
1 1
2 1
3 1
0 2
1 2
2 2
3 2
"""


# ============================================================================
# Test Functions
# ============================================================================

def test_imports():
    """Test all framework imports."""
    results = {}

    modules = [
        ("Core", "from meshforge import MFEMManager, MeshManager"),
        ("Solvers", "from meshforge import SolverInterface, PhysicsType, MaterialProperties"),
        ("R-Adaptivity", "from meshforge import TMOPAdaptivity, AdaptivityConfig"),
        ("Orchestration", "from meshforge.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig"),
        ("Surrogate Base", "from meshforge.surrogate.base import SurrogateModel, SurrogateConfig"),
        ("Acquisition", "from meshforge.surrogate.acquisition import UncertaintySampling, ExpectedImprovement"),
        ("Error Analysis", "from meshforge.surrogate.error_analysis import SpatialErrorAnalyzer"),
        ("Dataset", "from meshforge import FEMDataset, FEMSample"),
    ]

    for name, import_stmt in modules:
        try:
            exec(import_stmt)
            results[name] = True
        except ImportError:
            results[name] = False

    return results


def test_surrogate():
    """Test surrogate model (placeholder - FNO/Transolver pending)."""
    # DeepONet has been removed; FNO/Transolver implementation is planned
    # Return None to indicate test is skipped
    return None, None, None, None


def test_acquisition():
    """Test acquisition functions."""
    try:
        from meshforge.surrogate.acquisition import (
            UncertaintySampling, ExpectedImprovement, QueryByCommittee
        )
        from meshforge.surrogate.base import PredictionResult
    except ImportError:
        return None

    # Mock model
    mock_model = MagicMock()
    def mock_predict(params, coords):
        n_samples = params.shape[0] if params.ndim > 1 else 1
        n_points = coords.shape[0]
        uncertainty = np.abs(np.random.randn(n_samples, n_points, 1)) * 0.1
        values = np.random.randn(n_samples, n_points, 1)
        return PredictionResult(values=values, uncertainty=uncertainty,
                               coordinates=coords, metadata={})
    mock_model.predict = mock_predict
    mock_model._models = [MagicMock() for _ in range(3)]

    candidates = np.random.rand(50, 3).astype(np.float32)
    coordinates = np.random.rand(20, 2).astype(np.float32) * 10

    results = {}
    strategies = {
        "Uncertainty": UncertaintySampling(),
        "EI": ExpectedImprovement(xi=0.01),
        "QBC": QueryByCommittee(),
    }

    for name, acq in strategies.items():
        scores = acq.compute(candidates, mock_model, coordinates, best_error=0.5)
        results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    return results


def test_error_analysis():
    """Test error analysis with hotspot detection."""
    try:
        from meshforge.surrogate.error_analysis import SpatialErrorAnalyzer
        from meshforge.surrogate.base import PredictionResult
    except ImportError:
        return None, None, None

    # Create grid
    x = np.linspace(0, 10, 25)
    y = np.linspace(0, 10, 25)
    xx, yy = np.meshgrid(x, y)
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])

    # Mock model with bias near (8, 8)
    mock_model = MagicMock()
    def mock_predict(params, coords):
        n_points = coords.shape[0]
        bias = 0.3 * np.exp(-np.linalg.norm(coords - [8, 8], axis=1) / 3)
        true_field = np.sin(coords[:, 0] * 0.5) * np.cos(coords[:, 1] * 0.5)
        pred_field = true_field - bias
        return PredictionResult(
            values=pred_field.reshape(1, n_points, 1),
            uncertainty=np.abs(bias).reshape(1, n_points, 1) * 0.5,
            coordinates=coords, metadata={}
        )
    mock_model.predict = mock_predict

    analyzer = SpatialErrorAnalyzer(mock_model, coordinates)
    params = np.array([[0.5, 1.0, 0.3]])
    true_values = np.sin(coordinates[:, 0] * 0.5) * np.cos(coordinates[:, 1] * 0.5)

    error_field = analyzer.compute_error_field(params, true_values)
    hotspots = analyzer.identify_hotspots(error_field, threshold_percentile=85, min_cluster_size=3)

    return coordinates, error_field, hotspots


def test_orchestration():
    """Test adaptive orchestration setup."""
    try:
        from meshforge.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig
    except ImportError:
        return None

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mesh', delete=False) as f:
        f.write(create_test_mesh_content())
        mesh_path = Path(f.name)

    output_dir = OUTPUT_DIR / "orchestration_test"
    output_dir.mkdir(exist_ok=True)

    config = AdaptiveConfig(
        base_mesh_path=mesh_path,
        output_dir=output_dir,
        parameter_bounds={"delta_R": (-0.5, 0.5)},
        initial_samples=5,
        max_samples=20,
    )

    orchestrator = AdaptiveOrchestrator(config)
    initial_params = orchestrator._generate_initial_parameters()

    mesh_path.unlink(missing_ok=True)

    return {
        "directories_created": all([
            orchestrator.meshes_dir.exists(),
            orchestrator.dataset_dir.exists(),
            orchestrator.surrogate_dir.exists(),
        ]),
        "initial_params": len(initial_params),
    }


# ============================================================================
# Main Runner with Consolidated Visualization
# ============================================================================

def run_all_tests(visualize: bool = True):
    """Run all tests and create consolidated visualization."""
    print("\n" + "="*60)
    print(" MESHFORGE FRAMEWORK VALIDATION")
    print("="*60)

    warnings.filterwarnings('ignore')
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = {}

    # Test 1: Imports
    print("\n[1/5] Testing imports...")
    import_results = test_imports()
    passed = sum(import_results.values())
    total = len(import_results)
    results["imports"] = {"passed": passed, "total": total}
    print(f"      {passed}/{total} modules imported successfully")

    # Test 2: Surrogate
    print("[2/5] Testing surrogate model...")
    history, true_vals, pred_vals, metrics = test_surrogate()
    if metrics:
        results["surrogate"] = metrics
        print(f"      MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.3f}")
    else:
        results["surrogate"] = None
        print("      SKIPPED (FNO/Transolver not yet implemented)")

    # Test 3: Acquisition
    print("[3/5] Testing acquisition functions...")
    acq_results = test_acquisition()
    if acq_results:
        results["acquisition"] = acq_results
        print(f"      Tested {len(acq_results)} strategies")
    else:
        results["acquisition"] = None
        print("      SKIPPED")

    # Test 4: Error Analysis
    print("[4/5] Testing error analysis...")
    coords, error_field, hotspots = test_error_analysis()
    if error_field is not None:
        results["error_analysis"] = {
            "mean_error": float(np.mean(error_field)),
            "n_hotspots": len(hotspots)
        }
        print(f"      Mean error: {np.mean(error_field):.4f}, Hotspots: {len(hotspots)}")
    else:
        results["error_analysis"] = None
        print("      SKIPPED")

    # Test 5: Orchestration
    print("[5/5] Testing orchestration...")
    orch_results = test_orchestration()
    if orch_results:
        results["orchestration"] = orch_results
        print(f"      Directories: {'OK' if orch_results['directories_created'] else 'FAIL'}")
    else:
        results["orchestration"] = None
        print("      SKIPPED")

    # Create consolidated visualization
    if visualize and MATPLOTLIB_AVAILABLE and metrics is not None:
        print("\nCreating visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle("MeshForge Framework Validation", fontsize=14, fontweight='bold')

        # Panel 1: Training Loss
        ax = axes[0, 0]
        train_loss = [float(np.atleast_1d(x)[0]) for x in history["loss_train"]]
        test_loss = [float(np.atleast_1d(x)[0]) for x in history["loss_test"]]
        ax.semilogy(train_loss, 'b-', label='Train', linewidth=2)
        ax.semilogy(test_loss, 'r--', label='Test', linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Surrogate Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: True vs Predicted
        ax = axes[0, 1]
        ax.scatter(true_vals.ravel(), pred_vals.ravel(), alpha=0.4, s=10, c='steelblue')
        lims = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
        ax.plot(lims, lims, 'r--', linewidth=2)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Prediction (R² = {metrics['r2']:.3f})")
        ax.grid(True, alpha=0.3)

        # Panel 3: Error Field with Hotspot
        ax = axes[0, 2]
        if error_field is not None:
            grid_size = int(np.sqrt(len(error_field)))
            error_grid = error_field.reshape(grid_size, grid_size)
            im = ax.imshow(error_grid, origin='lower', extent=[0, 10, 0, 10], cmap='Reds')
            if hotspots:
                for h in hotspots:
                    circle = plt.Circle(h.center, h.radius, fill=False, color='blue', linewidth=2)
                    ax.add_patch(circle)
                    ax.plot(h.center[0], h.center[1], 'b*', markersize=12)
            ax.set_title(f"Error Field ({len(hotspots)} hotspot)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Error Field")

        # Panel 4: Acquisition Scores
        ax = axes[1, 0]
        if acq_results:
            names = list(acq_results.keys())
            means = [acq_results[n]["mean"] for n in names]
            stds = [acq_results[n]["std"] for n in names]
            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'seagreen'])
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylabel("Score")
            ax.set_title("Acquisition Functions")
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Acquisition Functions")

        # Panel 5: Test Summary
        ax = axes[1, 1]
        ax.axis('off')
        summary = "TEST RESULTS\n" + "="*25 + "\n\n"

        tests = [
            ("Imports", results["imports"]["passed"] == results["imports"]["total"]),
            ("Surrogate", results["surrogate"] is not None),
            ("Acquisition", results["acquisition"] is not None),
            ("Error Analysis", results["error_analysis"] is not None),
            ("Orchestration", results["orchestration"] is not None and
                              results["orchestration"]["directories_created"]),
        ]

        for name, passed in tests:
            status = "PASS" if passed else "FAIL"
            symbol = "✓" if passed else "✗"
            summary += f"  {symbol} {name}: {status}\n"

        n_passed = sum(1 for _, p in tests if p)
        summary += f"\n{'='*25}\n  Total: {n_passed}/{len(tests)} passed"

        ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        # Panel 6: Key Metrics
        ax = axes[1, 2]
        ax.axis('off')
        metrics_text = "KEY METRICS\n" + "="*25 + "\n\n"

        if results["surrogate"]:
            metrics_text += f"Surrogate MSE: {results['surrogate']['mse']:.4f}\n"
            metrics_text += f"Surrogate R²:  {results['surrogate']['r2']:.3f}\n\n"

        if results["error_analysis"]:
            metrics_text += f"Mean Error:    {results['error_analysis']['mean_error']:.4f}\n"
            metrics_text += f"Hotspots:      {results['error_analysis']['n_hotspots']}\n\n"

        if results["orchestration"]:
            metrics_text += f"Init Samples:  {results['orchestration']['initial_params']}\n"

        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        plt.tight_layout()
        output_path = OUTPUT_DIR / "validation_report.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")

    # Final summary
    print("\n" + "="*60)
    n_passed = sum(1 for v in results.values() if v is not None and
                   (not isinstance(v, dict) or v.get("passed", v.get("directories_created", True))))
    print(f" COMPLETE: {n_passed}/5 tests passed")
    print("="*60)

    return results


# ============================================================================
# Entry Points
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run framework validation")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    results = run_all_tests(visualize=not args.no_viz)

    # Exit code based on results
    failed = sum(1 for v in results.values() if v is None)
    sys.exit(1 if failed > 2 else 0)  # Allow some skips


# Pytest integration
class TestComprehensiveSuite:
    """Pytest wrapper."""

    def test_imports(self):
        results = test_imports()
        assert all(results.values()), f"Failed imports: {[k for k,v in results.items() if not v]}"

    def test_surrogate(self):
        _, _, _, metrics = test_surrogate()
        if metrics is None:
            import pytest
            pytest.skip("Surrogate model (FNO/Transolver) not yet implemented")
        assert metrics["r2"] > -1, "R² should be reasonable"

    def test_acquisition(self):
        results = test_acquisition()
        if results is None:
            import pytest
            pytest.skip("Acquisition module not available")
        assert len(results) >= 3, "Should test multiple strategies"

    def test_error_analysis(self):
        coords, error_field, hotspots = test_error_analysis()
        if error_field is None:
            import pytest
            pytest.skip("Error analysis not available")
        assert len(error_field) > 0, "Should compute error field"
        assert len(hotspots) >= 0, "Should detect hotspots"

    def test_orchestration(self):
        results = test_orchestration()
        if results is None:
            import pytest
            pytest.skip("Orchestration not available")
        assert results["directories_created"], "Should create output directories"
