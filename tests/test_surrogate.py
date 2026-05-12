"""
tests/test_surrogate.py — Tests for the PIANO surrogate and physics layers.

Sections:
  1. FEMDataset / FEMSample
  2. SurrogateTrainer + EnsembleModel
  3. SurrogateEvaluator
  4. SpatialErrorAnalyzer + ErrorDecomposer
  5. AcquisitionFunctions
  6. PhaseFieldConfig + FEMSample extensions
  7. PhaseFieldGenerator config (no deps)
  8. GmshMeshGenerator  (skipif: gmsh)
  9. FEniCSPhaseFieldSolver  (skipif: dolfinx)
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "E":      (150e9, 250e9),
    "nu":     (0.25,  0.35),
    "load_x": (50e6,  150e6),
    "load_y": (-100e6, 100e6),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())

HAS_GMSH    = False
HAS_DOLFINX = False
try:
    import gmsh;    HAS_GMSH    = True
except (ImportError, OSError): pass
try:
    import dolfinx; HAS_DOLFINX = True
except (ImportError, OSError): pass


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_trainer():
    """2-member ensemble trained on 12 synthetic samples (5 epochs, < 5 s)."""
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.surrogate.base import TransolverConfig

    rng = np.random.default_rng(0)
    N_SAMPLES, N_POINTS = 12, 64

    params  = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                          size=(N_SAMPLES, 4)).astype(np.float32)
    coords  = rng.uniform(0, 1, size=(N_POINTS, 2)).astype(np.float32)
    outputs = [rng.uniform(0, 1e8, size=(N_POINTS, 1)).astype(np.float32)
               for _ in range(N_SAMPLES)]

    cfg = TrainingConfig(
        surrogate_config=TransolverConfig(
            d_model=32, n_heads=2, n_layers=1, slice_num=4,
            epochs=5, patience=10, batch_size=4,
        ),
        use_ensemble=True, n_ensemble=2,
        train_test_split=0.2, random_seed=0,
    )
    trainer = SurrogateTrainer(cfg)
    result  = trainer.train(params, [coords] * N_SAMPLES, outputs)
    assert result.success, f"Fixture training failed: {result.error_message}"
    return trainer, coords


# ===========================================================================
# 1. FEMDataset / FEMSample
# ===========================================================================

def test_femdataset_add_and_query():
    from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig

    ds = FEMDataset(DatasetConfig(parameter_names=PARAM_NAMES,
                                  parameter_bounds=PARAM_BOUNDS))
    assert len(ds) == 0

    rng    = np.random.default_rng(2)
    coords = rng.random((50, 2)).astype(np.float32)
    vm     = rng.random((50, 1)).astype(np.float32)

    for i in range(6):
        ds.add_sample(FEMSample(
            sample_id=f"s{i:03d}",
            parameters={"E": 200e9, "nu": 0.3, "load_x": 1e8, "load_y": 0.0},
            coordinates=coords, von_mises=vm, is_valid=True,
        ))

    assert len(ds) == 6
    assert len(ds.get_valid_samples()) == 6
    region = {"E": (190e9, 210e9), "nu": (0.25, 0.35), "load_x": (0.5e8, 1.5e8)}
    assert len(ds.get_samples_in_region(region)) == 6


def test_femdataset_prepare_training_data():
    from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig

    ds  = FEMDataset(DatasetConfig(parameter_names=PARAM_NAMES,
                                   parameter_bounds=PARAM_BOUNDS))
    rng = np.random.default_rng(3)
    for i in range(8):
        ds.add_sample(FEMSample(
            sample_id=f"s{i}",
            parameters={
                "E":      float(rng.uniform(150e9, 250e9)),
                "nu":     float(rng.uniform(0.25,  0.35)),
                "load_x": float(rng.uniform(50e6,  150e6)),
                "load_y": float(rng.uniform(-100e6, 100e6)),
            },
            coordinates=rng.random((40, 2)).astype(np.float32),
            von_mises=rng.random((40, 1)).astype(np.float32),
            is_valid=True,
        ))

    params_arr, coords_list, outputs_list = ds.prepare_training_data("von_mises")
    assert params_arr.shape == (8, 4)
    assert len(coords_list) == 8
    assert all(c.shape == (40, 2) for c in coords_list)


def test_femsample_output_fields():
    from piano.data.dataset import FEMSample

    coords = np.ones((20, 2), dtype=np.float32)
    vm     = np.full((20, 1), 5.0, dtype=np.float32)
    sample = FEMSample(
        sample_id="x",
        parameters={"E": 200e9, "nu": 0.3, "load_x": 1e8, "load_y": 0.0},
        coordinates=coords, von_mises=vm, is_valid=True,
    )
    field = sample.get_output_field("von_mises")
    assert field is not None and field.shape == (20, 1)
    pvec = sample.get_parameter_vector(PARAM_NAMES)
    assert pvec.shape == (4,) and pvec[0] == pytest.approx(200e9)


# ===========================================================================
# 2. SurrogateTrainer + EnsembleModel
# ===========================================================================

def test_surrogate_trainer_success(tiny_trainer):
    trainer, _ = tiny_trainer
    assert trainer.model is not None
    assert trainer.model.is_trained


def test_surrogate_prediction_shape(tiny_trainer):
    trainer, coords = tiny_trainer
    params = np.random.default_rng(7).uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(1, 4)
    ).astype(np.float32)
    mean, unc = trainer.predict_with_uncertainty(params, coords)
    assert mean.shape[0] == coords.shape[0]
    assert unc is not None and unc.shape[0] == coords.shape[0]


def test_normalizer_roundtrip():
    from piano.surrogate.trainer import Normalizer

    data = np.random.default_rng(3).uniform(0, 1000, size=(100, 3)).astype(np.float64)
    n    = Normalizer()
    np.testing.assert_allclose(n.inverse_transform(n.fit_transform(data)), data, rtol=1e-5)


def test_training_history_recorded():
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.surrogate.base import TransolverConfig

    rng    = np.random.default_rng(99)
    coords = rng.random((30, 2)).astype(np.float32)
    params = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                         size=(6, 4)).astype(np.float32)
    out    = [rng.random((30, 1)).astype(np.float32) for _ in range(6)]

    result = SurrogateTrainer(TrainingConfig(
        surrogate_config=TransolverConfig(
            d_model=16, n_heads=2, n_layers=1, slice_num=4,
            epochs=3, patience=10, batch_size=4,
        ),
        use_ensemble=True, n_ensemble=2, train_test_split=0.2, random_seed=1,
    )).train(params, [coords] * 6, out)
    assert result.success
    assert isinstance(result.history, dict)


# ===========================================================================
# 3. SurrogateEvaluator
# ===========================================================================

def test_surrogate_evaluator_analyze_uncertainty(tiny_trainer):
    from piano.surrogate.evaluator import SurrogateEvaluator

    trainer, coords = tiny_trainer
    analysis = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS).analyze_uncertainty(
        coords, n_probe_samples=10, uncertainty_threshold=0.0
    )
    assert hasattr(analysis, "overall_uncertainty")
    assert isinstance(analysis.weak_regions, list)


def test_surrogate_evaluator_on_data(tiny_trainer):
    from piano.surrogate.evaluator import SurrogateEvaluator

    trainer, coords = tiny_trainer
    params      = np.random.default_rng(11).uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(1, 4)
    ).astype(np.float32)
    true_outputs = np.random.default_rng(11).uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)

    metrics = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS).evaluate_on_data(
        params, coords, true_outputs
    )
    assert isinstance(metrics, dict) and len(metrics) > 0


def test_surrogate_evaluator_suggest_samples(tiny_trainer):
    from piano.surrogate.evaluator import SurrogateEvaluator

    trainer, coords = tiny_trainer
    suggestions = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS).suggest_samples_active(
        budget=3, coordinates=coords, acquisition_type="uncertainty"
    )
    assert len(suggestions) <= 3 and all(isinstance(s, dict) for s in suggestions)


def test_weak_region_contains_and_sample():
    from piano.surrogate.evaluator import WeakRegion

    region = WeakRegion(
        parameter_ranges={"E": (190e9, 210e9), "nu": (0.28, 0.32), "load": (80e6, 120e6)},
        metric="uncertainty", metric_value=0.5, priority=1.0,
        sample_count=0, suggested_samples=3,
    )
    assert region.contains({"E": 200e9, "nu": 0.30, "load": 100e6})
    assert not region.contains({"E": 300e9, "nu": 0.30, "load": 100e6})
    samples = region.sample_uniform(n_samples=3)
    assert len(samples) == 3 and all(region.contains(s) for s in samples)


# ===========================================================================
# 4. SpatialErrorAnalyzer + ErrorDecomposer
# ===========================================================================

def test_spatial_error_field(tiny_trainer):
    from piano.surrogate.error_analysis import SpatialErrorAnalyzer

    trainer, coords = tiny_trainer
    rng       = np.random.default_rng(5)
    params    = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                            size=(4,)).astype(np.float32)
    true_vals = rng.uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)
    error_field = SpatialErrorAnalyzer(trainer.model, coords).compute_error_field(params, true_vals)
    assert error_field.shape[0] <= coords.shape[0] and np.all(error_field >= 0)


def test_spatial_error_full_analysis(tiny_trainer):
    from piano.surrogate.error_analysis import SpatialErrorAnalyzer

    trainer, coords = tiny_trainer
    rng       = np.random.default_rng(6)
    params    = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                            size=(4,)).astype(np.float32)
    true_vals = rng.uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)
    analysis  = SpatialErrorAnalyzer(trainer.model, coords).analyze(
        params, true_vals, parameter_names=PARAM_NAMES, hotspot_threshold=80
    )
    assert hasattr(analysis, "error_field") and hasattr(analysis, "hotspots")


def test_error_decomposer(tiny_trainer):
    from piano.surrogate.error_analysis import ErrorDecomposer

    trainer, coords = tiny_trainer
    params    = np.random.default_rng(8).uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(1, 4)
    ).astype(np.float32)
    true_vals = np.random.default_rng(8).uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)
    result = ErrorDecomposer(trainer.model).decompose(params, coords, true_vals)
    assert isinstance(result, dict) and len(result) > 0


# ===========================================================================
# 5. Acquisition functions
# ===========================================================================

def _candidates(n: int = 20) -> np.ndarray:
    return np.random.default_rng(9).uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(n, 4)
    ).astype(np.float32)


def test_uncertainty_sampling(tiny_trainer):
    from piano.surrogate.acquisition import UncertaintySampling

    trainer, coords = tiny_trainer
    result = UncertaintySampling().select_batch(_candidates(), trainer.model, coords, batch_size=3)
    assert len(result.best_indices) == 3 and result.scores.shape[0] == 20


def test_expected_improvement(tiny_trainer):
    from piano.surrogate.acquisition import ExpectedImprovement

    trainer, coords = tiny_trainer
    result = ExpectedImprovement().select_batch(_candidates(), trainer.model, coords, batch_size=3)
    assert len(result.best_indices) == 3


def test_query_by_committee(tiny_trainer):
    from piano.surrogate.acquisition import QueryByCommittee

    trainer, coords = tiny_trainer
    result = QueryByCommittee().select_batch(_candidates(), trainer.model, coords, batch_size=3)
    assert len(result.best_indices) == 3


def test_acquisition_diversity_no_duplicates(tiny_trainer):
    from piano.surrogate.acquisition import UncertaintySampling

    trainer, coords = tiny_trainer
    result = UncertaintySampling().select_batch(
        _candidates(30), trainer.model, coords, batch_size=5, diversity_weight=0.5
    )
    assert len(result.best_indices) == 5
    assert len(set(result.best_indices.tolist())) == 5


def test_acquisition_factory():
    from piano.surrogate.acquisition import get_acquisition_function, AcquisitionType

    for atype in AcquisitionType:
        assert get_acquisition_function(atype) is not None


# ===========================================================================
# 6. PhaseFieldConfig + FEMSample extensions
# ===========================================================================

def test_phase_field_config_defaults():
    from piano.solvers.base import PhaseFieldConfig

    cfg = PhaseFieldConfig()
    assert cfg.G_c == 2.7e3
    assert cfg.l_0 == 0.015
    assert cfg.k_res == 1e-7
    assert cfg.damage_threshold == 0.9


def test_phase_field_config_validation():
    from piano.solvers.base import PhaseFieldConfig

    with pytest.raises(ValueError, match="G_c must be positive"):
        PhaseFieldConfig(G_c=-1.0)
    with pytest.raises(ValueError, match="l_0 must be positive"):
        PhaseFieldConfig(l_0=-0.01)
    with pytest.raises(ValueError, match="k_res must be in"):
        PhaseFieldConfig(k_res=1.5)
    with pytest.raises(ValueError, match="Damage threshold must be in"):
        PhaseFieldConfig(damage_threshold=1.5)


def test_physics_type_phase_field():
    from piano.solvers.base import PhysicsType

    assert hasattr(PhysicsType, "PHASE_FIELD_FRACTURE")


def test_femsample_damage_and_crack_path():
    from piano.data.dataset import FEMSample

    coords = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    sample = FEMSample(
        sample_id="test",
        parameters={"E": 200e9},
        coordinates=coords,
        damage=np.array([0.0, 0.5, 1.0]),
        crack_path=np.array([[0.0, 0.5], [0.3, 0.5], [0.5, 0.6]]),
    )
    assert sample.damage is not None and len(sample.damage) == 3
    assert sample.crack_path is not None and sample.crack_path.shape == (3, 2)
    d = sample.to_dict()
    assert d["has_damage"] is True and d["has_crack_path"] is True


# ===========================================================================
# 7. PhaseFieldGenerator config (no external deps)
# ===========================================================================

@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_phase_field_generator_config():
    from piano.data.phase_field_generator import PhaseFieldFEMConfig, ParameterBounds

    cfg = PhaseFieldFEMConfig()
    assert cfg.geometry_type == "edge_crack"
    assert cfg.domain_width == 1.0

    bounds = ParameterBounds()
    assert bounds.E_range == (150e9, 250e9)
    assert bounds.nu_range == (0.25, 0.35)


# ===========================================================================
# 8. GmshMeshGenerator  (skipif: gmsh)
# ===========================================================================

@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_gmsh_edge_crack_mesh():
    from piano.mesh.gmsh_generator import GmshMeshGenerator, GmshMeshConfig
    from piano.geometry.crack import EdgeCrack

    geometry  = EdgeCrack(crack_length=0.3, width=1.0, height=1.0)
    generator = GmshMeshGenerator(geometry, GmshMeshConfig(base_size=0.05, tip_size=0.01))

    with tempfile.TemporaryDirectory() as tmpdir:
        verts, elems, meta = generator.generate(Path(tmpdir) / "mesh.msh")
        assert verts.shape[1] == 2
        assert elems.shape[1] == 3
        assert meta["crack_type"] == "edge"


# ===========================================================================
# 9. FEniCSPhaseFieldSolver  (skipif: dolfinx)
# ===========================================================================

@pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx not installed")
def test_fenics_solver_instantiation():
    from piano.solvers.fenics_phase_field import FEniCSPhaseFieldSolver

    solver = FEniCSPhaseFieldSolver()
    assert not solver.is_setup
    assert "displacement" in solver.get_available_fields()
    assert "damage" in solver.get_available_fields()
