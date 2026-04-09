"""
tests/test_transolver.py — Comprehensive test for the PIANO SciML pipeline.

Sections:
  1. FEMDataset / FEMSample                    (pure Python)
  2. SurrogateTrainer + EnsembleModel           (pure PyTorch)
  3. SurrogateEvaluator                         (pure PyTorch)
  4. SpatialErrorAnalyzer + ErrorDecomposer     (pure PyTorch)
  5. AcquisitionFunctions                       (pure PyTorch)
  6. MFEMManager                                (mfem marker)
  7. MFEMSolver + EvaluationPipeline            (mfem marker)
  8. End-to-end visualization                   (standalone __main__)

Run:
    pytest tests/test_transolver.py -v
    pytest tests/test_transolver.py -v -m "not mfem"
    python tests/test_transolver.py [--sample N] [--compare]
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN01      = PROJECT_ROOT / "train01"
TRAIN02      = PROJECT_ROOT / "train02"
TRAIN_FINE   = PROJECT_ROOT / "train_fine"

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "E":      (150e9,  250e9),
    "nu":     (0.25,   0.35),
    "load_x": (50e6,   150e6),   # x-traction on right face
    "load_y": (-100e6, 100e6),   # y-traction on top face (0 = uniaxial, ≠0 = biaxial)
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


# ---------------------------------------------------------------------------
# Shared fixture: 2-member ensemble trained on synthetic data (< 5 s)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_trainer():
    """Train a 2-member ensemble on 12 synthetic samples (5 epochs)."""
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.surrogate.base import TransolverConfig

    rng        = np.random.default_rng(0)
    N_SAMPLES  = 12
    N_POINTS   = 64

    params  = rng.uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
        size=(N_SAMPLES, 4)
    ).astype(np.float32)
    coords  = rng.uniform(0, 1, size=(N_POINTS, 2)).astype(np.float32)
    outputs = [
        rng.uniform(0, 1e8, size=(N_POINTS, 1)).astype(np.float32)
        for _ in range(N_SAMPLES)
    ]

    cfg = TrainingConfig(
        surrogate_config=TransolverConfig(
            d_model=32, n_heads=2, n_layers=1, slice_num=4,
            epochs=5, patience=10, batch_size=4,
        ),
        use_ensemble=True,
        n_ensemble=2,
        train_test_split=0.2,
        random_seed=0,
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
            coordinates=coords,
            von_mises=vm,
            is_valid=True,
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
                "nu":     float(rng.uniform(0.25, 0.35)),
                "load_x": float(rng.uniform(50e6, 150e6)),
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
        coordinates=coords,
        von_mises=vm,
        is_valid=True,
    )

    field = sample.get_output_field("von_mises")
    assert field is not None and field.shape == (20, 1)

    pvec = sample.get_parameter_vector(PARAM_NAMES)
    assert pvec.shape == (4,)
    assert pvec[0] == pytest.approx(200e9)


# ===========================================================================
# 2. SurrogateTrainer + EnsembleModel
# ===========================================================================

def test_surrogate_trainer_success(tiny_trainer):
    trainer, _ = tiny_trainer
    assert trainer.model is not None
    assert trainer.model.is_trained


def test_surrogate_prediction_shape(tiny_trainer):
    trainer, coords = tiny_trainer
    rng    = np.random.default_rng(7)
    params = rng.uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(1, 4)
    ).astype(np.float32)

    mean, unc = trainer.predict_with_uncertainty(params, coords)
    assert mean.shape[0] == coords.shape[0]
    assert unc is not None
    assert unc.shape[0] == coords.shape[0]


def test_normalizer_roundtrip():
    from piano.surrogate.trainer import Normalizer

    rng  = np.random.default_rng(3)
    data = rng.uniform(0, 1000, size=(100, 3)).astype(np.float64)
    n    = Normalizer()
    np.testing.assert_allclose(n.inverse_transform(n.fit_transform(data)),
                                data, rtol=1e-5)


def test_training_history_recorded(tiny_trainer):
    """TrainingResult must contain a history dict with train_losses."""
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.surrogate.base import TransolverConfig

    rng    = np.random.default_rng(99)
    coords = rng.random((30, 2)).astype(np.float32)
    params = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                         size=(6, 4)).astype(np.float32)
    out    = [rng.random((30, 1)).astype(np.float32) for _ in range(6)]

    cfg = TrainingConfig(
        surrogate_config=TransolverConfig(
            d_model=16, n_heads=2, n_layers=1, slice_num=4,
            epochs=3, patience=10, batch_size=4,
        ),
        use_ensemble=True, n_ensemble=2,
        train_test_split=0.2, random_seed=1,
    )
    result = SurrogateTrainer(cfg).train(params, [coords] * 6, out)
    assert result.success
    assert isinstance(result.history, dict)


# ===========================================================================
# 3. SurrogateEvaluator
# ===========================================================================

def test_surrogate_evaluator_analyze_uncertainty(tiny_trainer):
    from piano.surrogate.evaluator import SurrogateEvaluator

    trainer, coords = tiny_trainer
    evaluator = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS)
    analysis  = evaluator.analyze_uncertainty(
        coords, n_probe_samples=10, uncertainty_threshold=0.0
    )
    assert hasattr(analysis, "overall_uncertainty")
    assert hasattr(analysis, "weak_regions")
    assert isinstance(analysis.weak_regions, list)


def test_surrogate_evaluator_on_data(tiny_trainer):
    from piano.surrogate.evaluator import SurrogateEvaluator

    trainer, coords = tiny_trainer
    rng         = np.random.default_rng(11)
    params      = rng.uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(1, 4)
    ).astype(np.float32)
    true_outputs = rng.uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)

    evaluator = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS)
    metrics   = evaluator.evaluate_on_data(params, coords, true_outputs)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_surrogate_evaluator_suggest_samples(tiny_trainer):
    from piano.surrogate.evaluator import SurrogateEvaluator

    trainer, coords = tiny_trainer
    evaluator   = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS)
    suggestions = evaluator.suggest_samples_active(
        budget=3, coordinates=coords, acquisition_type="uncertainty"
    )
    assert len(suggestions) <= 3
    assert all(isinstance(s, dict) for s in suggestions)


def test_weak_region_contains_and_sample():
    from piano.surrogate.evaluator import WeakRegion

    region = WeakRegion(
        parameter_ranges={"E": (190e9, 210e9), "nu": (0.28, 0.32), "load": (80e6, 120e6)},
        metric="uncertainty",
        metric_value=0.5,
        priority=1.0,
        sample_count=0,
        suggested_samples=3,
    )
    assert region.contains({"E": 200e9, "nu": 0.30, "load": 100e6})
    assert not region.contains({"E": 300e9, "nu": 0.30, "load": 100e6})

    samples = region.sample_uniform(n_samples=3)
    assert len(samples) == 3
    assert all(region.contains(s) for s in samples)


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

    analyzer    = SpatialErrorAnalyzer(trainer.model, coords)
    error_field = analyzer.compute_error_field(params, true_vals)

    assert error_field.shape[0] <= coords.shape[0]
    assert np.all(error_field >= 0)


def test_spatial_error_hotspots(tiny_trainer):
    from piano.surrogate.error_analysis import SpatialErrorAnalyzer

    trainer, coords = tiny_trainer
    rng       = np.random.default_rng(14)
    params    = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                             size=(4,)).astype(np.float32)
    true_vals = rng.uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)

    analyzer = SpatialErrorAnalyzer(trainer.model, coords)
    error_field = analyzer.compute_error_field(params, true_vals)
    hotspots    = analyzer.identify_hotspots(error_field, threshold_percentile=80)
    assert isinstance(hotspots, list)


def test_spatial_error_full_analysis(tiny_trainer):
    from piano.surrogate.error_analysis import SpatialErrorAnalyzer

    trainer, coords = tiny_trainer
    rng       = np.random.default_rng(6)
    params    = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                             size=(4,)).astype(np.float32)
    true_vals = rng.uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)

    analysis = SpatialErrorAnalyzer(trainer.model, coords).analyze(
        params, true_vals, parameter_names=PARAM_NAMES, hotspot_threshold=80
    )
    assert hasattr(analysis, "error_field")
    assert hasattr(analysis, "hotspots")
    assert hasattr(analysis, "global_stats")


def test_error_decomposer(tiny_trainer):
    from piano.surrogate.error_analysis import ErrorDecomposer

    trainer, coords = tiny_trainer
    rng       = np.random.default_rng(8)
    params    = rng.uniform([150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6],
                             size=(1, 4)).astype(np.float32)
    true_vals = rng.uniform(0, 1e8, size=(coords.shape[0],)).astype(np.float32)

    result = ErrorDecomposer(trainer.model).decompose(params, coords, true_vals)
    assert isinstance(result, dict)
    assert len(result) > 0


# ===========================================================================
# 5. Acquisition functions
# ===========================================================================

def _make_candidates(rng: np.random.Generator, n: int = 20) -> np.ndarray:
    return rng.uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(n, 4)
    ).astype(np.float32)


def test_uncertainty_sampling(tiny_trainer):
    from piano.surrogate.acquisition import UncertaintySampling

    trainer, coords = tiny_trainer
    candidates = _make_candidates(np.random.default_rng(9))

    result = UncertaintySampling().select_batch(
        candidates, trainer.model, coords, batch_size=3
    )
    assert len(result.best_indices) == 3
    assert result.scores.shape[0] == len(candidates)


def test_expected_improvement(tiny_trainer):
    from piano.surrogate.acquisition import ExpectedImprovement

    trainer, coords = tiny_trainer
    candidates = _make_candidates(np.random.default_rng(10))

    result = ExpectedImprovement().select_batch(
        candidates, trainer.model, coords, batch_size=3
    )
    assert len(result.best_indices) == 3


def test_query_by_committee(tiny_trainer):
    from piano.surrogate.acquisition import QueryByCommittee

    trainer, coords = tiny_trainer
    candidates = _make_candidates(np.random.default_rng(12))

    result = QueryByCommittee().select_batch(
        candidates, trainer.model, coords, batch_size=3
    )
    assert len(result.best_indices) == 3


def test_acquisition_diversity_no_duplicates(tiny_trainer):
    from piano.surrogate.acquisition import UncertaintySampling

    trainer, coords = tiny_trainer
    candidates = _make_candidates(np.random.default_rng(13), n=30)

    result = UncertaintySampling().select_batch(
        candidates, trainer.model, coords,
        batch_size=5, diversity_weight=0.5
    )
    assert len(result.best_indices) == 5
    assert len(set(result.best_indices.tolist())) == 5


def test_acquisition_factory():
    from piano.surrogate.acquisition import get_acquisition_function, AcquisitionType

    for atype in AcquisitionType:
        acq = get_acquisition_function(atype)
        assert acq is not None


# ===========================================================================
# 6. MFEMManager  (mfem marker)
# ===========================================================================

@pytest.mark.mfem
def test_mfem_manager_load():
    pytest.importorskip("mfem.ser")
    from piano.mesh.mfem_manager import MFEMManager

    mgr = MFEMManager(str(TRAIN01 / "sample_000.mesh"))
    assert mgr.num_nodes > 0
    assert mgr.num_elements > 0
    assert mgr.dimension == 2
    assert mgr.get_nodes().shape[1] == 2
    assert len(mgr.get_elements()) == mgr.num_elements


@pytest.mark.mfem
def test_mfem_manager_save(tmp_path):
    pytest.importorskip("mfem.ser")
    from piano.mesh.mfem_manager import MFEMManager

    mgr   = MFEMManager(str(TRAIN01 / "sample_001.mesh"))
    saved = mgr.save(str(tmp_path / "copy.mesh"))
    assert saved.exists()


@pytest.mark.mfem
def test_mfem_manager_boundary_attributes():
    pytest.importorskip("mfem.ser")
    from piano.mesh.mfem_manager import MFEMManager

    mgr   = MFEMManager(str(TRAIN01 / "sample_002.mesh"))
    attrs = mgr.get_boundary_attributes()
    assert attrs.ndim == 1
    assert len(attrs) > 0


@pytest.mark.mfem
def test_train_fine_mesh_loads():
    """train_fine meshes (scipy Delaunay) must load correctly and have right boundary tags."""
    pytest.importorskip("mfem.ser")
    import mfem.ser as mfem
    from collections import Counter

    assert TRAIN_FINE.exists(), f"train_fine/ not found at {TRAIN_FINE}"

    for stem in ["sample_000.mesh", "sample_050.mesh", "sample_099.mesh"]:
        path = TRAIN_FINE / stem
        m    = mfem.Mesh(str(path), 1, 1)
        assert m.GetNV() > 400,  f"{stem}: too few nodes ({m.GetNV()})"
        assert m.GetNE() > 700,  f"{stem}: too few elements ({m.GetNE()})"

        tags = Counter(m.GetBdrAttribute(i) for i in range(m.GetNBE()))
        # Each outer side ≥ 8 edges; hole ≥ 20 edges
        for side in (1, 2, 3, 4):
            assert tags.get(side, 0) >= 8, \
                f"{stem}: boundary tag {side} has only {tags.get(side,0)} edges"
        assert tags.get(5, 0) >= 20, \
            f"{stem}: hole tag (5) has only {tags.get(5,0)} edges"


# ===========================================================================
# 7. MFEMSolver + EvaluationPipeline  (mfem marker)
# ===========================================================================

def _physics_config(E: float = 200e9, nu: float = 0.3,
                    load_x: float = 1e8, load_y: float = 0.0):
    """Build PhysicsConfig for 2-D linear elasticity.

    load_x: x-traction on right face  (boundary 2)
    load_y: y-traction on top face    (boundary 3) — 0 = uniaxial, ≠0 = biaxial
    """
    from piano.solvers.base import (
        PhysicsConfig, PhysicsType, MaterialProperties,
        BoundaryCondition, BoundaryConditionType,
    )
    bcs = [
        BoundaryCondition(BoundaryConditionType.SYMMETRY,
                          boundary_id=4, direction=0),   # left:   ux = 0
        BoundaryCondition(BoundaryConditionType.SYMMETRY,
                          boundary_id=1, direction=1),   # bottom: uy = 0
        BoundaryCondition(BoundaryConditionType.TRACTION,
                          boundary_id=2, value=np.array([load_x, 0.])),
    ]
    if load_y != 0.0:
        bcs.append(BoundaryCondition(BoundaryConditionType.TRACTION,
                                     boundary_id=3, value=np.array([0., load_y])))
    return PhysicsConfig(
        physics_type=PhysicsType.LINEAR_ELASTICITY,
        material=MaterialProperties(E=E, nu=nu),
        boundary_conditions=bcs,
    )


def _retag_boundaries(mfem_mesh, verts: np.ndarray) -> None:
    """Tag: bottom=1, right=2, top=3, left=4, hole=5."""
    eps = 1e-10
    for i in range(mfem_mesh.GetNBE()):
        iv = mfem_mesh.GetBdrElement(i).GetVerticesArray()
        xs = [verts[iv[j]][0] for j in range(len(iv))]
        ys = [verts[iv[j]][1] for j in range(len(iv))]
        if   all(y < eps for y in ys):        tag = 1
        elif all(x > 1.0 - eps for x in xs): tag = 2
        elif all(y > 1.0 - eps for y in ys): tag = 3
        elif all(x < eps for x in xs):        tag = 4
        else:                                  tag = 5
        mfem_mesh.GetBdrElement(i).SetAttribute(tag)
    mfem_mesh.SetAttributes()


def _extract_verts(mfem_mesh) -> np.ndarray:
    import ctypes
    nv    = mfem_mesh.GetNV()
    verts = np.zeros((nv, 2))
    for i in range(nv):
        v = mfem_mesh.GetVertex(i)
        p = ctypes.cast(int(v), ctypes.POINTER(ctypes.c_double))
        verts[i] = [p[0], p[1]]
    return verts


@pytest.mark.mfem
def test_mfem_solver_elasticity(tmp_path):
    pytest.importorskip("mfem.ser")
    from piano.mesh.mfem_manager import MFEMManager
    from piano.solvers.mfem_solver import MFEMSolver

    mgr   = MFEMManager(str(TRAIN01 / "sample_000.mesh"))
    verts = _extract_verts(mgr.mesh)
    _retag_boundaries(mgr.mesh, verts)

    solver = MFEMSolver(order=1)
    solver.setup(mgr, _physics_config())
    result = solver.solve(str(tmp_path))

    assert result.success, result.error_message
    vm = result.solution_data.get("von_mises")
    assert vm is not None and vm.shape[0] == mgr.num_elements
    assert np.any(vm > 0), "All-zero von Mises — solver did not converge"


@pytest.mark.mfem
def test_evaluation_pipeline_quick():
    pytest.importorskip("mfem.ser")
    from piano.mesh.mfem_manager import MFEMManager
    from piano.evaluation.pipeline import EvaluationPipeline, EvaluationStage

    mgr    = MFEMManager(str(TRAIN01 / "sample_002.mesh"))
    result = EvaluationPipeline(run_solver=False).quick_evaluate("q001", mgr)

    assert result.stage   == EvaluationStage.COMPLETE
    assert result.success
    assert result.preflight_result is not None
    assert result.overall_score    > 0


@pytest.mark.mfem
def test_evaluation_pipeline_with_solver(tmp_path):
    pytest.importorskip("mfem.ser")
    from piano.mesh.mfem_manager import MFEMManager
    from piano.solvers.mfem_solver import MFEMSolver
    from piano.evaluation.pipeline import EvaluationPipeline, EvaluationStage

    mgr   = MFEMManager(str(TRAIN01 / "sample_003.mesh"))
    verts = _extract_verts(mgr.mesh)
    _retag_boundaries(mgr.mesh, verts)

    pipeline = EvaluationPipeline(
        run_solver=True,
        solver=MFEMSolver(order=1),
        physics=_physics_config(),
    )
    result = pipeline.evaluate("s003", mgr, output_dir=str(tmp_path))

    assert result.stage == EvaluationStage.COMPLETE
    assert result.solver_completed


# ===========================================================================
# 8. End-to-end visualization  (standalone __main__)
# ===========================================================================

import argparse


def _collect_mesh_files(dirs: List[str], limit: Optional[int] = None) -> List[Path]:
    files = []
    for d in dirs:
        files.extend(sorted(Path(d).glob("sample_*.mesh")))
    return files if limit is None else files[:limit]


def _run_fem(mesh_path: str, params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Run MFEM solver; returns (displacement, von_mises, vertices, elements).

    displacement : (N_nodes, 2)  — nodal ux, uy  (training target + PINO input)
    von_mises    : (N_elements,) — element-centre von Mises (GT reference only)
    vertices     : (N_nodes, 2)  — node coordinates
    elements     : list of element connectivity
    """
    import ctypes
    from piano.mesh.mfem_manager import MFEMManager
    from piano.solvers.mfem_solver import MFEMSolver

    mgr   = MFEMManager(mesh_path)
    nv    = mgr.mesh.GetNV()
    verts = np.zeros((nv, 2))
    for i in range(nv):
        v = mgr.mesh.GetVertex(i)
        p = ctypes.cast(int(v), ctypes.POINTER(ctypes.c_double))
        verts[i] = [p[0], p[1]]
    _retag_boundaries(mgr.mesh, verts)

    solver = MFEMSolver(order=1)
    solver.setup(mgr, _physics_config(
        E=params.get("E", 200e9), nu=params.get("nu", 0.3),
        load_x=params.get("load_x", params.get("load", 1e8)),
        load_y=params.get("load_y", 0.0),
    ))
    with tempfile.TemporaryDirectory() as tmp:
        result = solver.solve(tmp)

    vertices = mgr.get_nodes()
    elements = [list(e[e >= 0]) for e in mgr.get_elements()]
    n_nodes  = len(vertices)
    n_elems  = len(elements)
    if not result.success:
        return (np.zeros((n_nodes, 2)),
                np.zeros(n_elems),
                vertices, elements)
    disp = result.solution_data.get("displacement", np.zeros((n_nodes, 2)))
    vm   = result.solution_data.get("von_mises",    np.zeros(n_elems))
    return disp, vm, vertices, elements


def _build_dataset(mesh_files: List[Path], rng: np.random.Generator):
    """Run FEM on mesh_files; train target is displacement (ux, uy) at nodes.

    Stores FEMSample.displacement = (N_nodes, 2) so that
    prepare_training_data("displacement") gives output_dim=2, enabling PINO.
    """
    from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig

    ds = FEMDataset(DatasetConfig(parameter_names=PARAM_NAMES,
                                   parameter_bounds=PARAM_BOUNDS))

    for i, mf in enumerate(mesh_files):
        params = {
            "E":      float(rng.uniform(150e9, 250e9)),
            "nu":     float(rng.uniform(0.25, 0.35)),
            "load_x": float(rng.uniform(50e6,  150e6)),
            "load_y": float(rng.uniform(-100e6, 100e6)),
        }
        try:
            disp, vm, verts, _ = _run_fem(str(mf), params)
            if not np.any(disp != 0):
                raise RuntimeError("trivial solution")
            ds.add_sample(FEMSample(
                sample_id=f"s{i:03d}",
                parameters=params,
                coordinates=verts.astype(np.float32),       # node coords
                displacement=disp.astype(np.float32),        # (N_nodes, 2)
                is_valid=True,
            ))
            print(f"  [{i+1}/{len(mesh_files)}] {mf.stem}  "
                  f"|u|_max={np.linalg.norm(disp, axis=1).max():.3e}  "
                  f"VM_max={vm.max():.2e}")
        except Exception as e:
            print(f"  [{i+1}/{len(mesh_files)}] {mf.stem} skipped: {e}")

    return ds


def _train_ensemble(ds, n_ensemble: int = 3, epochs: int = 80):
    """Train ensemble on displacement field (ux, uy) — enables PINO loss.

    output_dim=2  →  trainer activates PINO (equilibrium + energy-norm terms).
    Capacity raised to d_model=128 / n_layers=4; LR lowered to 5e-4.
    """
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.surrogate.base import TransolverConfig

    params_arr, coords_list, outputs_list = ds.prepare_training_data("displacement")
    cfg = TrainingConfig(
        surrogate_config=TransolverConfig(
            d_model=128, n_heads=8, n_layers=4, slice_num=16,
            epochs=epochs, patience=50, batch_size=8,
            learning_rate=5e-4,
            pino_weight=0.1, pino_eq_weight=0.1,
        ),
        use_ensemble=True,
        n_ensemble=n_ensemble,
        train_test_split=0.2,
        random_seed=42,
    )
    trainer = SurrogateTrainer(cfg)
    result  = trainer.train(params_arr, coords_list, outputs_list)
    if not result.success:
        raise RuntimeError(f"Ensemble training failed: {result.error_message}")
    print(f"  train_loss={result.train_loss:.4f}  "
          f"test_loss={result.test_loss:.4f}")
    return trainer, result


def run_visualization(
    samples_dirs: List[str],
    sample_idx:   int,
    output_file:  str,
    n_train:      int = -1,
    n_ensemble:   int = 3,
    epochs:       int = 80,
    n_test:       int = 20,
) -> None:
    """
    5-panel SciML loop visualization in a 2-row layout.

    Row 1 — plate-with-hole domain:
      ① Ensemble mean prediction
      ② Ensemble uncertainty
      ③ Error field with hotspots

    Row 2 — analytics:
      ④ Acquisition scores (sorted bar chart, top-5 highlighted)
      ⑤ Training convergence (train loss + test loss vs epoch)

    The last n_test files are held out as a strictly separate test set —
    the Transolver never sees them during training.  n_train=-1 means
    "use all available training samples" (everything except the test set).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import PolyCollection
    import matplotlib.colors as mcolors
    from piano.surrogate.evaluator import SurrogateEvaluator
    from piano.surrogate.error_analysis import SpatialErrorAnalyzer
    from piano.surrogate.acquisition import UncertaintySampling

    rng       = np.random.default_rng(42)
    all_files = _collect_mesh_files(samples_dirs)
    if not all_files:
        raise FileNotFoundError(f"No sample_*.mesh found in {samples_dirs}")

    # ── Strict train / test split ────────────────────────────────────────
    # Hold out last n_test files; they are NEVER included in training.
    n_test_actual = min(n_test, max(1, len(all_files) // 5))
    test_pool     = all_files[-n_test_actual:]
    train_pool    = all_files[:-n_test_actual]

    train_files = train_pool if n_train < 0 else train_pool[:n_train]
    test_file   = test_pool[sample_idx % len(test_pool)]

    dirs_label = " + ".join(Path(d).name for d in samples_dirs)
    print(f"Training dirs : {dirs_label}")
    print(f"  total files : {len(all_files)}  "
          f"→  train pool: {len(train_pool)}  |  test pool: {len(test_pool)}")
    print(f"  training on : {len(train_files)} meshes")
    print(f"Test mesh     : {test_file.name}  (held-out, never seen during training)")

    # ── Dataset & training ────────────────────────────────────────────────
    ds = _build_dataset(train_files, rng)
    valid = ds.get_valid_samples()
    if len(valid) < 3:
        raise RuntimeError(f"Need ≥3 valid FEM samples, got {len(valid)}")
    print(f"Training {n_ensemble}-member ensemble ({len(valid)} valid samples)…")
    trainer, train_result = _train_ensemble(ds, n_ensemble=n_ensemble, epochs=epochs)

    # ── Test mesh FEM + prediction ────────────────────────────────────────
    test_params = {
        "E":      float(rng.uniform(150e9, 250e9)),
        "nu":     float(rng.uniform(0.25, 0.35)),
        "load_x": float(rng.uniform(50e6,  150e6)),
        "load_y": float(rng.uniform(-100e6, 100e6)),
    }
    # disp_gt: (N_nodes, 2) — GT displacement at nodes
    # vm_gt:   (N_elements,) — GT von Mises at element centres (reference only)
    disp_gt, vm_gt, verts, elems = _run_fem(str(test_file), test_params)
    param_arr = np.array([[test_params[k] for k in PARAM_NAMES]],
                         dtype=np.float32)

    # Predict: mean_pred (N_nodes, 2), unc (N_nodes, 2) or None
    mean_pred, unc_raw = trainer.predict_with_uncertainty(param_arr, verts.astype(np.float32))
    # mean_pred may come back as (1, N_nodes, 2) or (N_nodes, 2)
    if mean_pred.ndim == 3:
        mean_pred = mean_pred[0]           # (N_nodes, 2)

    # Displacement magnitude at nodes for display
    disp_mag_pred = np.linalg.norm(mean_pred, axis=-1)    # (N_nodes,)
    disp_mag_gt   = np.linalg.norm(disp_gt,   axis=1)     # (N_nodes,)

    # Uncertainty: norm of per-component std across ensemble → scalar per node
    if unc_raw is not None:
        if unc_raw.ndim == 3:
            unc_raw = unc_raw[0]
        unc_nodes = np.linalg.norm(unc_raw, axis=-1) if unc_raw.ndim == 2 else unc_raw.flatten()
    else:
        unc_nodes = np.zeros(len(verts))

    # Interpolate nodal scalar fields to element centres for PolyCollection display
    def _node_to_elem(field_nodes: np.ndarray) -> np.ndarray:
        return np.array([field_nodes[e].mean() for e in elems])

    disp_mag_pred_e = _node_to_elem(disp_mag_pred)
    unc_e           = _node_to_elem(unc_nodes)
    error_nodes     = np.abs(disp_mag_pred - disp_mag_gt)
    error_e         = _node_to_elem(error_nodes)

    # Normalise to [0, 1] for clean colour display
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-10)

    unc_norm  = _norm(unc_e)

    # ── Surrogate evaluator ───────────────────────────────────────────────
    evaluator = SurrogateEvaluator(trainer.model, PARAM_NAMES, PARAM_BOUNDS)
    analysis  = evaluator.analyze_uncertainty(
        verts.astype(np.float32), n_probe_samples=30,
        uncertainty_threshold=float(np.mean(unc_norm))
    )

    # ── Acquisition scores ────────────────────────────────────────────────
    N_CAND      = 50
    candidates  = rng.uniform(
        [150e9, 0.25, 50e6, -100e6], [250e9, 0.35, 150e6, 100e6], size=(N_CAND, 4)
    ).astype(np.float32)
    acq_result  = UncertaintySampling().select_batch(
        candidates, trainer.model, verts.astype(np.float32), batch_size=5
    )
    scores      = acq_result.scores
    scores_norm = _norm(scores)
    sort_idx    = np.argsort(scores_norm)
    top5_set    = set(acq_result.best_indices.tolist())

    # ── Spatial error analysis (scalar displacement magnitude error) ───────
    analyzer    = SpatialErrorAnalyzer(trainer.model, verts.astype(np.float32))
    error_field = analyzer.compute_error_field(param_arr.flatten(), disp_mag_gt)
    err_norm    = _norm(error_e)          # use element-level field for display
    hotspots    = analyzer.identify_hotspots(error_field, threshold_percentile=85)

    # ── Training history ─────────────────────────────────────────────────
    history     = train_result.history      # keys: train_loss, test_loss
    train_loss  = history.get("train_loss", [])
    test_loss   = history.get("test_loss",  [])
    epochs_ax   = np.arange(1, len(train_loss) + 1)

    # ── Figure layout (2 rows × 3 cols) ──────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42, wspace=0.35,
        left=0.06, right=0.97, top=0.91, bottom=0.08,
    )
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_unc  = fig.add_subplot(gs[0, 1])
    ax_err  = fig.add_subplot(gs[0, 2])
    ax_acq  = fig.add_subplot(gs[1, 0:2])   # acquisition spans 2 cols
    ax_conv = fig.add_subplot(gs[1, 2])

    fig.suptitle(
        f"PIANO SciML Loop  ·  test: {test_file.stem}  ·  "
        f"training: {dirs_label}  ({len(valid)} samples, {len(epochs_ax)} epochs)",
        fontsize=11, fontweight="bold",
    )

    # ── Helper: plot a 2-D mesh field ─────────────────────────────────────
    def _mesh_field(ax, field, title, cmap, vmin=0.0, vmax=1.0):
        polys = [verts[e] for e in elems]
        v0 = float(field.min()) if vmin is None else vmin
        v1 = max(float(field.max()), v0 + 1e-10) if vmax is None else vmax
        pc = PolyCollection(
            polys, array=field, cmap=cmap,
            norm=mcolors.Normalize(vmin=v0, vmax=v1),
            edgecolors="k", linewidths=0.15,
        )
        ax.add_collection(pc)
        ax.set_xlim(verts[:, 0].min() - .04, verts[:, 0].max() + .04)
        ax.set_ylim(verts[:, 1].min() - .04, verts[:, 1].max() + .04)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
        return pc

    # ① Ensemble mean displacement magnitude (normalised)
    mean_norm = _norm(disp_mag_pred_e)
    c1 = _mesh_field(ax_mean, mean_norm,
                     f"① Ensemble Mean  |u|\n({len(elems)} elems  ·  "
                     f"peak={disp_mag_pred.max():.3e} m)",
                     "jet")
    cb1 = fig.colorbar(c1, ax=ax_mean, shrink=0.85)
    cb1.set_label("Norm. |u| predicted", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    ax_mean.text(0.02, 0.02,
                 f"E={test_params['E']/1e9:.0f} GPa  ν={test_params['nu']:.3f}\n"
                 f"Fx={test_params['load_x']/1e6:.0f} MPa\n"
                 f"Fy={test_params['load_y']/1e6:.0f} MPa",
                 transform=ax_mean.transAxes, fontsize=7, va="bottom",
                 bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.85))

    # ② Uncertainty: norm of ensemble std across displacement components
    c2 = _mesh_field(ax_unc, unc_norm,
                     f"② Ensemble Uncertainty\n"
                     f"(weak regions: {len(analysis.weak_regions)})",
                     "hot_r")
    cb2 = fig.colorbar(c2, ax=ax_unc, shrink=0.85)
    cb2.set_label("Norm. ‖σ_ens‖", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    ax_unc.text(0.02, 0.02,
                f"mean={np.mean(unc_norm):.3f}\nmax={unc_norm.max():.3f}",
                transform=ax_unc.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    # ③ Error field: |pred |u| − GT |u|| + hotspot markers
    c3 = _mesh_field(ax_err, err_norm,
                     f"③ Error  ||u|_pred − |u|_GT|\n"
                     f"({len(hotspots)} hotspots at 85th pct)",
                     "Reds")
    cb3 = fig.colorbar(c3, ax=ax_err, shrink=0.85)
    cb3.set_label("Norm. |u| error", fontsize=8)
    cb3.ax.tick_params(labelsize=7)
    for hs in hotspots[:6]:
        ax_err.plot(hs.center[0], hs.center[1], "b+", ms=9, mew=1.8, zorder=5)
    ax_err.text(0.02, 0.02,
                f"mean={np.mean(err_norm):.3f}\nmax={err_norm.max():.3f}",
                transform=ax_err.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.85))

    # ④ Acquisition scores — sorted bar chart (low → high left → right)
    bar_colors = [
        "#d62728" if sort_idx[i] in top5_set else "#aec7e8"
        for i in range(len(sort_idx))
    ]
    ax_acq.bar(np.arange(N_CAND), scores_norm[sort_idx],
               color=bar_colors, width=0.85, zorder=2)
    ax_acq.axhline(scores_norm[sort_idx[-5]], color="#d62728",
                   ls="--", lw=1.2, label="top-5 threshold")
    ax_acq.set_xlabel("Candidate index (sorted by score)", fontsize=9)
    ax_acq.set_ylabel("Normalised acquisition score", fontsize=9)
    ax_acq.set_title("④ Acquisition Scores  (UncertaintySampling, 50 candidates)",
                     fontsize=9, fontweight="bold")
    ax_acq.tick_params(labelsize=8)
    ax_acq.set_xlim(-0.5, N_CAND - 0.5)
    ax_acq.set_ylim(0, 1.08)
    ax_acq.grid(axis="y", alpha=0.3, zorder=1)
    ax_acq.legend(fontsize=8)
    # Annotate top-5 bars with E / ν values
    for rank, raw_idx in enumerate(sort_idx[-5:]):
        bar_x = list(sort_idx).index(raw_idx)   # position in sorted chart
        ax_acq.text(
            bar_x, scores_norm[raw_idx] + 0.02,
            f"E={candidates[raw_idx,0]/1e9:.0f}\nν={candidates[raw_idx,1]:.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#d62728",
        )

    # ⑤ Training convergence
    if len(train_loss) > 0:
        ax_conv.semilogy(epochs_ax, train_loss, lw=1.8,
                         color="#1f77b4", label="Train loss")
    if len(test_loss) > 0:
        ax_conv.semilogy(epochs_ax, test_loss,  lw=1.8,
                         color="#ff7f0e", ls="--", label="Test loss")
    ax_conv.set_xlabel("Epoch", fontsize=9)
    ax_conv.set_ylabel("MSE loss (log scale)", fontsize=9)
    ax_conv.set_title("⑤ Training Convergence", fontsize=9, fontweight="bold")
    ax_conv.tick_params(labelsize=8)
    ax_conv.legend(fontsize=8)
    ax_conv.grid(alpha=0.3)
    if len(train_loss) > 0:
        ax_conv.text(0.97, 0.97,
                     f"final train: {train_loss[-1]:.4f}\n"
                     f"final test : {test_loss[-1]:.4f}" if test_loss else
                     f"final train: {train_loss[-1]:.4f}",
                     transform=ax_conv.transAxes, fontsize=7.5,
                     ha="right", va="top",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_file}")
    print(f"\nSummary:")
    print(f"  Weak regions  : {len(analysis.weak_regions)}")
    print(f"  Error hotspots: {len(hotspots)}")
    print(f"  Top-5 acq idx : {acq_result.best_indices.tolist()}")
    if train_loss:
        print(f"  Final train / test loss: "
              f"{train_loss[-1]:.4f} / {test_loss[-1] if test_loss else 'n/a':.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PIANO SciML pipeline — uncertainty-driven visualization"
    )
    parser.add_argument("--sample",       type=int, default=0,
                        help="Test mesh index within the held-out test pool (default 0)")
    parser.add_argument("--samples-dirs", type=str, nargs="+", default=None,
                        help="Mesh dirs — pass multiple to combine datasets "
                             "(e.g. train_fine train01 train02)")
    parser.add_argument("--output",       type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--n-train",      type=int, default=-1,
                        help="Training meshes to use (-1 = all available, default)")
    parser.add_argument("--n-test",       type=int, default=20,
                        help="Held-out test meshes kept separate from training (default 20)")
    parser.add_argument("--epochs",       type=int, default=300,
                        help="Ensemble training epochs (default 300)")
    args = parser.parse_args()

    # Default: train_fine if available, else fall back to train01 + train02
    default_dirs = ([str(TRAIN_FINE)] if TRAIN_FINE.exists()
                    else [str(TRAIN01), str(TRAIN02)])
    dirs       = args.samples_dirs or default_dirs
    output_dir = PROJECT_ROOT / "tests" / "test_outputs"
    out        = args.output or str(output_dir / "sciml_loop.png")

    try:
        import mfem.ser  # noqa: F401
    except ImportError:
        print("mfem not available — visualization requires MFEM. "
              "Run pytest for pure-Python tests.")
        sys.exit(0)

    run_visualization(
        samples_dirs=dirs,
        sample_idx=args.sample,
        output_file=out,
        n_train=args.n_train,
        epochs=args.epochs,
        n_test=args.n_test,
    )
