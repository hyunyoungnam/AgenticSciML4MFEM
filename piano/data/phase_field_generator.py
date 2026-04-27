"""
Phase field fracture dataset generator.

Generates FEM samples using the DOLFINx phase field solver for
training surrogate models on crack propagation problems.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .dataset import FEMSample, FEMDataset, DatasetConfig

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

try:
    from mpi4py import MPI
    import dolfinx
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


@dataclass
class PhaseFieldFEMConfig:
    """
    Configuration for phase field FEM dataset generation.

    Attributes:
        geometry_type: Type of crack geometry ("edge_crack" or "center_crack")
        domain_width: Width of domain (m)
        domain_height: Height of domain (m)
        crack_length: Initial crack length (ratio of width for edge, total for center)
        crack_y: Y-position of crack (ratio of height, 0.5 = center)
        l_0: Regularization length (m)
        G_c: Fracture toughness (J/m²)
        k_res: Residual stiffness
        resolution: Base mesh resolution (elements per unit)
        n_load_steps: Number of load steps
        stagger_tol: Staggered scheme tolerance
        stagger_max_iter: Max iterations per load step
        output_dir: Directory for temporary files
    """
    geometry_type: str = "edge_crack"
    domain_width: float = 1.0
    domain_height: float = 1.0
    crack_length: float = 0.3
    crack_y: float = 0.5
    l_0: float = 0.015
    G_c: float = 2.7e3
    k_res: float = 1e-7
    resolution: int = 50
    n_load_steps: int = 50
    stagger_tol: float = 1e-4
    stagger_max_iter: int = 100
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


@dataclass
class ParameterBounds:
    """
    Parameter bounds for dataset generation.

    Attributes:
        E_range: Young's modulus range (Pa)
        nu_range: Poisson's ratio range
        traction_range: Applied traction range (Pa)
        G_c_range: Fracture toughness range (J/m²)
        crack_length_range: Crack length range (ratio)
    """
    E_range: Tuple[float, float] = (150e9, 250e9)
    nu_range: Tuple[float, float] = (0.25, 0.35)
    traction_range: Tuple[float, float] = (50e6, 150e6)
    G_c_range: Tuple[float, float] = (1e3, 5e3)
    crack_length_range: Tuple[float, float] = (0.2, 0.5)


def _check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    if not HAS_GMSH:
        missing.append("gmsh")
    if not HAS_DOLFINX:
        missing.append("dolfinx")

    if missing:
        raise ImportError(
            f"Missing dependencies for phase field generator: {', '.join(missing)}. "
            "Install with: conda install -c conda-forge fenics-dolfinx gmsh"
        )


def generate_phase_field_sample(
    E: float,
    nu: float,
    traction: float,
    G_c: float,
    config: PhaseFieldFEMConfig,
    crack_length: Optional[float] = None,
    sample_id: Optional[str] = None,
) -> Optional[FEMSample]:
    """
    Generate a single phase field fracture FEM sample.

    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        traction: Applied traction on top boundary (Pa)
        G_c: Fracture toughness (J/m²)
        config: Generation configuration
        crack_length: Override crack length from config
        sample_id: Optional sample ID (generated if not provided)

    Returns:
        FEMSample or None if solver failed
    """
    _check_dependencies()

    from ..geometry.crack import EdgeCrack, CenterCrack
    from ..mesh.gmsh_generator import GmshMeshGenerator, GmshMeshConfig
    from ..mesh.fenics_manager import FEniCSManager
    from ..solvers.base import (
        PhysicsConfig, PhysicsType, MaterialProperties,
        BoundaryCondition, BoundaryConditionType, PhaseFieldConfig
    )
    from ..solvers.fenics_phase_field import FEniCSPhaseFieldSolver

    if sample_id is None:
        sample_id = str(uuid.uuid4())[:8]

    crack_len = crack_length if crack_length is not None else config.crack_length

    # Create geometry
    if config.geometry_type == "edge_crack":
        geometry = EdgeCrack(
            crack_length=crack_len * config.domain_width,
            crack_y=config.crack_y,
            width=config.domain_width,
            height=config.domain_height,
        )
    elif config.geometry_type == "center_crack":
        geometry = CenterCrack(
            crack_length=crack_len * config.domain_width,
            width=config.domain_width,
            height=config.domain_height,
        )
    else:
        raise ValueError(f"Unknown geometry type: {config.geometry_type}")

    # Generate mesh with Gmsh
    mesh_config = GmshMeshConfig(
        base_size=1.0 / config.resolution,
        tip_size=config.l_0 / 5,
        tip_radius=3 * config.l_0,
    )

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("phase_field_mesh")

    try:
        generator = GmshMeshGenerator(geometry, mesh_config)

        # Generate mesh but don't finalize gmsh yet
        W = geometry.width
        H = geometry.height

        # Create domain
        p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_config.base_size)
        p2 = gmsh.model.geo.addPoint(W, 0, 0, mesh_config.base_size)
        p3 = gmsh.model.geo.addPoint(W, H, 0, mesh_config.base_size)
        p4 = gmsh.model.geo.addPoint(0, H, 0, mesh_config.base_size)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([loop])

        # Add crack
        crack_path = geometry.get_crack_path()
        crack_points = []
        for pt in crack_path:
            is_tip = any(np.linalg.norm(pt - tip.position) < 1e-10 for tip in geometry.tips)
            size = mesh_config.tip_size if is_tip else mesh_config.base_size
            crack_points.append(gmsh.model.geo.addPoint(pt[0], pt[1], 0, size))

        crack_lines = []
        for i in range(len(crack_points) - 1):
            crack_lines.append(gmsh.model.geo.addLine(crack_points[i], crack_points[i + 1]))

        gmsh.model.geo.synchronize()

        # Embed crack
        for line in crack_lines:
            gmsh.model.mesh.embed(1, [line], 2, surface)

        # Physical groups
        gmsh.model.addPhysicalGroup(1, [l1], 1, "bottom")
        gmsh.model.addPhysicalGroup(1, [l2], 2, "right")
        gmsh.model.addPhysicalGroup(1, [l3], 3, "top")
        gmsh.model.addPhysicalGroup(1, [l4], 4, "left")
        gmsh.model.addPhysicalGroup(1, crack_lines, 5, "crack")
        gmsh.model.addPhysicalGroup(2, [surface], 1, "domain")

        # Size fields for refinement
        for i, tip in enumerate(geometry.tips):
            tip_pt = gmsh.model.geo.addPoint(tip.position[0], tip.position[1], 0, mesh_config.tip_size)
            gmsh.model.geo.synchronize()

            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "PointsList", [tip_pt])

            thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", mesh_config.tip_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", mesh_config.base_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", mesh_config.tip_radius)

        # Set background field
        if geometry.tips:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", list(range(1, len(geometry.tips) * 2 + 1, 2)))
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # Generate mesh
        gmsh.model.mesh.generate(2)

        # Create DOLFINx mesh from Gmsh model
        from dolfinx.io import gmshio
        mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
        )

    finally:
        gmsh.finalize()

    # Create mesh manager
    mesh_manager = FEniCSManager(mesh=mesh, cell_markers=cell_markers, facet_markers=facet_markers)

    # Set up physics
    material = MaterialProperties(E=E, nu=nu)
    pf_config = PhaseFieldConfig(
        G_c=G_c,
        l_0=config.l_0,
        k_res=config.k_res,
        stagger_tol=config.stagger_tol,
        stagger_max_iter=config.stagger_max_iter,
        n_load_steps=config.n_load_steps,
    )

    # Boundary conditions: fixed bottom, traction on top
    bcs = [
        BoundaryCondition(
            bc_type=BoundaryConditionType.DISPLACEMENT,
            boundary_id=1,  # bottom
            value=np.array([0.0, 0.0]),
        ),
        BoundaryCondition(
            bc_type=BoundaryConditionType.TRACTION,
            boundary_id=3,  # top
            value=np.array([0.0, traction]),
        ),
    ]

    physics = PhysicsConfig(
        physics_type=PhysicsType.PHASE_FIELD_FRACTURE,
        material=material,
        boundary_conditions=bcs,
        phase_field=pf_config,
    )

    # Create and run solver
    solver = FEniCSPhaseFieldSolver()

    try:
        solver.setup(mesh_manager, physics)

        output_dir = config.output_dir or Path("./phase_field_output")
        result = solver.solve(output_dir / sample_id)

        if not result.success:
            return None

        # Create FEMSample
        sample = FEMSample(
            sample_id=sample_id,
            parameters={
                "E": E,
                "nu": nu,
                "traction": traction,
                "G_c": G_c,
                "crack_length": crack_len,
                "l_0": config.l_0,
            },
            coordinates=mesh_manager.get_nodes(),
            displacement=result.solution_data.get("displacement"),
            von_mises=result.solution_data.get("von_mises"),
            damage=result.solution_data.get("damage"),
            crack_path=result.solution_data.get("crack_path"),
            metadata={
                "geometry_type": config.geometry_type,
                "n_load_steps": config.n_load_steps,
                "solve_time": result.solve_time,
                "max_damage": result.metrics.get("max_damage"),
            },
            is_valid=True,
        )

        return sample

    except Exception as e:
        print(f"Solver failed for sample {sample_id}: {e}")
        return None


def generate_phase_field_dataset(
    n_samples: int,
    config: PhaseFieldFEMConfig,
    bounds: Optional[ParameterBounds] = None,
    seed: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Generate a dataset of phase field fracture samples.

    Uses Latin Hypercube Sampling for parameter space coverage.

    Args:
        n_samples: Number of samples to generate
        config: Generation configuration
        bounds: Parameter bounds (uses defaults if None)
        seed: Random seed for reproducibility
        progress_callback: Optional callback(i, n_samples) for progress updates

    Returns:
        Tuple of (parameters, coordinates_list, outputs_list)
        - parameters: (N_valid, n_params) array
        - coordinates_list: List of (N_nodes, dim) arrays
        - outputs_list: List of output dictionaries
    """
    _check_dependencies()

    if bounds is None:
        bounds = ParameterBounds()

    rng = np.random.default_rng(seed)

    # Latin Hypercube Sampling
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=5, seed=seed)
    lhs_samples = sampler.random(n=n_samples)

    # Scale to parameter bounds
    E_samples = bounds.E_range[0] + lhs_samples[:, 0] * (bounds.E_range[1] - bounds.E_range[0])
    nu_samples = bounds.nu_range[0] + lhs_samples[:, 1] * (bounds.nu_range[1] - bounds.nu_range[0])
    traction_samples = bounds.traction_range[0] + lhs_samples[:, 2] * (bounds.traction_range[1] - bounds.traction_range[0])
    G_c_samples = bounds.G_c_range[0] + lhs_samples[:, 3] * (bounds.G_c_range[1] - bounds.G_c_range[0])
    crack_samples = bounds.crack_length_range[0] + lhs_samples[:, 4] * (bounds.crack_length_range[1] - bounds.crack_length_range[0])

    parameters = []
    coordinates_list = []
    outputs_list = []

    for i in range(n_samples):
        if progress_callback:
            progress_callback(i, n_samples)

        sample = generate_phase_field_sample(
            E=E_samples[i],
            nu=nu_samples[i],
            traction=traction_samples[i],
            G_c=G_c_samples[i],
            config=config,
            crack_length=crack_samples[i],
        )

        if sample is not None and sample.is_valid:
            parameters.append([
                sample.parameters["E"],
                sample.parameters["nu"],
                sample.parameters["traction"],
                sample.parameters["G_c"],
                sample.parameters["crack_length"],
            ])
            coordinates_list.append(sample.coordinates)
            outputs_list.append({
                "displacement": sample.displacement,
                "damage": sample.damage,
                "von_mises": sample.von_mises,
                "crack_path": sample.crack_path,
            })

    return np.array(parameters), coordinates_list, outputs_list


def create_phase_field_dataset(
    n_samples: int,
    config: PhaseFieldFEMConfig,
    bounds: Optional[ParameterBounds] = None,
    dataset_name: str = "phase_field_dataset",
    storage_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> FEMDataset:
    """
    Create a complete FEMDataset with phase field samples.

    Args:
        n_samples: Number of samples to generate
        config: Generation configuration
        bounds: Parameter bounds
        dataset_name: Name for the dataset
        storage_dir: Directory to save dataset
        seed: Random seed

    Returns:
        FEMDataset containing all valid samples
    """
    _check_dependencies()

    if bounds is None:
        bounds = ParameterBounds()

    dataset_config = DatasetConfig(
        name=dataset_name,
        parameter_names=["E", "nu", "traction", "G_c", "crack_length"],
        parameter_bounds={
            "E": bounds.E_range,
            "nu": bounds.nu_range,
            "traction": bounds.traction_range,
            "G_c": bounds.G_c_range,
            "crack_length": bounds.crack_length_range,
        },
        output_fields=["displacement", "damage", "von_mises"],
        coordinate_dim=2,
        storage_dir=storage_dir,
    )

    dataset = FEMDataset(dataset_config)

    # Generate samples
    rng = np.random.default_rng(seed)

    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    lhs_samples = sampler.random(n=n_samples)

    E_samples = bounds.E_range[0] + lhs_samples[:, 0] * (bounds.E_range[1] - bounds.E_range[0])
    nu_samples = bounds.nu_range[0] + lhs_samples[:, 1] * (bounds.nu_range[1] - bounds.nu_range[0])
    traction_samples = bounds.traction_range[0] + lhs_samples[:, 2] * (bounds.traction_range[1] - bounds.traction_range[0])
    G_c_samples = bounds.G_c_range[0] + lhs_samples[:, 3] * (bounds.G_c_range[1] - bounds.G_c_range[0])
    crack_samples = bounds.crack_length_range[0] + lhs_samples[:, 4] * (bounds.crack_length_range[1] - bounds.crack_length_range[0])

    for i in range(n_samples):
        print(f"Generating sample {i + 1}/{n_samples}...")

        sample = generate_phase_field_sample(
            E=E_samples[i],
            nu=nu_samples[i],
            traction=traction_samples[i],
            G_c=G_c_samples[i],
            config=config,
            crack_length=crack_samples[i],
        )

        if sample is not None:
            dataset.add_sample(sample)

    print(f"Generated {len(dataset)} valid samples out of {n_samples} attempts")

    if storage_dir:
        dataset.save(storage_dir)

    return dataset
