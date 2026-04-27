"""
FEM dataset structures.

Defines data structures for storing and managing FEM simulation
samples used for training surrogate models.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


@dataclass
class FEMSample:
    """
    A single FEM simulation sample.

    Represents one simulation with input parameters, mesh coordinates,
    and output field values.

    Attributes:
        sample_id: Unique identifier for the sample
        parameters: Input parameters dict (delta_R, E, nu, load, etc.)
        coordinates: Node coordinates (N_nodes, dim)
        displacement: Displacement field (N_nodes, dim) - optional
        stress: Stress field (N_elements, n_components) - optional
        temperature: Temperature field (N_nodes,) - optional
        von_mises: Von Mises stress (N_elements,) - optional
        damage: Damage/phase field (N_nodes,) for fracture - optional
        crack_path: Crack path coordinates (N_points, dim) - optional
        mesh_file: Path to the mesh file
        metadata: Additional metadata
        created_at: Creation timestamp
        is_valid: Whether the simulation converged successfully
    """
    sample_id: str
    parameters: Dict[str, float]
    coordinates: np.ndarray
    displacement: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    von_mises: Optional[np.ndarray] = None
    damage: Optional[np.ndarray] = None
    crack_path: Optional[np.ndarray] = None
    mesh_file: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_valid: bool = True

    def get_output_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get a specific output field by name."""
        field_map = {
            "displacement": self.displacement,
            "stress": self.stress,
            "temperature": self.temperature,
            "von_mises": self.von_mises,
            "damage": self.damage,
            "crack_path": self.crack_path,
        }
        return field_map.get(field_name)

    def get_parameter_vector(self, param_names: List[str]) -> np.ndarray:
        """Get parameters as a vector in specified order."""
        return np.array([self.parameters.get(name, 0.0) for name in param_names])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "sample_id": self.sample_id,
            "parameters": self.parameters,
            "n_nodes": len(self.coordinates) if self.coordinates is not None else 0,
            "has_displacement": self.displacement is not None,
            "has_stress": self.stress is not None,
            "has_temperature": self.temperature is not None,
            "has_damage": self.damage is not None,
            "has_crack_path": self.crack_path is not None,
            "mesh_file": str(self.mesh_file) if self.mesh_file else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "is_valid": self.is_valid,
        }


@dataclass
class DatasetConfig:
    """
    Configuration for FEM dataset.

    Attributes:
        name: Dataset name
        parameter_names: Names of input parameters
        parameter_bounds: Bounds for each parameter
        output_fields: List of output field names to store
        coordinate_dim: Spatial dimension (2 or 3)
        storage_dir: Directory for storing dataset
    """
    name: str = "fem_dataset"
    parameter_names: List[str] = field(default_factory=lambda: ["delta_R", "E", "nu"])
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    output_fields: List[str] = field(default_factory=lambda: ["displacement", "von_mises"])
    coordinate_dim: int = 2
    storage_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "parameter_names": self.parameter_names,
            "parameter_bounds": self.parameter_bounds,
            "output_fields": self.output_fields,
            "coordinate_dim": self.coordinate_dim,
            "storage_dir": str(self.storage_dir) if self.storage_dir else None,
        }


@dataclass
class DatasetStatistics:
    """
    Statistics about a dataset.

    Attributes:
        n_samples: Total number of samples
        n_valid: Number of valid (converged) samples
        n_invalid: Number of invalid samples
        parameter_stats: Statistics for each parameter
        output_stats: Statistics for each output field
        coverage: Coverage metrics for parameter space
    """
    n_samples: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    parameter_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    output_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)


class FEMDataset:
    """
    Dataset for FEM simulation samples.

    Manages a collection of FEM samples with utilities for:
    - Adding/removing samples
    - Querying by parameters
    - Preparing data for surrogate training
    - Computing statistics
    - Saving/loading to disk
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize dataset.

        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()
        self._samples: Dict[str, FEMSample] = {}
        self._sample_order: List[str] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[FEMSample]:
        for sample_id in self._sample_order:
            yield self._samples[sample_id]

    def __getitem__(self, idx: Union[int, str]) -> FEMSample:
        if isinstance(idx, int):
            return self._samples[self._sample_order[idx]]
        return self._samples[idx]

    def add_sample(self, sample: FEMSample) -> None:
        """Add a sample to the dataset."""
        if sample.sample_id in self._samples:
            # Update existing sample
            self._samples[sample.sample_id] = sample
        else:
            self._samples[sample.sample_id] = sample
            self._sample_order.append(sample.sample_id)

    def add_samples(self, samples: List[FEMSample]) -> None:
        """Add multiple samples."""
        for sample in samples:
            self.add_sample(sample)

    def remove_sample(self, sample_id: str) -> Optional[FEMSample]:
        """Remove a sample by ID."""
        if sample_id in self._samples:
            sample = self._samples.pop(sample_id)
            self._sample_order.remove(sample_id)
            return sample
        return None

    def get_valid_samples(self) -> List[FEMSample]:
        """Get all valid (converged) samples."""
        return [s for s in self if s.is_valid]

    def get_samples_in_region(
        self,
        param_ranges: Dict[str, Tuple[float, float]]
    ) -> List[FEMSample]:
        """Get samples within specified parameter ranges."""
        result = []
        for sample in self:
            in_region = True
            for name, (min_val, max_val) in param_ranges.items():
                if name in sample.parameters:
                    val = sample.parameters[name]
                    if not (min_val <= val <= max_val):
                        in_region = False
                        break
            if in_region:
                result.append(sample)
        return result

    def prepare_training_data(
        self,
        output_field: str = "displacement",
        valid_only: bool = True,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Prepare data for surrogate model training.

        Each sample carries its own coordinate array so samples solved on
        different meshes (e.g. different geometries) can coexist in the dataset.

        Args:
            output_field: Which output field to use as target
            valid_only: Only include valid samples

        Returns:
            Tuple of (parameters, coordinates, outputs)
            - parameters:  (N_samples, n_params)
            - coordinates: List of (N_i, coord_dim) — one array per sample
            - outputs:     List of (N_i, output_dim) — one array per sample
        """
        samples = self.get_valid_samples() if valid_only else list(self)

        if not samples:
            raise ValueError("No samples available for training")

        param_names = self.config.parameter_names
        parameters = []
        coordinates = []
        outputs = []

        for sample in samples:
            parameters.append(sample.get_parameter_vector(param_names))

            output = sample.get_output_field(output_field)
            if output is None:
                raise ValueError(
                    f"Sample {sample.sample_id} missing field '{output_field}'"
                )
            # Ensure output is 2-D: (N_points, output_dim)
            if output.ndim == 1:
                output = output[:, np.newaxis]
            outputs.append(output)
            coordinates.append(sample.coordinates)

        return np.array(parameters), coordinates, outputs

    def compute_statistics(self) -> DatasetStatistics:
        """Compute dataset statistics."""
        stats = DatasetStatistics()

        samples = list(self)
        stats.n_samples = len(samples)
        stats.n_valid = sum(1 for s in samples if s.is_valid)
        stats.n_invalid = stats.n_samples - stats.n_valid

        if not samples:
            return stats

        # Parameter statistics
        for param_name in self.config.parameter_names:
            values = [s.parameters.get(param_name, 0.0) for s in samples]
            if values:
                stats.parameter_stats[param_name] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        # Output field statistics
        for field_name in self.config.output_fields:
            field_values = []
            for sample in samples:
                field_data = sample.get_output_field(field_name)
                if field_data is not None:
                    field_values.append(field_data.flatten())

            if field_values:
                all_values = np.concatenate(field_values)
                stats.output_stats[field_name] = {
                    "min": float(np.min(all_values)),
                    "max": float(np.max(all_values)),
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                }

        # Coverage analysis
        for param_name in self.config.parameter_names:
            if param_name in self.config.parameter_bounds:
                min_bound, max_bound = self.config.parameter_bounds[param_name]
                values = [s.parameters.get(param_name, 0.0) for s in samples]
                if values:
                    actual_range = max(values) - min(values)
                    full_range = max_bound - min_bound
                    stats.coverage[param_name] = actual_range / full_range if full_range > 0 else 0.0

        return stats

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save dataset to disk.

        Args:
            path: Directory to save to (uses config.storage_dir if None)

        Returns:
            Path where dataset was saved
        """
        path = Path(path) if path else self.config.storage_dir
        if path is None:
            raise ValueError("No save path specified")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save sample metadata
        metadata_path = path / "samples.json"
        metadata = {
            "sample_order": self._sample_order,
            "samples": {sid: s.to_dict() for sid, s in self._samples.items()}
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save arrays
        arrays_dir = path / "arrays"
        arrays_dir.mkdir(exist_ok=True)

        for sample_id, sample in self._samples.items():
            sample_dir = arrays_dir / sample_id
            sample_dir.mkdir(exist_ok=True)

            np.save(sample_dir / "coordinates.npy", sample.coordinates)

            if sample.displacement is not None:
                np.save(sample_dir / "displacement.npy", sample.displacement)
            if sample.stress is not None:
                np.save(sample_dir / "stress.npy", sample.stress)
            if sample.temperature is not None:
                np.save(sample_dir / "temperature.npy", sample.temperature)
            if sample.von_mises is not None:
                np.save(sample_dir / "von_mises.npy", sample.von_mises)
            if sample.damage is not None:
                np.save(sample_dir / "damage.npy", sample.damage)
            if sample.crack_path is not None:
                np.save(sample_dir / "crack_path.npy", sample.crack_path)

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FEMDataset":
        """
        Load dataset from disk.

        Args:
            path: Directory to load from

        Returns:
            Loaded FEMDataset
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = DatasetConfig(
            name=config_dict.get("name", "fem_dataset"),
            parameter_names=config_dict.get("parameter_names", []),
            parameter_bounds=config_dict.get("parameter_bounds", {}),
            output_fields=config_dict.get("output_fields", []),
            coordinate_dim=config_dict.get("coordinate_dim", 2),
            storage_dir=Path(config_dict["storage_dir"]) if config_dict.get("storage_dir") else None,
        )

        dataset = cls(config)

        # Load sample metadata
        metadata_path = path / "samples.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        arrays_dir = path / "arrays"

        # Load samples
        for sample_id in metadata["sample_order"]:
            sample_meta = metadata["samples"][sample_id]
            sample_dir = arrays_dir / sample_id

            # Load arrays
            coordinates = np.load(sample_dir / "coordinates.npy")

            displacement = None
            stress = None
            temperature = None
            von_mises = None
            damage = None
            crack_path = None

            if (sample_dir / "displacement.npy").exists():
                displacement = np.load(sample_dir / "displacement.npy")
            if (sample_dir / "stress.npy").exists():
                stress = np.load(sample_dir / "stress.npy")
            if (sample_dir / "temperature.npy").exists():
                temperature = np.load(sample_dir / "temperature.npy")
            if (sample_dir / "von_mises.npy").exists():
                von_mises = np.load(sample_dir / "von_mises.npy")
            if (sample_dir / "damage.npy").exists():
                damage = np.load(sample_dir / "damage.npy")
            if (sample_dir / "crack_path.npy").exists():
                crack_path = np.load(sample_dir / "crack_path.npy")

            sample = FEMSample(
                sample_id=sample_id,
                parameters=sample_meta["parameters"],
                coordinates=coordinates,
                displacement=displacement,
                stress=stress,
                temperature=temperature,
                von_mises=von_mises,
                damage=damage,
                crack_path=crack_path,
                mesh_file=Path(sample_meta["mesh_file"]) if sample_meta.get("mesh_file") else None,
                metadata=sample_meta.get("metadata", {}),
                created_at=sample_meta.get("created_at", ""),
                is_valid=sample_meta.get("is_valid", True),
            )

            dataset.add_sample(sample)

        return dataset
