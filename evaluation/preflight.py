"""
Pre-flight checking for Abaqus models.

Wraps the existing validator.py functionality with an interface
suitable for the agent system.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PreflightStatus(Enum):
    """Status of pre-flight check."""
    PASSED = "passed"
    WARNINGS = "warnings"
    FAILED = "failed"


@dataclass
class PreflightResult:
    """Result of pre-flight validation."""
    status: PreflightStatus = PreflightStatus.PASSED
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mesh_quality: Dict[str, Any] = field(default_factory=dict)
    material_check: Dict[str, Any] = field(default_factory=dict)
    bc_check: Dict[str, Any] = field(default_factory=dict)
    geometry_check: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "mesh_quality": self.mesh_quality,
            "material_check": self.material_check,
            "bc_check": self.bc_check,
            "geometry_check": self.geometry_check,
            "score": self.score,
        }

    def get_summary(self) -> str:
        """Get a summary of the pre-flight check."""
        lines = [
            f"Pre-flight Status: {self.status.value.upper()}",
            f"Valid: {self.is_valid}",
            f"Score: {self.score:.2f}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors[:5]:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings[:5]:
                lines.append(f"  - {warn}")

        return "\n".join(lines)


class PreflightChecker:
    """
    Pre-flight checker for Abaqus models.

    Wraps the AbaqusValidator to provide pre-flight validation
    before solver execution.
    """

    # Default thresholds
    DEFAULT_MIN_JACOBIAN = 0.01
    DEFAULT_MAX_ASPECT_RATIO = 100.0
    DEFAULT_MIN_NODE_DISTANCE = 1e-10

    def __init__(
        self,
        min_jacobian: float = DEFAULT_MIN_JACOBIAN,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
        min_node_distance: float = DEFAULT_MIN_NODE_DISTANCE,
    ):
        """
        Initialize the pre-flight checker.

        Args:
            min_jacobian: Minimum acceptable Jacobian ratio
            max_aspect_ratio: Maximum acceptable aspect ratio
            min_node_distance: Minimum distance between nodes
        """
        self.min_jacobian = min_jacobian
        self.max_aspect_ratio = max_aspect_ratio
        self.min_node_distance = min_node_distance

    def check_file(self, inp_path: str) -> PreflightResult:
        """
        Run pre-flight checks on an .inp file.

        Args:
            inp_path: Path to the .inp file

        Returns:
            PreflightResult
        """
        result = PreflightResult()

        try:
            from manager import AbaqusManager
            from validator import AbaqusValidator

            # Load the model
            if not Path(inp_path).exists():
                result.status = PreflightStatus.FAILED
                result.is_valid = False
                result.errors.append(f"File not found: {inp_path}")
                result.score = 0.0
                return result

            manager = AbaqusManager(inp_path)
            return self.check_manager(manager)

        except Exception as e:
            result.status = PreflightStatus.FAILED
            result.is_valid = False
            result.errors.append(f"Failed to load model: {str(e)}")
            result.score = 0.0
            return result

    def check_manager(self, manager) -> PreflightResult:
        """
        Run pre-flight checks on an AbaqusManager instance.

        Args:
            manager: AbaqusManager instance

        Returns:
            PreflightResult
        """
        result = PreflightResult()

        try:
            from validator import AbaqusValidator

            # Run the validator
            validator = AbaqusValidator(manager)
            report = validator.validate_all()

            # Transfer results
            result.is_valid = report.is_valid
            result.errors = report.errors.copy()
            result.warnings = report.warnings.copy()

            # Additional mesh quality checks
            result.mesh_quality = self._check_mesh_quality(manager)
            result.material_check = self._check_materials(manager)
            result.bc_check = self._check_boundary_conditions(manager)
            result.geometry_check = self._check_geometry(manager)

            # Add warnings from our checks
            if result.mesh_quality.get("jacobian_warning"):
                result.warnings.append(result.mesh_quality["jacobian_warning"])
            if result.mesh_quality.get("aspect_ratio_warning"):
                result.warnings.append(result.mesh_quality["aspect_ratio_warning"])

            # Determine status
            if not result.is_valid or result.errors:
                result.status = PreflightStatus.FAILED
            elif result.warnings:
                result.status = PreflightStatus.WARNINGS
            else:
                result.status = PreflightStatus.PASSED

            # Calculate score
            result.score = self._calculate_score(result)

        except Exception as e:
            result.status = PreflightStatus.FAILED
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
            result.score = 0.0

        return result

    def _check_mesh_quality(self, manager) -> Dict[str, Any]:
        """Check mesh quality metrics."""
        quality = {
            "num_nodes": 0,
            "num_elements": 0,
            "jacobian_min": None,
            "jacobian_avg": None,
            "aspect_ratio_max": None,
            "aspect_ratio_avg": None,
        }

        try:
            import numpy as np

            if manager.nodes is not None:
                quality["num_nodes"] = len(manager.nodes.ids)

            if manager.elements is not None:
                quality["num_elements"] = len(manager.elements.ids)

                # Calculate aspect ratios for elements
                aspect_ratios = []
                coords = manager.nodes.data
                node_ids = manager.nodes.ids

                for elem_idx in range(len(manager.elements.ids)):
                    connectivity = manager.elements.data[elem_idx]
                    try:
                        elem_coords = []
                        for node_id in connectivity:
                            idx = np.where(node_ids == node_id)[0]
                            if len(idx) > 0:
                                elem_coords.append(coords[idx[0]])

                        if len(elem_coords) >= 2:
                            elem_coords = np.array(elem_coords)
                            # Simple bounding box aspect ratio
                            ranges = np.ptp(elem_coords, axis=0)
                            nonzero = ranges[ranges > 0]
                            if len(nonzero) >= 2:
                                ar = max(nonzero) / min(nonzero)
                                aspect_ratios.append(ar)
                    except Exception:
                        continue

                if aspect_ratios:
                    quality["aspect_ratio_max"] = max(aspect_ratios)
                    quality["aspect_ratio_avg"] = sum(aspect_ratios) / len(aspect_ratios)

                    if quality["aspect_ratio_max"] > self.max_aspect_ratio:
                        quality["aspect_ratio_warning"] = (
                            f"High aspect ratio detected: {quality['aspect_ratio_max']:.2f} "
                            f"(max allowed: {self.max_aspect_ratio})"
                        )

        except Exception as e:
            quality["error"] = str(e)

        return quality

    def _check_materials(self, manager) -> Dict[str, Any]:
        """Check material definitions."""
        check = {
            "num_materials": len(manager.materials),
            "materials": {},
            "valid": True,
        }

        for name, material in manager.materials.items():
            mat_info = {"valid": True, "issues": []}

            if "elastic" in material.data:
                elastic = material.data["elastic"]
                E = elastic.get("E", 0)
                nu = elastic.get("nu", 0)

                if E <= 0:
                    mat_info["issues"].append("Non-positive elastic modulus")
                    mat_info["valid"] = False

                if nu < 0 or nu >= 0.5:
                    mat_info["issues"].append(f"Invalid Poisson's ratio: {nu}")
                    mat_info["valid"] = False

                mat_info["E"] = E
                mat_info["nu"] = nu

            check["materials"][name] = mat_info
            if not mat_info["valid"]:
                check["valid"] = False

        return check

    def _check_boundary_conditions(self, manager) -> Dict[str, Any]:
        """Check boundary conditions."""
        check = {
            "num_bcs": len(manager.boundary_conditions),
            "bcs": {},
            "valid": True,
        }

        for name, bc in manager.boundary_conditions.items():
            bc_info = {"valid": True, "issues": []}

            set_name = bc.data.get("set_name", "")
            if set_name and set_name not in manager.node_sets:
                bc_info["issues"].append(f"References undefined node set: {set_name}")
                bc_info["valid"] = False

            check["bcs"][name] = bc_info
            if not bc_info["valid"]:
                check["valid"] = False

        return check

    def _check_geometry(self, manager) -> Dict[str, Any]:
        """Check geometry validity."""
        check = {
            "bounds": {},
            "center": None,
            "valid": True,
        }

        try:
            import numpy as np

            if manager.nodes is not None:
                coords = manager.nodes.data
                check["bounds"] = {
                    "x_min": float(coords[:, 0].min()),
                    "x_max": float(coords[:, 0].max()),
                    "y_min": float(coords[:, 1].min()),
                    "y_max": float(coords[:, 1].max()),
                }
                check["center"] = [
                    float(coords[:, 0].mean()),
                    float(coords[:, 1].mean()),
                ]

                # Check for NaN or Inf
                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                    check["valid"] = False
                    check["issues"] = ["Invalid coordinates (NaN or Inf detected)"]

        except Exception as e:
            check["error"] = str(e)

        return check

    def _calculate_score(self, result: PreflightResult) -> float:
        """Calculate overall pre-flight score."""
        score = 1.0

        # Penalize errors heavily
        score -= 0.3 * len(result.errors)

        # Penalize warnings lightly
        score -= 0.05 * len(result.warnings)

        # Adjust for aspect ratio
        ar_max = result.mesh_quality.get("aspect_ratio_max")
        if ar_max is not None:
            if ar_max > self.max_aspect_ratio:
                score -= 0.2
            elif ar_max > self.max_aspect_ratio / 2:
                score -= 0.1

        # Adjust for material validity
        if not result.material_check.get("valid", True):
            score -= 0.2

        # Adjust for BC validity
        if not result.bc_check.get("valid", True):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def quick_check(self, manager) -> bool:
        """
        Quick validity check (no detailed metrics).

        Args:
            manager: AbaqusManager instance

        Returns:
            True if model passes basic checks
        """
        try:
            from validator import AbaqusValidator

            validator = AbaqusValidator(manager)
            report = validator.validate_all()
            return report.is_valid
        except Exception:
            return False
