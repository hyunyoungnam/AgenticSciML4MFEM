"""
R-adaptivity package using TMOP.

Implements error-driven mesh adaptation using MFEM's Target-Matrix Optimization
Paradigm (TMOP). Nodes are redistributed to cluster in high-error regions based
on the surrogate model's error field.

Key features:
- Error-driven r-adaptivity: nodes move toward high-error regions
- TMOP barrier functions: prevent element inversion
- Quality preservation: maintains valid mesh during adaptation

Usage:
    from meshforge.morphing import TMOPAdaptivity, AdaptivityConfig

    # Create adaptivity engine
    config = AdaptivityConfig(
        size_scale_min=0.3,  # Small elements in high-error regions
        size_scale_max=2.0,  # Large elements in low-error regions
    )
    adaptivity = TMOPAdaptivity(config)

    # Get error field from surrogate model
    error_field = surrogate.compute_pointwise_error(coords, params)

    # Adapt mesh
    result = adaptivity.adapt(mesh_manager, error_field)
    if result.success:
        print(f"Adapted {len(result.coords_adapted)} nodes")
"""

from .r_adaptivity import (
    TMOPAdaptivity,
    AdaptivityConfig,
    AdaptivityResult,
    is_tmop_available,
)

__all__ = [
    "TMOPAdaptivity",
    "AdaptivityConfig",
    "AdaptivityResult",
    "is_tmop_available",
]
