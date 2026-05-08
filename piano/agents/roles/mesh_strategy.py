"""
Mesh Strategy Agent.

Runs alongside AdaptiveProposerAgent in the active learning loop.
Decides where to apply r-refinement (move nodes toward singularity) vs
h-refinement (subdivide elements for coverage) based on:
- Surrogate error map (spatial distribution of prediction error)
- Current mesh topology (element sizes near tip vs far field)
- Crack tip location
- Active learning iteration history

MFEM-specific: understands the r/h-refinement API and generates concrete
mesh strategy recommendations consumed by AdaptiveOrchestrator.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from piano.agents.base import AgentContext, AgentRole, BaseAgent

logger = logging.getLogger(__name__)


_MESH_SYSTEM = """You are a finite element mesh strategy expert for fracture mechanics simulations.

You receive:
1. A surrogate model error map (mean absolute error per spatial zone)
2. Current mesh statistics (element sizes, node density near tip vs far field)
3. The crack tip location
4. History of previous refinement decisions

Your task is to recommend ONE mesh adaptation strategy for the NEXT active learning iteration.

Choose from:
- **r-refinement**: Move existing nodes toward the crack tip (preserves topology, improves near-tip resolution without adding DOFs)
  - Use when: near-tip error is high but overall error is acceptable
- **h-refinement**: Subdivide elements to add new nodes (increases DOFs, improves resolution in specific zones)
  - Use when: a spatial zone has consistently high error that cannot be resolved by moving nodes
- **both**: Apply r-refinement near tip + h-refinement in one high-error far-field zone
  - Use when: both near-tip AND far-field errors are high and budget allows
- **none**: No refinement this iteration
  - Use when: error is uniformly low or budget is exhausted

Be quantitative. Use the exact numbers provided.

Output format:
STRATEGY: <r_only|h_only|both|none>
R_REFINEMENT_TARGET: <zone description, or "none">
H_REFINEMENT_ZONE: <zone description, or "none">
EXPECTED_IMPROVEMENT: <brief quantitative estimate>
REASONING: <2-3 sentences>"""


_MESH_PROMPT = """Surrogate error map (mean absolute error by zone):
{error_map}

Current mesh statistics:
- Total nodes: {n_nodes}
- Tip location: ({tip_x:.4f}, {tip_y:.4f})
- Near-tip element size (r < 0.05): {h_near_tip:.6f}
- Far-field element size: {h_far_field:.6f}
- Near-tip node fraction: {near_tip_fraction:.3f}

Active learning iteration: {iteration}
Samples collected so far: {n_samples}
Previous refinement decisions: {prev_decisions}

Recommend the mesh adaptation strategy for the next iteration."""


@dataclass
class MeshStrategyDecision:
    """Output of the Mesh Strategy Agent."""
    strategy: str = "none"
    r_refinement_target: str = "none"
    h_refinement_zone: str = "none"
    expected_improvement: str = ""
    reasoning: str = ""
    raw_response: str = ""

    def to_summary(self) -> str:
        return (
            f"Strategy: {self.strategy} | "
            f"R-target: {self.r_refinement_target} | "
            f"H-zone: {self.h_refinement_zone} | "
            f"{self.reasoning[:100]}"
        )

    def needs_r_refinement(self) -> bool:
        return self.strategy in ("r_only", "both")

    def needs_h_refinement(self) -> bool:
        return self.strategy in ("h_only", "both")


class MeshStrategyAgent(BaseAgent[MeshStrategyDecision]):
    """
    Mesh adaptation strategy agent for active learning.

    Analyzes the surrogate error distribution and current mesh topology,
    then recommends r- and/or h-refinement targets for the next iteration.

    Runs once per active learning iteration, after surrogate evaluation
    but before new parameter selection.
    """

    def __init__(self, model: str = "gpt-4-turbo", **kwargs):
        super().__init__(role=AgentRole.ANALYST, model=model, temperature=0.2, **kwargs)
        self._decision_history: List[str] = []

    def get_system_prompt(self) -> str:
        return _MESH_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        return _MESH_PROMPT.format(**kwargs["mesh_stats"])

    def parse_response(self, response: str) -> MeshStrategyDecision:
        result = MeshStrategyDecision(raw_response=response)

        m = re.search(r"STRATEGY:\s*(\S+)", response, re.IGNORECASE)
        if m:
            result.strategy = m.group(1).strip(".,").lower()

        m = re.search(r"R_REFINEMENT_TARGET:\s*(.+?)(?=H_REFINEMENT_ZONE:|EXPECTED_IMPROVEMENT:|REASONING:|$)", response, re.DOTALL | re.IGNORECASE)
        if m:
            result.r_refinement_target = m.group(1).strip()

        m = re.search(r"H_REFINEMENT_ZONE:\s*(.+?)(?=EXPECTED_IMPROVEMENT:|REASONING:|$)", response, re.DOTALL | re.IGNORECASE)
        if m:
            result.h_refinement_zone = m.group(1).strip()

        m = re.search(r"EXPECTED_IMPROVEMENT:\s*(.+?)(?=REASONING:|$)", response, re.DOTALL | re.IGNORECASE)
        if m:
            result.expected_improvement = m.group(1).strip()

        m = re.search(r"REASONING:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if m:
            result.reasoning = m.group(1).strip()

        return result

    @staticmethod
    def compute_mesh_stats(
        coordinates: np.ndarray,
        errors: np.ndarray,
        tip_coords: np.ndarray,
        iteration: int,
        n_samples: int,
        prev_decisions: Optional[List[str]] = None,
        n_zones: int = 4,
    ) -> Dict[str, Any]:
        """
        Compute mesh statistics for the LLM prompt.

        Args:
            coordinates: Node coordinates (N, 2)
            errors: Per-node absolute error (N,) — from surrogate evaluation
            tip_coords: Crack tip (2,)
            iteration: Current active learning iteration index
            n_samples: Total samples collected
            prev_decisions: List of previous strategy strings
            n_zones: Number of radial zones for error map
        """
        r = np.linalg.norm(coordinates[:, :2] - tip_coords[:2], axis=1)
        domain_size = np.percentile(r, 95)

        # Radial zone boundaries
        zone_edges = np.linspace(0, domain_size, n_zones + 1)
        error_map_parts = []
        for i in range(n_zones):
            mask = (r >= zone_edges[i]) & (r < zone_edges[i + 1])
            if mask.sum() > 0:
                zone_err = float(errors[mask].mean())
                zone_r = f"{zone_edges[i]:.4f}–{zone_edges[i+1]:.4f}"
                n_nodes = mask.sum()
                error_map_parts.append(
                    f"  r=[{zone_r}]: MAE={zone_err:.6f}, n_nodes={n_nodes}"
                )
        error_map = "\n".join(error_map_parts) or "  (no error data)"

        # Near-tip vs far-field element size approximation (nearest-neighbor distance)
        from scipy.spatial import KDTree
        if len(coordinates) > 1:
            tree = KDTree(coordinates[:, :2])
            dists, _ = tree.query(coordinates[:, :2], k=2)
            nn_dists = dists[:, 1]
            near_mask = r < 0.05 * domain_size
            h_near = float(nn_dists[near_mask].mean()) if near_mask.sum() > 0 else float(nn_dists.mean())
            h_far = float(nn_dists[~near_mask].mean()) if (~near_mask).sum() > 0 else float(nn_dists.mean())
            near_frac = float(near_mask.mean())
        else:
            h_near = h_far = 0.0
            near_frac = 0.0

        return {
            "error_map": error_map,
            "n_nodes": len(coordinates),
            "tip_x": float(tip_coords[0]),
            "tip_y": float(tip_coords[1]),
            "h_near_tip": h_near,
            "h_far_field": h_far,
            "near_tip_fraction": near_frac,
            "iteration": iteration,
            "n_samples": n_samples,
            "prev_decisions": ", ".join(prev_decisions[-3:]) if prev_decisions else "none",
        }

    def decide_sync(
        self,
        context: AgentContext,
        coordinates: np.ndarray,
        errors: np.ndarray,
        tip_coords: np.ndarray,
        iteration: int,
        n_samples: int,
    ) -> MeshStrategyDecision:
        """Synchronous wrapper around decide()."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.decide(context, coordinates, errors, tip_coords, iteration, n_samples),
                    )
                    return future.result()
            return loop.run_until_complete(
                self.decide(context, coordinates, errors, tip_coords, iteration, n_samples)
            )
        except RuntimeError:
            return asyncio.run(
                self.decide(context, coordinates, errors, tip_coords, iteration, n_samples)
            )

    async def decide(
        self,
        context: AgentContext,
        coordinates: np.ndarray,
        errors: np.ndarray,
        tip_coords: np.ndarray,
        iteration: int,
        n_samples: int,
    ) -> MeshStrategyDecision:
        """
        Recommend a mesh adaptation strategy.

        Args:
            context: Agent context
            coordinates: Node coordinates (N, 2)
            errors: Per-node surrogate absolute error (N,)
            tip_coords: Crack tip location (2,)
            iteration: Active learning iteration index
            n_samples: Total collected samples

        Returns:
            MeshStrategyDecision
        """
        try:
            from scipy.spatial import KDTree
        except ImportError:
            logger.warning("MeshStrategyAgent: scipy not available, returning 'none' strategy")
            return MeshStrategyDecision(strategy="none", reasoning="scipy unavailable")

        mesh_stats = self.compute_mesh_stats(
            coordinates, errors, tip_coords, iteration, n_samples,
            prev_decisions=self._decision_history,
        )
        decision = await self.execute(context, mesh_stats=mesh_stats)
        self._decision_history.append(decision.strategy)
        logger.info(f"MeshStrategyAgent: {decision.to_summary()}")
        return decision

    def decide_sync(
        self,
        coordinates: np.ndarray,
        errors: np.ndarray,
        tip_coords: np.ndarray,
        iteration: int,
        n_samples: int,
    ) -> MeshStrategyDecision:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.decide(AgentContext(), coordinates, errors, tip_coords, iteration, n_samples)
            )
        finally:
            loop.close()
