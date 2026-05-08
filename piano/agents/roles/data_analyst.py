"""
Data Analyst Agent.

Runs before training begins (Phase 0) to analyze the FEM dataset:
- Node distribution and mesh topology statistics
- Crack tip proximity distribution (near-tip density)
- Singularity strength estimate (r^{-0.5} fit quality)
- Output field statistics (displacement range, von Mises peak, skewness)

Writes a persistent `data_analysis.md` report that all downstream agents
receive as context. Modeled after the Data Analyst in Jiang & Karniadakis (2026).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from piano.agents.base import AgentContext, AgentRole, BaseAgent

logger = logging.getLogger(__name__)


_ANALYST_SYSTEM = """You are a scientific data analyst for physics-informed machine learning (PIML) datasets.

You receive statistics about a finite element mesh dataset for crack/fracture mechanics. Your job is to:
1. Identify data quality issues that will affect neural operator training
2. Flag imbalances in near-tip vs far-field node sampling
3. Estimate singularity strength from the proximity distribution
4. Recommend specific preprocessing steps (tip weighting, normalization, enrichment)

Be concise and quantitative. Use the exact numbers provided. Do NOT suggest hyperparameters — only
describe the data characteristics and preprocessing recommendations.

Output format (exactly):
DATASET_QUALITY: <good|moderate|poor>
N_SAMPLES: <int>
N_NODES_RANGE: <min>-<max>
NEAR_TIP_FRACTION: <fraction of nodes within r < 0.05*domain_size>
OUTPUT_SKEWNESS: <skewness metric description>
RECOMMENDATIONS:
- <bullet 1>
- <bullet 2>
...
SUMMARY: <2-3 sentences summarizing the dataset for downstream agents>"""


_ANALYST_PROMPT = """Dataset statistics for {n_samples} FEM samples:

Node counts: min={n_nodes_min}, max={n_nodes_max}, mean={n_nodes_mean:.0f}
Variable mesh topology: {variable_topology} ({n_unique_sizes} unique sizes)
Tip coordinates: {tip_coords}
Near-tip node fraction (r < 0.05): {near_tip_fraction:.3f}
Near-tip node fraction (r < 0.01): {very_near_tip_fraction:.3f}

Output field statistics:
{output_stats}

Parameter space:
{param_stats}

Analyze this dataset for neural operator training suitability."""


@dataclass
class DataAnalysis:
    """Structured output of the Data Analyst."""
    dataset_quality: str = "moderate"
    n_samples: int = 0
    n_nodes_range: str = ""
    near_tip_fraction: float = 0.0
    output_skewness: str = ""
    recommendations: List[str] = None
    summary: str = ""
    raw_response: str = ""

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

    def to_context_string(self) -> str:
        recs = "\n".join(f"  - {r}" for r in self.recommendations)
        return (
            f"[DATA ANALYST REPORT]\n"
            f"Dataset Quality: {self.dataset_quality}\n"
            f"Samples: {self.n_samples}, Nodes: {self.n_nodes_range}\n"
            f"Near-tip fraction: {self.near_tip_fraction:.3f}\n"
            f"Output skewness: {self.output_skewness}\n"
            f"Recommendations:\n{recs}\n"
            f"Summary: {self.summary}"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# Data Analysis Report\n\n")
            f.write(self.to_context_string())
            f.write("\n\n## Raw LLM Response\n\n")
            f.write(self.raw_response)
        logger.info(f"DataAnalyst: report saved to {path}")


class DataAnalystAgent(BaseAgent[DataAnalysis]):
    """
    Pre-training dataset analysis agent.

    Analyzes FEM mesh data before training starts. Writes a persistent
    data_analysis.md report consumed by all downstream agents as context.

    Call analyze() once per training session; the result is injected into
    AgentContext.knowledge_context alongside any KB entry from KnowledgeRetriever.
    """

    def __init__(self, model: str = "gpt-4-turbo", **kwargs):
        super().__init__(role=AgentRole.ANALYST, model=model, temperature=0.1, **kwargs)

    def get_system_prompt(self) -> str:
        return _ANALYST_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        stats = kwargs["stats"]
        return _ANALYST_PROMPT.format(**stats)

    def parse_response(self, response: str) -> DataAnalysis:
        result = DataAnalysis(raw_response=response)

        m = re.search(r"DATASET_QUALITY:\s*(\S+)", response)
        if m:
            result.dataset_quality = m.group(1).strip(".,").lower()

        m = re.search(r"N_SAMPLES:\s*(\d+)", response)
        if m:
            result.n_samples = int(m.group(1))

        m = re.search(r"N_NODES_RANGE:\s*(\S+)", response)
        if m:
            result.n_nodes_range = m.group(1).strip()

        m = re.search(r"NEAR_TIP_FRACTION:\s*([0-9.]+)", response)
        if m:
            result.near_tip_fraction = float(m.group(1))

        m = re.search(r"OUTPUT_SKEWNESS:\s*(.+?)(?=RECOMMENDATIONS:|SUMMARY:|$)", response, re.DOTALL)
        if m:
            result.output_skewness = m.group(1).strip()

        m = re.search(r"RECOMMENDATIONS:\s*(.+?)(?=SUMMARY:|$)", response, re.DOTALL)
        if m:
            recs_text = m.group(1)
            result.recommendations = [
                line.lstrip("- •").strip()
                for line in recs_text.splitlines()
                if line.strip().startswith("-") or line.strip().startswith("•")
            ]

        m = re.search(r"SUMMARY:\s*(.+)$", response, re.DOTALL)
        if m:
            result.summary = m.group(1).strip()

        return result

    @staticmethod
    def compute_stats(
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        parameters: np.ndarray,
        tip_coords: Optional[np.ndarray],
        parameter_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute dataset statistics to pass to the LLM.

        Args:
            coordinates: Per-sample coordinate arrays (N_i, coord_dim)
            outputs: Per-sample output arrays (N_i, output_dim)
            parameters: Parameter matrix (n_samples, n_params)
            tip_coords: Crack tip location (2,) or None
            parameter_names: Names for parameter columns
        """
        node_counts = [c.shape[0] for c in coordinates]
        unique_sizes = len(set(node_counts))
        variable_topology = unique_sizes > 1

        # Near-tip fractions
        near_tip_frac = 0.0
        very_near_tip_frac = 0.0
        if tip_coords is not None and len(coordinates) > 0:
            all_r = []
            for c in coordinates:
                r = np.linalg.norm(c[:, :2] - tip_coords[:2], axis=1)
                all_r.append(r)
            all_r_flat = np.concatenate(all_r)
            domain_size = np.percentile(all_r_flat, 95)
            near_tip_frac = np.mean(all_r_flat < 0.05 * domain_size)
            very_near_tip_frac = np.mean(all_r_flat < 0.01 * domain_size)
            tip_str = f"({tip_coords[0]:.4f}, {tip_coords[1]:.4f})"
        else:
            tip_str = "not provided"

        # Output statistics
        output_parts = []
        if outputs:
            all_out = np.concatenate(outputs, axis=0)
            for dim in range(all_out.shape[1]):
                col = all_out[:, dim]
                skew = float(np.mean(((col - col.mean()) / (col.std() + 1e-10))**3))
                output_parts.append(
                    f"  dim {dim}: mean={col.mean():.4f}, std={col.std():.4f}, "
                    f"min={col.min():.4f}, max={col.max():.4f}, skewness={skew:.2f}"
                )
        output_stats = "\n".join(output_parts) or "  (no outputs)"

        # Parameter stats
        param_parts = []
        if parameters is not None and parameters.ndim == 2:
            names = parameter_names or [f"p{i}" for i in range(parameters.shape[1])]
            for i, name in enumerate(names):
                col = parameters[:, i]
                param_parts.append(
                    f"  {name}: [{col.min():.4f}, {col.max():.4f}], mean={col.mean():.4f}"
                )
        param_stats = "\n".join(param_parts) or "  (no parameters)"

        return {
            "n_samples": len(coordinates),
            "n_nodes_min": min(node_counts) if node_counts else 0,
            "n_nodes_max": max(node_counts) if node_counts else 0,
            "n_nodes_mean": np.mean(node_counts) if node_counts else 0.0,
            "variable_topology": str(variable_topology),
            "n_unique_sizes": unique_sizes,
            "tip_coords": tip_str,
            "near_tip_fraction": near_tip_frac,
            "very_near_tip_fraction": very_near_tip_frac,
            "output_stats": output_stats,
            "param_stats": param_stats,
        }

    async def analyze(
        self,
        context: AgentContext,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        parameters: np.ndarray,
        tip_coords: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        report_path: Optional[Path] = None,
    ) -> DataAnalysis:
        """
        Run dataset analysis and optionally write a persistent report.

        Args:
            context: Agent context (analysis summary added to knowledge_context)
            coordinates: Per-sample coordinate arrays
            outputs: Per-sample output arrays
            parameters: Parameter matrix
            tip_coords: Crack tip position for near-tip stats
            parameter_names: Names for parameter columns
            report_path: If set, write data_analysis.md here

        Returns:
            DataAnalysis with structured findings
        """
        stats = self.compute_stats(coordinates, outputs, parameters, tip_coords, parameter_names)
        result = await self.execute(context, stats=stats)
        result.n_samples = stats["n_samples"]

        if report_path:
            result.save(Path(report_path))

        context.knowledge_context.append({
            "source": "data_analyst",
            "text": result.to_context_string(),
        })

        return result

    def analyze_sync(
        self,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        parameters: np.ndarray,
        tip_coords: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        report_path: Optional[Path] = None,
    ) -> DataAnalysis:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.analyze(
                    AgentContext(),
                    coordinates, outputs, parameters,
                    tip_coords, parameter_names, report_path,
                )
            )
        finally:
            loop.close()
