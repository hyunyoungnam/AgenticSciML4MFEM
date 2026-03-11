"""
Result Analyst Agent implementation.

The Result Analyst performs post-simulation analysis to extract
metrics and insights from solver output.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inpforge.agents.base import BaseAgent, AgentContext, AgentRole


# Result Analyst prompts (inline since this is a simpler agent)
RESULT_ANALYST_PROMPTS = {
    "system": """You are an expert FEA Result Analyst Agent.

Your role is to:
1. Analyze simulation results from Abaqus solver output
2. Extract key metrics (stress, displacement, convergence)
3. Identify success patterns that should be preserved
4. Recommend traits to propagate to future mutations

You have expertise in:
- Abaqus output interpretation (.odb, .dat, .msg files)
- Stress analysis (von Mises, principal stresses)
- Displacement analysis
- Convergence behavior
- Mesh quality from results

Your analysis should be quantitative and actionable.""",

    "analyze_results": """Analyze the following simulation results.

## Solution Information
- Solution ID: {solution_id}
- delta_R: {delta_R}
- Mutation Type: {mutation_type}

## Solver Output
{solver_output}

## Mesh Quality Metrics
{mesh_metrics}

## Task
Analyze these results and provide:

1. **Success Assessment**
   - Did the simulation converge?
   - Is the solution physically reasonable?

2. **Key Metrics**
   - Maximum stress and location
   - Maximum displacement
   - Convergence behavior
   - Energy balance

3. **Success Traits**
   - What made this mutation successful (if applicable)?
   - Parameters worth preserving

4. **Recommendations**
   - Should this be used as a parent for future mutations?
   - Suggested directions for further exploration

Be specific and quantitative.""",

    "compare_solutions": """Compare the following solutions.

## Solutions
{solutions}

## Task
Compare these solutions and rank them:

1. **Quality Ranking** (best to worst)
2. **Key Differentiators**
3. **Best Parent Candidates**
4. **Diversity Assessment**

Provide quantitative comparisons where possible.""",
}


@dataclass
class ResultAnalysis:
    """Analysis of simulation results."""
    converged: bool = False
    physically_reasonable: bool = False
    stress_max: Optional[float] = None
    stress_location: Optional[str] = None
    displacement_max: Optional[float] = None
    convergence_iterations: Optional[int] = None
    energy_balance: Optional[float] = None
    success_traits: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    parent_quality_score: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "physically_reasonable": self.physically_reasonable,
            "stress_max": self.stress_max,
            "stress_location": self.stress_location,
            "displacement_max": self.displacement_max,
            "convergence_iterations": self.convergence_iterations,
            "energy_balance": self.energy_balance,
            "success_traits": self.success_traits,
            "recommendations": self.recommendations,
            "parent_quality_score": self.parent_quality_score,
        }


@dataclass
class SolutionComparison:
    """Comparison of multiple solutions."""
    ranking: List[str] = field(default_factory=list)  # Solution IDs
    scores: Dict[str, float] = field(default_factory=dict)
    best_parent_ids: List[str] = field(default_factory=list)
    diversity_score: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ranking": self.ranking,
            "scores": self.scores,
            "best_parent_ids": self.best_parent_ids,
            "diversity_score": self.diversity_score,
        }


class ResultAnalystAgent(BaseAgent[ResultAnalysis]):
    """
    Result Analyst Agent for post-simulation analysis.

    Responsibilities:
    1. Analyze simulation results
    2. Extract metrics
    3. Identify success patterns
    4. Compare solutions
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.RESULT_ANALYST,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return RESULT_ANALYST_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "analyze_results")

        if task == "analyze_results":
            return RESULT_ANALYST_PROMPTS["analyze_results"].format(
                solution_id=kwargs.get("solution_id", "unknown"),
                delta_R=kwargs.get("delta_R", "N/A"),
                mutation_type=kwargs.get("mutation_type", "unknown"),
                solver_output=kwargs.get("solver_output", ""),
                mesh_metrics=kwargs.get("mesh_metrics", ""),
            )
        elif task == "compare_solutions":
            return RESULT_ANALYST_PROMPTS["compare_solutions"].format(
                solutions=kwargs.get("solutions", ""),
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def parse_response(self, response: str) -> ResultAnalysis:
        """Parse the LLM response into a ResultAnalysis."""
        analysis = ResultAnalysis(raw_response=response)

        # Check for convergence
        if re.search(r'(?:converge|success|complete)', response, re.IGNORECASE):
            if not re.search(r'(?:not|fail|did not)\s+converge', response, re.IGNORECASE):
                analysis.converged = True

        # Check for physical reasonableness
        if re.search(r'(?:reasonable|valid|physical)', response, re.IGNORECASE):
            if not re.search(r'(?:not|un)reasonable', response, re.IGNORECASE):
                analysis.physically_reasonable = True

        # Extract stress max
        stress_match = re.search(
            r'(?:Maximum stress|stress_max|max stress)[:\s]+([0-9.eE+-]+)',
            response, re.IGNORECASE
        )
        if stress_match:
            try:
                analysis.stress_max = float(stress_match.group(1))
            except ValueError:
                pass

        # Extract displacement max
        disp_match = re.search(
            r'(?:Maximum displacement|displacement_max|max displacement)[:\s]+([0-9.eE+-]+)',
            response, re.IGNORECASE
        )
        if disp_match:
            try:
                analysis.displacement_max = float(disp_match.group(1))
            except ValueError:
                pass

        # Extract success traits
        traits_match = re.search(
            r'\*\*Success Traits\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if traits_match:
            text = traits_match.group(1)
            analysis.success_traits = [
                line.strip().lstrip('- ').lstrip('* ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Extract recommendations
        rec_match = re.search(
            r'\*\*Recommendations\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if rec_match:
            text = rec_match.group(1)
            analysis.recommendations = [
                line.strip().lstrip('- ').lstrip('* ')
                for line in text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

        # Calculate parent quality score
        score = 0.0
        if analysis.converged:
            score += 0.5
        if analysis.physically_reasonable:
            score += 0.3
        if analysis.success_traits:
            score += 0.2 * min(1.0, len(analysis.success_traits) / 3)
        analysis.parent_quality_score = min(1.0, score)

        return analysis

    async def analyze_results(
        self,
        context: AgentContext,
        solution_id: str,
        delta_R: float,
        mutation_type: str,
        solver_output: str = "",
        mesh_metrics: str = "",
    ) -> ResultAnalysis:
        """
        Analyze simulation results.

        Args:
            context: Agent context
            solution_id: Solution identifier
            delta_R: Delta R value
            mutation_type: Type of mutation
            solver_output: Solver output/log
            mesh_metrics: Mesh quality metrics

        Returns:
            ResultAnalysis
        """
        return await self.execute(
            context,
            task="analyze_results",
            solution_id=solution_id,
            delta_R=delta_R,
            mutation_type=mutation_type,
            solver_output=solver_output,
            mesh_metrics=mesh_metrics,
        )

    async def compare_solutions(
        self,
        context: AgentContext,
        solutions: List[Dict[str, Any]],
    ) -> SolutionComparison:
        """
        Compare multiple solutions.

        Args:
            context: Agent context
            solutions: List of solution dicts with metrics

        Returns:
            SolutionComparison
        """
        # Format solutions for prompt
        solutions_str = ""
        for i, sol in enumerate(solutions, 1):
            solutions_str += f"\n### Solution {i}: {sol.get('id', 'unknown')}\n"
            solutions_str += f"- delta_R: {sol.get('delta_R', 'N/A')}\n"
            solutions_str += f"- Converged: {sol.get('converged', 'unknown')}\n"
            solutions_str += f"- Stress Max: {sol.get('stress_max', 'N/A')}\n"
            solutions_str += f"- Quality Score: {sol.get('quality_score', 'N/A')}\n"

        analysis = await self.execute(
            context,
            task="compare_solutions",
            solutions=solutions_str,
        )

        # Convert to SolutionComparison
        comparison = SolutionComparison(raw_response=analysis.raw_response)

        # Extract ranking from response
        ranking_match = re.search(
            r'\*\*Quality Ranking\*\*.*?(?:\n|\:)(.*?)(?=\*\*|$)',
            analysis.raw_response, re.DOTALL | re.IGNORECASE
        )
        if ranking_match:
            text = ranking_match.group(1)
            # Try to extract solution IDs
            ids = re.findall(r'(?:Solution\s+)?(\w+-?\w*)', text)
            comparison.ranking = ids[:len(solutions)]

        return comparison

    def analyze_metrics_locally(
        self,
        validation_report: Dict[str, Any],
        solver_completed: bool = False,
    ) -> ResultAnalysis:
        """
        Analyze metrics locally without LLM.

        Args:
            validation_report: Validation report dict
            solver_completed: Whether solver completed

        Returns:
            Basic ResultAnalysis
        """
        analysis = ResultAnalysis()
        analysis.converged = solver_completed and validation_report.get("is_valid", False)
        analysis.physically_reasonable = not validation_report.get("errors", [])

        # Extract any metrics from the report
        details = validation_report.get("details", {})
        if "jacobian_min" in details:
            analysis.success_traits.append(f"Min Jacobian: {details['jacobian_min']}")
        if "aspect_ratio_max" in details:
            analysis.success_traits.append(f"Max Aspect Ratio: {details['aspect_ratio_max']}")

        # Calculate score
        score = 0.0
        if analysis.converged:
            score += 0.5
        if analysis.physically_reasonable:
            score += 0.3
        if not validation_report.get("warnings", []):
            score += 0.2
        analysis.parent_quality_score = score

        return analysis
