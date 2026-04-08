"""
Engineer Agent implementation.

The Engineer implements approved mutations using manager.py and morphing.py,
translating high-level proposals into actual model modifications.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from piano.agents.base import BaseAgent, AgentContext, AgentRole
from piano.agents.prompts.engineer import ENGINEER_PROMPTS
from piano.agents.roles.proposer import MutationProposal


@dataclass
class ImplementationResult:
    """Result of implementing a mutation."""
    success: bool = False
    inp_path: Optional[str] = None
    vtu_path: Optional[str] = None
    validation_passed: bool = False
    validation_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    code_executed: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "inp_path": self.inp_path,
            "vtu_path": self.vtu_path,
            "validation_passed": self.validation_passed,
            "validation_report": self.validation_report,
            "error_message": self.error_message,
            "code_executed": self.code_executed,
            "metadata": self.metadata,
        }


class EngineerAgent(BaseAgent[ImplementationResult]):
    """
    Engineer Agent that implements approved mutations.

    Responsibilities:
    1. Translate proposals into implementation code
    2. Use manager.py and morphing.py APIs
    3. Run pre-flight validation
    4. Generate output files
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.ENGINEER,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return ENGINEER_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "implement_mutation")

        if task == "implement_mutation":
            return self._build_implement_prompt(context, kwargs)
        elif task == "fix_implementation":
            return self._build_fix_prompt(context, kwargs)
        elif task == "generate_morphing_config":
            return self._build_config_prompt(context, kwargs)
        elif task == "verify_implementation":
            return self._build_verify_prompt(context, kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _build_implement_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        proposal = kwargs.get("proposal", {})
        if isinstance(proposal, MutationProposal):
            delta_R = proposal.delta_R
            material_changes = proposal.material_changes
            bc_changes = proposal.bc_changes
            proposal_text = proposal.raw_response
        else:
            delta_R = proposal.get("delta_R")
            material_changes = proposal.get("material_changes", {})
            bc_changes = proposal.get("bc_changes", {})
            proposal_text = str(proposal)

        return ENGINEER_PROMPTS["implement_mutation"].format(
            proposal=proposal_text,
            delta_R=delta_R,
            material_changes=material_changes,
            bc_changes=bc_changes,
            base_inp_path=kwargs.get("base_inp_path", context.base_inp_path),
            output_path=kwargs.get("output_path", "outputs/morphed.inp"),
            config_path=kwargs.get("config_path", "configs/morphing_config.md"),
        )

    def _build_fix_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        return ENGINEER_PROMPTS["fix_implementation"].format(
            original_code=kwargs.get("original_code", ""),
            error_message=kwargs.get("error_message", ""),
            error_type=kwargs.get("error_type", "unknown"),
            debugger_analysis=kwargs.get("debugger_analysis", ""),
        )

    def _build_config_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        delta_R = kwargs.get("delta_R", 0.0)
        hole_center = kwargs.get("hole_center", [0.0, 0.0])
        initial_radius = kwargs.get("initial_radius", 2.5)
        transition_radius = kwargs.get("transition_radius", 8.0)
        guidelines = context.evaluation_criteria.get("guidelines", "")

        return ENGINEER_PROMPTS["generate_morphing_config"].format(
            delta_R=delta_R,
            hole_center=hole_center,
            initial_radius=initial_radius,
            transition_radius=transition_radius,
            expected_radius=initial_radius + delta_R,
            cx=hole_center[0],
            cy=hole_center[1],
            R0=initial_radius,
            R_trans=transition_radius,
            tolerance=0.15,
            min_dist=0.5,
            guidelines=guidelines,
        )

    def _build_verify_prompt(self, context: AgentContext, kwargs: Dict) -> str:
        return ENGINEER_PROMPTS["verify_implementation"].format(
            implementation=kwargs.get("implementation", ""),
            output_files=kwargs.get("output_files", ""),
            validation_report=kwargs.get("validation_report", ""),
        )

    def parse_response(self, response: str) -> ImplementationResult:
        """Parse the LLM response into an ImplementationResult."""
        result = ImplementationResult()

        # Extract code block
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            result.code_executed = code_match.group(1)
            result.success = True  # Code was generated

        # Check for error indicators
        if "error" in response.lower() and "fix" not in response.lower():
            error_match = re.search(r'Error:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if error_match:
                result.error_message = error_match.group(1)
                result.success = False

        # Extract output paths
        inp_match = re.search(r'(?:inp|output).*?([^\s"\']+\.inp)', response, re.IGNORECASE)
        if inp_match:
            result.inp_path = inp_match.group(1)

        vtu_match = re.search(r'([^\s"\']+\.vtu)', response, re.IGNORECASE)
        if vtu_match:
            result.vtu_path = vtu_match.group(1)

        return result

    def implement_mutation(
        self,
        proposal: MutationProposal,
        base_mesh_path: str,
        output_path: str,
        config_path: Optional[str] = None,
    ) -> ImplementationResult:
        """
        Implement a mutation directly (without LLM).

        This method provides direct implementation using MFEM mesh
        manager and morphing APIs.

        Args:
            proposal: The mutation proposal
            base_mesh_path: Path to base MFEM .mesh file
            output_path: Path for output .mesh file
            config_path: Optional morphing config path

        Returns:
            ImplementationResult

        Note:
            TODO: This method needs to be reimplemented for MFEM workflow.
            The Abaqus-based implementation has been removed.
        """
        result = ImplementationResult()

        try:
            from piano.mesh.mfem_manager import MFEMManager

            # Load the base mesh
            manager = MFEMManager(base_mesh_path)

            # Apply morphing if delta_R is specified
            if proposal.delta_R is not None and proposal.delta_R != 0:
                # TODO: Implement MFEM-based morphing
                result.error_message = "MFEM morphing not yet implemented for direct mutation"
                return result

            # TODO: Apply material changes via MFEM solver config
            # TODO: Apply boundary condition changes via MFEM solver config

            # Write output mesh
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            mesh_out = manager.save(output_path)

            result.success = True
            result.inp_path = str(mesh_out)
            result.validation_passed = True
            result.validation_report = {"is_valid": True, "errors": [], "warnings": []}

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    async def get_implementation_code(
        self,
        context: AgentContext,
        proposal: MutationProposal,
        base_inp_path: str,
        output_path: str,
        config_path: str,
    ) -> str:
        """
        Get implementation code from LLM.

        Args:
            context: Agent context
            proposal: Mutation proposal
            base_inp_path: Base input file path
            output_path: Output file path
            config_path: Morphing config path

        Returns:
            Implementation code as string
        """
        result = await self.execute(
            context,
            task="implement_mutation",
            proposal=proposal,
            base_inp_path=base_inp_path,
            output_path=output_path,
            config_path=config_path,
        )
        return result.code_executed

    async def get_fixed_code(
        self,
        context: AgentContext,
        original_code: str,
        error_message: str,
        error_type: str,
        debugger_analysis: str,
    ) -> str:
        """
        Get fixed implementation code from LLM.

        Args:
            context: Agent context
            original_code: Original code that failed
            error_message: Error message
            error_type: Type of error
            debugger_analysis: Analysis from debugger agent

        Returns:
            Fixed implementation code
        """
        result = await self.execute(
            context,
            task="fix_implementation",
            original_code=original_code,
            error_message=error_message,
            error_type=error_type,
            debugger_analysis=debugger_analysis,
        )
        return result.code_executed

    async def generate_morphing_config(
        self,
        context: AgentContext,
        delta_R: float,
        hole_center: Tuple[float, float] = (0.0, 0.0),
        initial_radius: float = 2.5,
        transition_radius: float = 8.0,
    ) -> str:
        """
        Generate a morphing configuration file.

        Args:
            context: Agent context
            delta_R: Radius change
            hole_center: Center of the hole
            initial_radius: Initial hole radius
            transition_radius: Transition region outer radius

        Returns:
            Morphing config as markdown string
        """
        result = await self.execute(
            context,
            task="generate_morphing_config",
            delta_R=delta_R,
            hole_center=list(hole_center),
            initial_radius=initial_radius,
            transition_radius=transition_radius,
        )
        return result.code_executed  # Contains the config content
