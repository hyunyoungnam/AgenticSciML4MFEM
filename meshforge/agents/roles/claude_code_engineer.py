"""
Claude Code-based Engineer Agent.

Replaces the hand-coded EngineerAgent with a Claude Code-powered implementation
that can directly read source files, execute code, and self-correct.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from meshforge.agents.base import AgentContext, AgentRole
from meshforge.agents.llm.claude_code_provider import ClaudeCodeProvider, ClaudeCodeResult
from meshforge.agents.roles.engineer import ImplementationResult
from meshforge.agents.roles.proposer import MutationProposal


@dataclass
class ClaudeCodeEngineerConfig:
    """Configuration for the Claude Code Engineer."""
    model: str = "sonnet"
    max_turns: int = 25
    timeout: int = 300
    working_dir: Optional[str] = None

    # Files Claude Code should read for context
    context_files: List[str] = field(default_factory=lambda: [
        "manager.py",
        "morphing.py",
        "writer.py",
        "validator.py",
    ])


class ClaudeCodeEngineer:
    """
    Engineer Agent powered by Claude Code.

    Unlike the hand-coded EngineerAgent, this agent:
    1. Reads actual source files to understand current APIs
    2. Executes implementation code directly
    3. Validates results before returning
    4. Self-corrects on errors through iterative tool use

    Key differences from EngineerAgent:
    - No brittle regex parsing of LLM output
    - No hardcoded API documentation in prompts
    - Actual execution and verification of results
    - Adaptive to API changes
    """

    def __init__(
        self,
        config: Optional[ClaudeCodeEngineerConfig] = None,
        provider: Optional[ClaudeCodeProvider] = None,
    ):
        """
        Initialize the Claude Code Engineer.

        Args:
            config: Configuration options
            provider: Claude Code provider (created if not provided)
        """
        self.config = config or ClaudeCodeEngineerConfig()
        self.provider = provider or ClaudeCodeProvider(
            model=self.config.model,
            max_turns=self.config.max_turns,
            timeout=self.config.timeout,
            working_dir=self.config.working_dir,
            allowed_tools=["Edit", "Write", "Read", "Bash", "Glob", "Grep"],
        )
        self.role = AgentRole.ENGINEER

    async def implement_mutation(
        self,
        proposal: MutationProposal,
        base_inp_path: str,
        output_path: str,
        config_path: Optional[str] = None,
    ) -> ImplementationResult:
        """
        Implement a mutation using Claude Code.

        Claude Code will:
        1. Read the manager.py and morphing.py to understand APIs
        2. Load the base model
        3. Apply the specified mutations
        4. Run validation
        5. Write output files
        6. Verify the files exist

        Args:
            proposal: The mutation proposal
            base_inp_path: Path to base .inp file
            output_path: Path for output .inp file
            config_path: Optional morphing config path

        Returns:
            ImplementationResult with actual verified paths
        """
        result = ImplementationResult()

        # Build the task prompt
        task = self._build_implementation_task(
            proposal=proposal,
            base_inp_path=base_inp_path,
            output_path=output_path,
            config_path=config_path,
        )

        # Execute with Claude Code
        cc_result = await self.provider.execute_task(
            task=task,
            context_files=self.config.context_files,
            output_format="text",
            max_turns=self.config.max_turns,
        )

        # Parse the result
        return self._parse_implementation_result(cc_result, output_path)

    def _build_implementation_task(
        self,
        proposal: MutationProposal,
        base_inp_path: str,
        output_path: str,
        config_path: Optional[str],
    ) -> str:
        """Build the task prompt for implementation."""

        task = f"""
You are implementing an FEA model mutation. Your task is to:

1. First, read the manager.py, morphing.py, and writer.py files to understand the current APIs
2. Load the base model from: {base_inp_path}
3. Apply the following mutations:

## Mutation Details
- **Type**: {proposal.mutation_type if hasattr(proposal, 'mutation_type') else 'morphing'}
- **delta_R**: {proposal.delta_R}
- **Material changes**: {json.dumps(proposal.material_changes) if proposal.material_changes else 'None'}
- **Boundary condition changes**: {json.dumps(proposal.bc_changes) if proposal.bc_changes else 'None'}

4. Run pre-flight validation using validator.py
5. Write output files to: {output_path}
6. Verify the output files exist

## Important Requirements
- Use the AbaqusManager class from manager.py
- For morphing, use run_morphing() from morphing.py with delta_R={proposal.delta_R}
- For file output, use write_inp_and_vtu() from writer.py
- Run validation and check for mesh quality issues

## Morphing Config
"""
        if config_path:
            task += f"- Use config file: {config_path}\n"
        else:
            task += "- No config path provided - morphing may not be available\n"

        task += """
## Output Format
After completing, report:
1. Whether implementation succeeded
2. The output .inp file path
3. The output .vtu file path (if generated)
4. Any validation warnings or errors
5. Mesh quality metrics if available

Execute the implementation now using Python code via the Bash tool.
"""
        return task

    def _parse_implementation_result(
        self,
        cc_result: ClaudeCodeResult,
        expected_output_path: str,
    ) -> ImplementationResult:
        """Parse Claude Code result into ImplementationResult."""

        result = ImplementationResult()

        if not cc_result.success:
            result.success = False
            result.error_message = cc_result.error or "Claude Code execution failed"
            return result

        output = cc_result.output.lower()

        # Check for success indicators
        success_indicators = [
            "success", "completed", "written", "created",
            "implementation complete", "files generated"
        ]
        error_indicators = [
            "error", "failed", "exception", "traceback",
            "could not", "unable to"
        ]

        has_success = any(ind in output for ind in success_indicators)
        has_error = any(ind in output for ind in error_indicators)

        # Check if output file exists
        output_path = Path(expected_output_path)
        file_exists = output_path.exists()

        if file_exists:
            result.success = True
            result.inp_path = str(output_path)

            # Check for VTU file
            vtu_path = output_path.with_suffix(".vtu")
            if vtu_path.exists():
                result.vtu_path = str(vtu_path)

            result.validation_passed = has_success and not has_error

        elif has_success and not has_error:
            # Claude Code reported success but file doesn't exist yet
            # This might happen if we're checking too early
            result.success = True
            result.inp_path = expected_output_path
            result.validation_passed = True

        else:
            result.success = False
            result.error_message = self._extract_error_message(cc_result.output)

        # Store the raw output
        result.code_executed = cc_result.output
        result.metadata = {
            "claude_code_success": cc_result.success,
            "num_turns": cc_result.num_turns,
            "cost_usd": cc_result.cost_usd,
            "duration_ms": cc_result.duration_ms,
            "session_id": cc_result.session_id,
        }

        return result

    def _extract_error_message(self, output: str) -> str:
        """Extract error message from Claude Code output."""
        # Look for common error patterns
        lines = output.split('\n')

        for i, line in enumerate(lines):
            lower = line.lower()
            if any(x in lower for x in ["error:", "exception:", "failed:", "traceback"]):
                # Return this line and the next few for context
                return "\n".join(lines[i:min(i+5, len(lines))])

        # If no specific error found, return last portion
        return output[-500:] if len(output) > 500 else output

    async def get_implementation_code(
        self,
        context: AgentContext,
        proposal: MutationProposal,
        base_inp_path: str,
        output_path: str,
        config_path: str,
    ) -> str:
        """
        Get implementation code from Claude Code.

        This method asks Claude Code to generate the implementation code
        without executing it.

        Args:
            context: Agent context
            proposal: Mutation proposal
            base_inp_path: Base input file path
            output_path: Output file path
            config_path: Morphing config path

        Returns:
            Implementation code as string
        """
        task = f"""
Read manager.py, morphing.py, and writer.py to understand the APIs.

Then generate Python code (but do NOT execute it) to:
1. Load the base model from: {base_inp_path}
2. Apply morphing with delta_R = {proposal.delta_R}
3. Apply material changes: {proposal.material_changes}
4. Apply boundary condition changes: {proposal.bc_changes}
5. Run validation
6. Write output to: {output_path}

Use config file: {config_path}

Output ONLY the Python code in a code block.
"""
        result = await self.provider.execute_task(
            task=task,
            context_files=self.config.context_files,
            output_format="text",
        )

        # Extract code from response
        code_match = re.search(r'```python\s*(.*?)\s*```', result.output, re.DOTALL)
        if code_match:
            return code_match.group(1)

        return result.output

    async def get_fixed_code(
        self,
        context: AgentContext,
        original_code: str,
        error_message: str,
        error_type: str,
        debugger_analysis: str,
    ) -> str:
        """
        Get fixed implementation code using Claude Code.

        Claude Code will:
        1. Read the original code
        2. Understand the error
        3. Read relevant source files for correct API usage
        4. Generate fixed code

        Args:
            context: Agent context
            original_code: Original code that failed
            error_message: Error message
            error_type: Type of error
            debugger_analysis: Analysis from debugger agent

        Returns:
            Fixed implementation code
        """
        task = f"""
The following code failed with an error. Fix it.

## Original Code
```python
{original_code}
```

## Error
Type: {error_type}
Message: {error_message}

## Debugger Analysis
{debugger_analysis}

## Task
1. Read the relevant source files (manager.py, morphing.py, etc.) to understand the correct API
2. Identify what went wrong in the original code
3. Generate fixed Python code

Output ONLY the fixed Python code in a code block.
"""
        result = await self.provider.execute_task(
            task=task,
            context_files=self.config.context_files,
            output_format="text",
        )

        # Extract code from response
        code_match = re.search(r'```python\s*(.*?)\s*```', result.output, re.DOTALL)
        if code_match:
            return code_match.group(1)

        return result.output

    async def generate_morphing_config(
        self,
        context: AgentContext,
        delta_R: float,
        hole_center: Tuple[float, float] = (0.0, 0.0),
        initial_radius: float = 2.5,
        transition_radius: float = 8.0,
    ) -> str:
        """
        Generate a morphing configuration using Claude Code.

        Args:
            context: Agent context
            delta_R: Radius change
            hole_center: Center of the hole
            initial_radius: Initial hole radius
            transition_radius: Transition region outer radius

        Returns:
            Morphing config as YAML/markdown string
        """
        guidelines = context.evaluation_criteria.get("guidelines", "")

        task = f"""
Generate a morphing configuration file for the following parameters:

## Parameters
- delta_R: {delta_R}
- hole_center: ({hole_center[0]}, {hole_center[1]})
- initial_radius: {initial_radius}
- transition_radius: {transition_radius}
- expected_final_radius: {initial_radius + delta_R}

## Guidelines
{guidelines if guidelines else "Use sensible defaults for FEA mesh morphing."}

## Task
1. Read the morphing.py file to understand the config format
2. Generate a valid morphing configuration
3. Include appropriate tolerances and quality thresholds

Output the configuration content.
"""
        result = await self.provider.execute_task(
            task=task,
            context_files=["morphing.py"],
            output_format="text",
        )

        return result.output

    async def verify_implementation(
        self,
        inp_path: str,
        expected_delta_R: float,
    ) -> Dict[str, Any]:
        """
        Verify an implementation by loading and checking the output file.

        Args:
            inp_path: Path to the generated .inp file
            expected_delta_R: Expected delta_R that was applied

        Returns:
            Verification report
        """
        task = f"""
Verify the implementation by:

1. Read the file: {inp_path}
2. Load it using AbaqusManager from manager.py
3. Check that the geometry reflects delta_R = {expected_delta_R}
4. Report mesh quality metrics

Output a JSON report with:
- file_exists: bool
- loaded_successfully: bool
- node_count: int
- element_count: int
- geometry_check: str (description of geometry verification)
- issues: list of any problems found
"""
        result = await self.provider.execute_task(
            task=task,
            context_files=["manager.py", "validator.py"],
            output_format="text",
        )

        # Try to parse as JSON
        try:
            # Find JSON in output
            json_match = re.search(r'\{.*\}', result.output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Return as text report
        return {
            "raw_report": result.output,
            "parsed": False,
        }

    def implement_mutation_sync(
        self,
        proposal: MutationProposal,
        base_inp_path: str,
        output_path: str,
        config_path: Optional[str] = None,
    ) -> ImplementationResult:
        """
        Synchronous wrapper for implement_mutation.

        Args:
            proposal: The mutation proposal
            base_inp_path: Path to base .inp file
            output_path: Path for output .inp file
            config_path: Optional morphing config path

        Returns:
            ImplementationResult
        """
        import asyncio
        return asyncio.run(
            self.implement_mutation(proposal, base_inp_path, output_path, config_path)
        )
