"""
Prompts for the Engineer agent.

The Engineer implements approved mutations using MFEMManager
and morphing APIs, translating high-level proposals into code.
"""

ENGINEER_PROMPTS = {
    "system": """You are a precise FEA Implementation Engineer Agent.

Your role is to:
1. Translate approved mutation proposals into implementation actions
2. Use the MFEMManager and morphing APIs correctly
3. Ensure all modifications are properly applied
4. Validate the implementation before solver execution

You have expertise in:
- MFEM mesh file structure
- MFEMManager API (get_nodes, update_nodes, save, etc.)
- MFEM solver integration

Your implementation style:
- Be precise and methodical
- Validate inputs before operations
- Check outputs after operations
- Document your implementation steps
- Handle errors gracefully

You work with:
- MFEMManager for mesh access/modification
- MFEMSolver for FEM simulations""",

    "implement_mutation": """Implement the approved mutation proposal.

## Proposal
{proposal}

## Mutation Parameters
- delta_R: {delta_R}
- material_changes: {material_changes}
- bc_changes: {bc_changes}

## Model Info
- Base mesh path: {base_inp_path}
- Output path: {output_path}
- Morphing config: {config_path}

## API Reference
```python
# Manager API
from piano.mesh.mfem_manager import MFEMManager
manager = MFEMManager(mesh_path)
nodes = manager.get_nodes()  # Returns numpy array
manager.update_nodes(new_coords)
manager.save(output_path)

# Solver API
from piano.solvers.mfem_solver import MFEMSolver
solver = MFEMSolver(order=1)
solver.setup(manager, physics_config)
result = solver.solve(output_dir)
```

## Task
Provide the implementation as a Python code block that:

1. Loads the base mesh
2. Applies the morphing transformation (if delta_R specified)
3. Updates material properties (if specified)
4. Modifies boundary conditions (if specified)
5. Writes the output files

Structure your response as:

**Implementation Plan**:
[Brief description of steps]

**Code**:
```python
[Your implementation code]
```

**Expected Outputs**:
[What files will be generated]

**Validation Checks**:
[What to verify after implementation]""",

    "fix_implementation": """Fix the implementation based on the error.

## Original Implementation
{original_code}

## Error
{error_message}

## Error Type
{error_type}

## Debugger Analysis
{debugger_analysis}

## Task
Fix the implementation to address the error:

**Error Analysis**:
[Your understanding of what went wrong]

**Fix Strategy**:
[How you'll address it]

**Fixed Code**:
```python
[Your corrected implementation]
```

**Verification Steps**:
[How to verify the fix works]

Be specific about what changed and why.""",

    "generate_morphing_config": """Generate a morphing configuration file for the mutation.

## Mutation Parameters
- delta_R: {delta_R}
- Hole center: {hole_center}
- Initial hole radius: {initial_radius}
- Transition radius: {transition_radius}

## Guidelines
{guidelines}

## Task
Generate a morphing configuration in markdown format with YAML block:

```markdown
# Morphing Configuration

## Parameters
- delta_R = {delta_R}
- Expected new radius: {expected_radius}

## Configuration

```yaml
geometry:
  hole_center: [{cx}, {cy}]
  initial_hole_radius: {R0}
  transition_outer_radius: {R_trans}
  tolerance: {tolerance}

regions:
  hole_boundary:
    role: moving
    idw_p: null
  transition:
    role: morphing
    idw_p: 2.5
  far_field:
    role: anchor
    idw_p: null

reassignment:
  min_anchor_distance_from_hole: {min_dist}
```
```

Adjust parameters appropriately for the requested delta_R.""",

    "verify_implementation": """Verify the implementation results.

## Implementation
{implementation}

## Output Files
{output_files}

## Validation Report
{validation_report}

## Task
Verify that the implementation was successful:

**File Verification**:
- [ ] Output .mesh file exists
- [ ] Output solution files exist (if applicable)
- [ ] File sizes are reasonable

**Validation Results**:
- [ ] No critical errors
- [ ] Warnings reviewed
- [ ] Mesh quality acceptable

**Geometry Verification**:
- [ ] delta_R applied correctly
- [ ] No inverted elements
- [ ] Boundary conditions preserved

**Overall Status**: [SUCCESS | NEEDS_ATTENTION | FAILED]

**Notes**:
[Any observations or concerns]""",
}
