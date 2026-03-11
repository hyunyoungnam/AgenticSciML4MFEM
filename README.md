# meshforge

**meshforge** is a multi-agent AI system for autonomous MFEM mesh generation for AI/ML training datasets. It uses evolutionary tree search with structured debate between AI agents to generate diverse, validated mesh files through intelligent mesh morphing.

Stop wasting time manually creating mesh files for your ML training datasets.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [The Four Phases](#the-four-phases)
- [Agent Roles](#agent-roles)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [Usage Examples](#usage-examples)
- [Core Pipeline](#core-pipeline)

---

## Quick Start

```bash
# 1. Install meshforge
pip install meshforge

# 2. Set API keys (for real LLM usage)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# 3. Run with test mode (no API calls needed)
meshforge inputs/beam-quad.mesh --test --config configs/quarter_plate_with_hole_morphing.md

# 4. Check outputs
ls outputs/gen_0/
```

---

## Installation

### From PyPI (Recommended)

```bash
pip install meshforge
```

### From Source

```bash
git clone https://github.com/qjiang/meshforge.git
cd meshforge
pip install -e .
```

### Prerequisites
- Python 3.9+
- (Optional) PyMFEM for MFEM solver execution

### Optional Dependencies

```bash
# Install with visualization support
pip install meshforge[viz]

# Install with development tools
pip install meshforge[dev]
```

---

## How It Works

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLUTIONARY LOOP                                    │
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ Phase 1  │───▶│ Phase 2  │───▶│ Phase 3  │───▶│ Phase 4  │             │
│   │ Analysis │    │Knowledge │    │ Debate   │    │Execution │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │                                               │                     │
│        │         Evaluator creates                     │                     │
│        │         Guideline.md &                        ▼                     │
│        │         Evaluate.py            ┌─────────────────────────┐         │
│        │                                │   Generated .mesh files  │         │
│        │                                │   with morphed meshes    │         │
│        │                                └─────────────────────────┘         │
│        │                                               │                     │
│        │                                               ▼                     │
│        │                                ┌─────────────────────────┐         │
│        └───────────────────────────────│   Solution Tree          │         │
│                  Feedback               │   (Best solutions       │         │
│                                         │    become parents)      │         │
│                                         └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Evolutionary Tree Search**: Solutions evolve over generations. Best solutions become "parents" for the next generation.

2. **Multi-Agent Debate**: Before implementing any change, a Proposer agent suggests mutations and a Critic agent validates them through 4 rounds of debate.

3. **Mesh Morphing**: The system changes hole sizes, geometries, etc. using IDW (Inverse Distance Weighting) while preserving mesh topology and quality. Node roles (moving/anchor/morphing) are classified dynamically based on `delta_R`.

4. **Knowledge-Guided**: Agents use a curated FEA knowledge base to make intelligent decisions and avoid known failure patterns.

---

## The Four Phases

### Phase 1: Analysis & Evaluation Contract

**Purpose**: Understand the base model and establish quality criteria.

```
Human User                    Evaluator Agent
     │                              │
     │   "Plate with hole"          │
     ├─────────────────────────────▶│
     │                              │
     │                              ▼
     │                    ┌─────────────────┐
     │                    │ Analyze model   │
     │                    │ - Element types │
     │                    │ - Mesh bounds   │
     │                    │ - Materials     │
     │                    └────────┬────────┘
     │                             │
     │                             ▼
     │                    ┌─────────────────┐
     │                    │ Generate:       │
     │                    │ • Guideline.md  │
     │                    │ • Evaluate.py   │
     │                    └────────┬────────┘
     │                             │
     │   Approval request          │
     │◀────────────────────────────┤
     │                             │
     │   "Approved"                │
     ├─────────────────────────────▶│
```

**Outputs**:
- `Guideline.md`: Mesh quality bounds, allowable delta_R range, material constraints
- `Evaluate.py`: Scoring script for solutions (Jacobian, aspect ratio checks)

---

### Phase 2: Knowledge Funnel

**Purpose**: Gather context to make intelligent mutation decisions.

```
┌─────────────────────────────────────────────────────────┐
│                   KNOWLEDGE FUNNEL                       │
│                                                          │
│  ┌────────────────┐   ┌────────────────┐                │
│  │ FEA Knowledge  │   │ Failure Memory │                │
│  │ Base (70+      │   │ (Past errors,  │                │
│  │ entries)       │   │ bad delta_R)   │                │
│  └───────┬────────┘   └───────┬────────┘                │
│          │                    │                          │
│          └────────┬───────────┘                          │
│                   ▼                                      │
│          ┌────────────────┐                              │
│          │ Proposer       │                              │
│          │ Context        │  "delta_R > 1.5 caused      │
│          │                │   element inversion in      │
│          │                │   generation 2"             │
│          └────────────────┘                              │
└─────────────────────────────────────────────────────────┘
```

**Knowledge Sources**:
- LLM built-in knowledge: FEA best practices (no static knowledge base needed)
- `failure_memory.json`: Dynamically updated with past failures

---

### Phase 3: Proposer-Critic Debate (4 Rounds)

**Purpose**: Ensure only viable mutations proceed through adversarial validation.

```
Round 1: Initial Proposal
┌─────────────┐                      ┌─────────────┐
│  Proposer   │  "delta_R = 0.75"    │   Critic    │
│  (GPT-4)    │─────────────────────▶│  (Claude)   │
│             │                      │             │
│             │◀─────────────────────│  "LEAN_     │
│             │  Concerns: mesh      │   APPROVE"  │
│             │  quality near hole   │             │
└─────────────┘                      └─────────────┘

Round 2: Refinement
┌─────────────┐                      ┌─────────────┐
│  Proposer   │  "Added monitoring"  │   Critic    │
│             │─────────────────────▶│             │
│             │◀─────────────────────│  "LEAN_     │
│             │  "Acceptable"        │   APPROVE"  │
└─────────────┘                      └─────────────┘

Round 3: Synthesis
┌─────────────┐                      ┌─────────────┐
│  Proposer   │  Final proposal      │   Critic    │
│             │─────────────────────▶│             │
└─────────────┘                      └─────────────┘

Round 4: Final Vote
┌─────────────┐                      ┌─────────────┐
│  Proposer   │                      │   Critic    │
│             │◀─────────────────────│  "APPROVE"  │
│             │  or "REJECT"         │   ✓ or ✗    │
└─────────────┘                      └─────────────┘
```

**Outcomes**:
- `APPROVE`: Proceed to Phase 4
- `REJECT`: Solution marked as rejected, skip execution

---

### Phase 4: Execution & Feedback

**Purpose**: Implement approved mutations and validate results.

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXECUTION LOOP                            │
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │ Engineer │────▶│ Morphing │────▶│Validator │               │
│   │ (GPT-4)  │     │ Algorithm│     │          │               │
│   └──────────┘     └──────────┘     └────┬─────┘               │
│                                          │                       │
│                    ┌─────────────────────┼─────────────────────┐│
│                    │                     │                     ││
│                    ▼                     ▼                     ││
│              ┌──────────┐          ┌──────────┐               ││
│              │ SUCCESS  │          │  FAILED  │               ││
│              │          │          │          │               ││
│              │ .mesh    │          │ Debugger │               ││
│              │ .vtu     │          │ attempts │               ││
│              │ generated│          │ fix (x3) │               ││
│              └──────────┘          └──────────┘               ││
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Result Analyst  │
                    │  - Compute score │
                    │  - Update tree   │
                    │  - Feed back to  │
                    │    Phase 2       │
                    └──────────────────┘
```

**Outputs**:
- Morphed `.mesh` files
- `.vtu` files for visualization
- Metrics (Jacobian, aspect ratio, convergence)

---

## Agent Roles

| Agent | LLM | Role | Temperature |
|-------|-----|------|-------------|
| **Evaluator** | Claude | Analyzes base model, creates Guideline.md & Evaluate.py | 0.2 |
| **Proposer** | GPT-4 | Suggests mutations (delta_R, materials, BCs) | 0.7 |
| **Critic** | Claude | Validates proposals, pre-flight checks | 0.2 |
| **Engineer** | GPT-4 | Implements mutations using morphing.py | 0.1 |
| **Debugger** | Claude | Diagnoses and fixes solver errors | 0.3 |
| **Result Analyst** | GPT-4 | Analyzes results, computes metrics | 0.4 |

---

## Configuration

### Main Config Files

```
configs/
├── evolution_config.yaml          # Evolution parameters
├── agent_config.yaml              # Agent settings
└── quarter_plate_with_hole_morphing.md  # Morphing rules
```

### evolution_config.yaml

```yaml
evolution:
  max_generations: 20      # Number of evolutionary generations
  population_size: 10      # Solutions per generation
  num_parents: 3           # Parents for next generation

evaluation:
  preflight:
    min_jacobian: 0.1      # Minimum acceptable Jacobian
    max_aspect_ratio: 10.0 # Maximum acceptable aspect ratio
  solver:
    run_solver: false      # Run MFEM solver?
    timeout: 3600          # Solver timeout (seconds)
```

### agent_config.yaml

```yaml
agents:
  proposer:
    model: "gpt-4-turbo"
    temperature: 0.7
  critic:
    model: "claude-3-opus-20240229"
    temperature: 0.2

debate:
  num_rounds: 4
  consensus_threshold: 0.7
```

### Morphing Config (quarter_plate_with_hole_morphing.md)

Defines how mesh morphing works:

```yaml
geometry:
  hole_center: [0.0, 0.0]
  initial_hole_radius: 2.5
  transition_outer_radius: 8.0

symmetry:
  x_symmetry:
    axis: 0              # Nodes on x=0 only move in Y
    tolerance: 1.0e-6
  y_symmetry:
    axis: 1              # Nodes on y=0 only move in X
    tolerance: 1.0e-6

regions:
  hole_boundary:
    role: moving         # These nodes move by delta_R
  transition:
    role: morphing       # These nodes interpolate (IDW)
  far_field:
    role: anchor         # These nodes stay fixed
```

---

## Directory Structure

```
meshforge/
├── meshforge/                   # Main package
│   ├── __init__.py             # Public API
│   ├── cli.py                  # CLI entry point
│   ├── morphing.py             # IDW mesh morphing
│   ├── schema.py               # Data structure definitions
│   │
│   ├── mesh/                   # Mesh format handlers
│   │   ├── base.py             # Abstract MeshManager
│   │   └── mfem_manager.py     # MFEM mesh implementation
│   │
│   ├── solvers/                # Solver interfaces
│   │   ├── base.py             # Abstract SolverInterface
│   │   └── mfem_solver.py      # MFEM solver implementation
│   │
│   ├── agents/                 # AI Agent implementations
│   │   ├── base.py             # BaseAgent class
│   │   ├── llm/                # LLM providers (OpenAI, Anthropic)
│   │   ├── roles/              # Agent implementations
│   │   ├── prompts/            # Agent prompts
│   │   └── debate/             # Debate controller
│   │
│   ├── orchestration/          # Workflow management
│   ├── evolution/              # Evolutionary search
│   ├── evaluation/             # Solution evaluation
│   ├── knowledge/              # Knowledge base
│   └── configs/                # Configuration files
│
├── inputs/                     # Example input .mesh files
├── tests/                      # Test suite
├── pyproject.toml              # Package configuration
├── LICENSE                     # MIT License
└── README.md
```

---

## Usage Examples

### Basic Run (Test Mode)

```bash
# Uses mock LLM providers (no API keys needed)
meshforge inputs/beam-quad.mesh --test \
    --config configs/quarter_plate_with_hole_morphing.md
```

### Run with Real LLMs

```bash
# Set API keys first
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run 5 generations with population of 5
meshforge inputs/beam-quad.mesh \
    --config configs/quarter_plate_with_hole_morphing.md \
    --generations 5 \
    --population 5 \
    --output outputs/my_run
```

### Dry Run (Initialize Only)

```bash
# Just run Phase 1 to generate Guideline.md and Evaluate.py
meshforge inputs/beam-quad.mesh --test --dry-run
```

### Run with MFEM Solver

```bash
# Actually run MFEM solver on generated .mesh files
meshforge inputs/beam-quad.mesh \
    --config configs/quarter_plate_with_hole_morphing.md \
    --run-solver
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | Morphing config (.md) | None |
| `--generations`, `-g` | Number of generations | From YAML (20) |
| `--population`, `-p` | Population size | From YAML (10) |
| `--output`, `-o` | Output directory | `outputs` |
| `--test` | Use mock LLM providers | False |
| `--dry-run` | Initialize only | False |
| `--run-solver` | Run MFEM solver | False |
| `--knowledge` | Knowledge base path | (built-in LLM knowledge) |
| `--log-level` | Logging level | `INFO` |

---

## Core Pipeline

The low-level mesh processing pipeline:

```
input.mesh → MFEMManager → Morphing → MFEMManager.save() → output.mesh
```

### 1. MeshManager (`mesh/base.py`)
Abstract base class defining the mesh interface:
- Load/save mesh files
- Access nodes and coordinates
- Access elements and connectivity
- Get bounding box and dimensions

### 2. MFEMManager (`mesh/mfem_manager.py`)
MFEM implementation using PyMFEM:
- Reads MFEM `.mesh` format
- Supports 2D and 3D meshes
- Handles various element types (triangles, quads, tets, hexes)
- Provides mesh refinement capabilities

### 3. Morphing (`morphing.py`)
Applies IDW-based mesh morphing:
- **Moving nodes** (role=0): Hole boundary, displaced by delta_R
- **Anchor nodes** (role=2): Far field, stay fixed
- **Morphing nodes** (role=1): Transition zone, interpolated via IDW

Node classification is computed dynamically based on `delta_R` and stored in a `MorphingContext` object for internal use.

#### Standalone Morphing Usage

```bash
# Basic morphing
python -m meshforge.morphing inputs/beam-quad.mesh configs/quarter_plate_with_hole_morphing.md 0.5

# With debug VTU for ParaView visualization
python -m meshforge.morphing inputs/beam-quad.mesh configs/quarter_plate_with_hole_morphing.md 0.5 --debug

# With interactive PyVista preview
python -m meshforge.morphing inputs/beam-quad.mesh configs/quarter_plate_with_hole_morphing.md 0.5 --preview
```

#### Python API

```python
from meshforge import MFEMManager, apply_morphing

# Load and modify a model
manager = MFEMManager("model.mesh")
apply_morphing(manager, config_path="morphing.md", delta_r=0.5)

# Save morphed mesh
manager.save("output.mesh")
```

#### Debug VTU Visualization (ParaView)

When using `--debug`, a `*_debug.vtu` file is generated with PointData fields:

| Field | Description |
|-------|-------------|
| `MorphingRole` | 0=moving, 1=morphing, 2=anchor |
| `NodeID` | Original node ID from mesh |

To visualize in ParaView:
1. Open `outputs/beam-quad_morphed_debug.vtu`
2. Set **Coloring** to `MorphingRole`
3. Use **Selection Display Inspector** to hover and inspect node values

#### PyVista Dependency

For debug visualization, install PyVista:
```bash
pip install pyvista
```

### 4. Solver Integration (`solvers/mfem_solver.py`)
Run MFEM finite element analysis:
- Linear elasticity
- Heat transfer
- Returns solver results with metrics

---

## Example Output

After running:
```bash
meshforge inputs/beam-quad.mesh --test --config configs/quarter_plate_with_hole_morphing.md
```

You get:
```
outputs/
├── gen_0/
│   ├── a1b2c3d4-....mesh   # Morphed mesh (hole radius +30%)
│   └── a1b2c3d4-....vtu    # Visualization file
├── Guideline.md            # Generated quality guidelines
├── Evaluate.py             # Generated scoring script
├── solution_tree.json      # Full evolution tree
├── failure_memory.json     # Recorded failures
└── run_summary.json        # Run statistics
```

### Sample run_summary.json

```json
{
  "status": "completed",
  "total_generations": 1,
  "total_solutions": 1,
  "best_solutions": [
    {
      "id": "a1b2c3d4-...",
      "genome": { "delta_R": 0.75 },
      "status": "converged",
      "metrics": {
        "jacobian_min": 0.571,
        "aspect_ratio_max": 1.40,
        "preflight_score": 1.0
      }
    }
  ]
}
```

---

## Troubleshooting

### "No API key found"
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
# Or use --test mode for mock providers
```

### "Morphing config not found"
```bash
# Make sure to specify the morphing config
meshforge inputs/beam-quad.mesh --config configs/quarter_plate_with_hole_morphing.md
```

### "delta_R not applied"
- Check that morphing config path is correct
- Verify delta_R is within allowable range (see Guideline.md)

---

## References

- Q.Jiang & G.Karniadakis (2025): AgenticSciML Framework
- [MFEM Documentation](https://mfem.org/): Finite Element Methods Library
- IDW Morphing: Inverse Distance Weighting for mesh deformation
