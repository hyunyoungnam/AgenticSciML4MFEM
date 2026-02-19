# AgenticSciML for Abaqus

<img src="./image/AgenticSciML4Abaqus_framework.png" alt="Agentic4Abaqus Framework" width="700">

**AgenticSciML** is a multi-agent AI system for autonomous Abaqus FEA dataset generation. It uses evolutionary tree search with structured debate between AI agents to generate diverse, validated `.inp` files through intelligent mesh morphing.

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
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (for real LLM usage)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# 3. Run with test mode (no API calls needed)
python run_agentic.py inputs/BaseInp2D.inp --test --config configs/quarter_plate_with_hole_morphing.md

# 4. Check outputs
ls outputs/gen_0/
```

---

## Installation

### Prerequisites
- Python 3.9+
- (Optional) Abaqus installation for solver execution

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy`, `scipy` - Numerical computation
- `openai`, `anthropic` - LLM providers
- `pyyaml` - Configuration parsing
- `aiohttp` - Async HTTP

Optional packages:
- `pyvista` - Debug visualization of morphing roles in ParaView

```bash
# Install optional visualization support
pip install pyvista
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
│        │                                │   Generated .inp files   │         │
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
- `knowledge/fea_knowledge.json`: Curated FEA best practices
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
│              │ .inp     │          │ Debugger │               ││
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
- Morphed `.inp` files
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
    run_solver: false      # Run Abaqus solver?
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
AgenticSciML4Abaqus/
├── run_agentic.py              # Main entry point
├── manager.py                  # Abaqus model manager
├── morphing.py                 # IDW mesh morphing
├── parser.py                   # .inp file parser
├── writer.py                   # .inp file writer
├── validator.py                # Mesh validation
│
├── agents/                     # AI Agent implementations
│   ├── base.py                 # BaseAgent class
│   ├── llm/                    # LLM providers (OpenAI, Anthropic)
│   ├── roles/                  # Agent implementations
│   │   ├── evaluator.py
│   │   ├── proposer.py
│   │   ├── critic.py
│   │   ├── engineer.py
│   │   ├── debugger.py
│   │   └── result_analyst.py
│   ├── prompts/                # Agent prompts
│   └── debate/                 # Debate controller
│
├── orchestration/              # Workflow management
│   ├── orchestrator.py         # Main coordinator
│   └── phases.py               # Phase controllers
│
├── evolution/                  # Evolutionary search
│   ├── solution.py             # Solution data model
│   ├── tree.py                 # Solution tree
│   └── selection.py            # Parent selection
│
├── evaluation/                 # Solution evaluation
│   ├── pipeline.py
│   ├── preflight.py
│   └── metrics.py
│
├── knowledge/                  # Knowledge base
│   ├── base.py
│   ├── fea_knowledge.json      # FEA best practices
│   └── failure_memory.py
│
├── configs/                    # Configuration files
│   ├── evolution_config.yaml
│   ├── agent_config.yaml
│   └── quarter_plate_with_hole_morphing.md
│
├── inputs/                     # Input .inp files
│   └── BaseInp2D.inp
│
└── outputs/                    # Generated outputs
    ├── gen_0/                  # Generation 0 solutions
    │   ├── <solution-id>.inp
    │   ├── <solution-id>.vtu
    │   └── <solution-id>_debug.vtu  # Debug VTU (with --debug)
    ├── solution_tree.json
    └── run_summary.json
```

---

## Usage Examples

### Basic Run (Test Mode)

```bash
# Uses mock LLM providers (no API keys needed)
python run_agentic.py inputs/BaseInp2D.inp --test \
    --config configs/quarter_plate_with_hole_morphing.md
```

### Run with Real LLMs

```bash
# Set API keys first
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run 5 generations with population of 5
python run_agentic.py inputs/BaseInp2D.inp \
    --config configs/quarter_plate_with_hole_morphing.md \
    --generations 5 \
    --population 5 \
    --output outputs/my_run
```

### Dry Run (Initialize Only)

```bash
# Just run Phase 1 to generate Guideline.md and Evaluate.py
python run_agentic.py inputs/BaseInp2D.inp --test --dry-run
```

### Run with Abaqus Solver

```bash
# Actually run Abaqus solver on generated .inp files
python run_agentic.py inputs/BaseInp2D.inp \
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
| `--run-solver` | Run Abaqus solver | False |
| `--knowledge` | Knowledge base path | `knowledge/fea_knowledge.json` |
| `--log-level` | Logging level | `INFO` |

---

## Core Pipeline

The low-level mesh processing pipeline:

```
input.inp → Parser → Manager → Morphing → Writer → output.inp
```

### 1. Parser (`parser.py`)
Reads `.inp` file and splits into keyword chunks (*Node, *Element, *Nset, etc.)

### 2. Manager (`manager.py`)
Builds in-memory model with API for reading/updating:
- Nodes and coordinates
- Elements and connectivity
- Node/element sets
- Materials and properties
- Boundary conditions

### 3. Morphing (`morphing.py`)
Applies IDW-based mesh morphing:
- **Moving nodes** (role=0): Hole boundary, displaced by delta_R
- **Anchor nodes** (role=1): Far field, stay fixed
- **Morphing nodes** (role=2): Transition zone, interpolated via IDW

Node classification is computed dynamically based on `delta_R` and stored in a `MorphingContext` object for internal use (never written to INP).

#### Standalone Morphing Usage

```bash
# Basic morphing
python morphing.py inputs/BaseInp2D.inp configs/quarter_plate_with_hole_morphing.md 0.5

# With debug VTU for ParaView visualization
python morphing.py inputs/BaseInp2D.inp configs/quarter_plate_with_hole_morphing.md 0.5 --debug

# With interactive PyVista preview
python morphing.py inputs/BaseInp2D.inp configs/quarter_plate_with_hole_morphing.md 0.5 --preview
```

#### Debug VTU Visualization (ParaView)

When using `--debug`, a `*_debug.vtu` file is generated with PointData fields:

| Field | Description |
|-------|-------------|
| `MorphingRole` | 0=moving, 1=anchor, 2=morphing |
| `NodeID` | Original node ID from INP |

To visualize in ParaView:
1. Open `outputs/OutputInp2D_morphed_debug.vtu`
2. Set **Coloring** to `MorphingRole`
3. Use **Selection Display Inspector** to hover and inspect node values

#### PyVista Dependency

For debug visualization, install PyVista:
```bash
pip install pyvista
```

### 4. Writer (`writer.py`)
Writes modified model back to `.inp` format with:
- Updated node coordinates
- Preserved element connectivity
- Regenerated sets

---

## Example Output

After running:
```bash
python run_agentic.py inputs/BaseInp2D.inp --test --config configs/quarter_plate_with_hole_morphing.md
```

You get:
```
outputs/
├── gen_0/
│   ├── a1b2c3d4-....inp    # Morphed mesh (hole radius +30%)
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
python run_agentic.py inputs/BaseInp2D.inp --config configs/quarter_plate_with_hole_morphing.md
```

### "delta_R not applied"
- Check that morphing config path is correct
- Verify delta_R is within allowable range (see Guideline.md)

---

## References

- Q.Jiang & G.Karniadakis (2025): AgenticSciML Framework
- Abaqus Documentation: Element types, mesh quality metrics
- IDW Morphing: Inverse Distance Weighting for mesh deformation
