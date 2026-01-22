# AgenticSciML for Abaqus

## Framework Overview: AgenticSciML for Autonomous FEA Dataset Generation

This framework automates the generation of high-fidelity Finite Element Analysis (FEA) datasets by employing a collaborative multi-agent system. Unlike traditional automated scripts, this system utilizes evolutionary reasoning to refine geometry and material parameters, ensuring physical validity and data diversity.

## Phase 1: Multi-Faceted Analysis & Evaluation Contract

The process begins with the **Input Analyzer**, which performs a "Structural Chunking" of the human-provided base `.inp` file. It deconstructs the flat text into logical objects: Nodes, Elements, Materials, and Steps.

Concurrently, the **Evaluator** establishes the "Success Criteria" beyond mere completion. It defines a multi-metric contract including:

- **Mesh Integrity**: Aspect ratio, Jacobian, and distortion limits.
- **Physical Sanity**: Reaction force balance and energy conservation (e.g., Internal vs. Artificial energy).
- **Numerical Stability**: Convergence rates and singularity detection.

## Phase 2: The Knowledge Funnel (Proposer Context)

At the heart of the system is the **Context Funnel**, where diverse information streams converge to guide the next discovery:

- **Analysis Base**: Current statistical trends and DOE (Design of Experiments) results.
- **Knowledge Base**: A static library of Abaqus documentation and physical constraints (e.g., minimum thickness-to-hole ratios).
- **Dynamic Memory**: A "Failure Mode Registry" that stores insights from previous failed simulations to prevent redundant errors.
- **Surrogate Insights**: Meta-learning data that suggests which parameter spaces are likely to yield emergent physical behaviors.

## Phase 3: Tiered Modification & Fan-Out Generation

The framework employs a **Two-Tier Modifier Suite** to handle different levels of complexity:

- **Morphing Agent (High-Complexity)**: Responsible for large-scale coordinate transformations. It uses geometric algorithms to modify hole radii or plate dimensions while maintaining mesh quality.
- **Parameter Agent (Low-Complexity)**: Handles rule-based changes such as material law constants, boundary condition magnitudes, or step increments.

The system then "Fans-out", generating a population of diverse candidate `.inp` files from a single parent, significantly increasing the exploration of the design space.

## Phase 4: Execution & Evolutionary Feedback Loop

The **Abaqus Executor** manages the solver runs, monitoring `.sta` and `.msg` files in real-time. Successful runs are passed to the **Result Analyst**, which extracts ODB data and generates contour plots.

The most critical feature is the **Evolutionary Loop**:

- **Selection**: The Analyst filters "high-value" data points (e.g., simulations that captured onset of plasticity).
- **Feedback**: These successful traits are fed back into the Knowledge Funnel, triggering a new iteration of refined modifications. This ensures the generated dataset isn't just random, but strategically evolved to cover critical physical regimes.
