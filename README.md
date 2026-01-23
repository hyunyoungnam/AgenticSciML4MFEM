# AgenticSciML for Abaqus
![Agentic4Abaqus Framework](./image/Agentic4Abaqus_framework.png)

## Framework Overview: AgenticSciML for Autonomous FEA Dataset Generation

### Phase 1: Analysis & Evaluation Contract
The process begins with the **Human User** defining the problem parameters (e.g., Plate with a hole) and specific requirements. The **Evaluator** agent then performs a "Contractual Setup" to ensure the base model is robust enough for mutation.
* **Guideline Generation**: Produces `Guideline.md` defining unit systems (e.g., SI), mesh safety bounds, and step increment limits.
* **Automated Scoring**: Develops `Evaluate.py` to quantify mesh integrity (Jacobian, Aspect ratio) and physical sanity (Reaction force balance).
* **Human Approval**: The evaluation criteria must be approved by the human user to synchronize the system's "success" definition with research goals.

### Phase 2: The Knowledge Funnel (Proposer Context)
Information from various sources is distilled into a **Proposer Context** to guide intelligent discovery.
* **Knowledge Retriever**: Injects domain expertise from the **Abaqus Manual** and fundamental **Physical Laws**.
* **Dynamic Memory**: Integrates **Failure Memory** (previous divergent logs) provided by the **Result Analyst** to avoid redundant computational waste.
* **Strategy Alignment**: Incorporates Design of Experiments (DOE) and surrogate insights to focus on critical physical regimes.

### Phase 3: Fan-out Generation (The Proposer-Critic Debate)
Instead of blind mutations, this phase employs an iterative **Proposer-Critic Loop** (repeated $XN$ times) to ensure only viable candidates proceed.
* **Proposer Agent**: Suggests geometric mutations (e.g., "increase radius!") or material property shifts.
* **Critic Agent**: Performs "pre-flight checks" on proposed `.inp` modifications. It flags potential issues such as "INP #27 will not converge due to distorted elements".
* **Outcome**: This adversarial interaction ensures that the subsequent execution phase uses high-probability-of-success models.

### Phase 4: Execution & Evolutionary Feedback Loop
The **Engineer** agent implements the final proposals and manages the computational resource.
* **Autonomous Debugging**: If a solver error occurs, the **Engineer** collaborates with a **Debugger** ($XN$ iterations) to fix `.inp` syntax or numerical divergence in real-time.
* **Dataset Production (Fan-out)**: The system generates a massive population of diverse, validated `.inp` files (`INP #1` to `INP #N`).
* **Evolutionary Selection**: The **Result Analyst** ranks datasets based on success/interest and feeds "Success Traits" back into **Phase 2**, triggering the next generation of data evolution.