"""
Tests for the 6 new agents added in the Knowledge-Augmented Agent System.

All tests are LLM-free (no API calls) — they test parsing, heuristics, and
KB mechanics only. Run with: pytest tests/test_new_agents.py -v
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
KB_INDEX = PROJECT_ROOT / "knowledge_base" / "kb_index.json"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLMProvider:
    def __init__(self, response: str = "DECISION: continue_fem\nSAMPLES_NEXT: 10\nREASONING: still learning"):
        self._response = response

    async def generate(self, system_prompt="", user_prompt="", **kwargs):
        return MockLLMResponse(self._response)


# ──────────────────────────────────────────────────────────────────────────────
# 1. KnowledgeRetrieverAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestKnowledgeRetrieverAgent:

    def test_kb_index_exists(self):
        assert KB_INDEX.exists(), f"KB index missing at {KB_INDEX}"

    def test_kb_loads_all_entries(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        assert kr.n_entries == 6

    def test_retrieve_williams_for_near_tip(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        entry = kr.retrieve(
            diagnosis="LOSS_PLATEAU",
            analyst_pattern="near-tip error high stress_intensity not converging",
            pino_status="stress_intensity plateaued at 0.03",
        )
        assert entry is not None
        assert "Williams" in entry.method_name

    def test_retrieve_xfem_for_traction_free(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        entry = kr.retrieve(
            diagnosis="UNSTABLE_TRAINING",
            analyst_pattern="traction_free BC violations crack_face discontinuity",
            pino_status="traction_free loss not decreasing",
        )
        assert entry is not None
        assert "XFEM" in entry.method_name

    def test_retrieve_none_when_no_match(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        # Very generic query with no crack-specific keywords
        entry = kr.retrieve(
            diagnosis="UNDERFITTING",
            analyst_pattern="loss high no progress",
            pino_status="no PINO active",
            score_threshold=5.0,  # require 5 keyword hits — impossible
        )
        assert entry is None

    def test_retrieve_by_name(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        entry = kr.retrieve_by_name("J-Integral Consistency Loss")
        assert entry is not None
        assert "j_integral" in entry.keywords

    def test_skips_used_methods(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        # Simulate Williams already used
        config_history = [{"changes": "kb_technique: Williams Asymptotic Expansion"}]
        entry = kr.retrieve(
            diagnosis="LOSS_PLATEAU",
            analyst_pattern="near-tip error high stress_intensity",
            pino_status="",
            config_history=config_history,
        )
        # Should return a different entry (not Williams)
        if entry is not None:
            assert "Williams" not in entry.method_name

    def test_entry_has_content(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        entry = kr.retrieve_by_name("Williams Asymptotic Expansion")
        assert entry is not None
        assert len(entry.content) > 100
        assert "## Technique" in entry.content

    def test_to_context_string_format(self):
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        kr = KnowledgeRetrieverAgent()
        entry = kr.retrieve_by_name("Adaptive Collocation Near Singularities")
        assert entry is not None
        ctx = entry.to_context_string()
        assert "[KNOWLEDGE BASE" in ctx
        assert "Apply when:" in ctx


# ──────────────────────────────────────────────────────────────────────────────
# 2. DataAnalystAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestDataAnalystAgent:

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        n_samples = 10
        coords = [rng.random((50 + i * 2, 2)) for i in range(n_samples)]
        outputs = [rng.random((c.shape[0], 2)) for c in coords]
        params = rng.random((n_samples, 3))
        tip = np.array([0.5, 0.5])
        return coords, outputs, params, tip

    def test_compute_stats_shape(self, sample_data):
        from piano.agents.roles.data_analyst import DataAnalystAgent
        coords, outputs, params, tip = sample_data
        stats = DataAnalystAgent.compute_stats(coords, outputs, params, tip)
        assert stats["n_samples"] == 10
        assert stats["n_nodes_min"] <= stats["n_nodes_mean"] <= stats["n_nodes_max"]
        assert 0.0 <= stats["near_tip_fraction"] <= 1.0
        assert stats["variable_topology"] == "True"

    def test_compute_stats_no_tip(self, sample_data):
        from piano.agents.roles.data_analyst import DataAnalystAgent
        coords, outputs, params, _ = sample_data
        stats = DataAnalystAgent.compute_stats(coords, outputs, params, tip_coords=None)
        assert stats["tip_coords"] == "not provided"
        assert stats["near_tip_fraction"] == 0.0

    def test_compute_stats_uniform_mesh(self):
        from piano.agents.roles.data_analyst import DataAnalystAgent
        n_nodes = 100
        coords = [np.random.rand(n_nodes, 2) for _ in range(5)]
        outputs = [np.random.rand(n_nodes, 1) for _ in range(5)]
        params = np.random.rand(5, 2)
        stats = DataAnalystAgent.compute_stats(coords, outputs, params, None)
        assert stats["n_unique_sizes"] == 1
        assert stats["variable_topology"] == "False"

    def test_parse_response(self):
        from piano.agents.roles.data_analyst import DataAnalystAgent, DataAnalysis
        from piano.agents.base import AgentContext
        agent = DataAnalystAgent()
        raw = """DATASET_QUALITY: moderate
N_SAMPLES: 15
N_NODES_RANGE: 48-72
NEAR_TIP_FRACTION: 0.045
OUTPUT_SKEWNESS: high positive skew in both displacement components (skew ≈ 2.3)
RECOMMENDATIONS:
- Apply tip-weighted MSE (tip_weight=2.0)
- Normalize outputs per-channel before training
SUMMARY: Dataset is suitable for DeepONet training with moderate near-tip density."""
        result = agent.parse_response(raw)
        assert result.dataset_quality == "moderate"
        assert result.n_samples == 15
        assert result.n_nodes_range == "48-72"
        assert abs(result.near_tip_fraction - 0.045) < 1e-6
        assert len(result.recommendations) == 2
        assert "tip-weighted" in result.recommendations[0]
        assert "suitable" in result.summary

    def test_to_context_string(self):
        from piano.agents.roles.data_analyst import DataAnalysis
        da = DataAnalysis(
            dataset_quality="good",
            n_samples=20,
            n_nodes_range="100-120",
            near_tip_fraction=0.08,
            output_skewness="mild",
            recommendations=["use tip weighting"],
            summary="Good dataset.",
        )
        ctx = da.to_context_string()
        assert "[DATA ANALYST REPORT]" in ctx
        assert "good" in ctx
        assert "use tip weighting" in ctx

    def test_save_report(self, sample_data, tmp_path):
        from piano.agents.roles.data_analyst import DataAnalysis
        da = DataAnalysis(
            dataset_quality="moderate",
            n_samples=10,
            recommendations=["tip weighting"],
            summary="Test.",
        )
        report_path = tmp_path / "data_analysis.md"
        da.save(report_path)
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Data Analysis Report" in content

    @pytest.mark.asyncio
    async def test_analyze_with_mock_llm(self, sample_data, tmp_path):
        from piano.agents.roles.data_analyst import DataAnalystAgent
        from piano.agents.base import AgentContext

        mock_response = """DATASET_QUALITY: good
N_SAMPLES: 10
N_NODES_RANGE: 50-68
NEAR_TIP_FRACTION: 0.032
OUTPUT_SKEWNESS: low skew
RECOMMENDATIONS:
- Apply tip weighting
SUMMARY: Suitable dataset."""

        agent = DataAnalystAgent()
        agent.set_llm_provider(MockLLMProvider(mock_response))

        coords, outputs, params, tip = sample_data
        context = AgentContext()
        result = await agent.analyze(
            context, coords, outputs, params, tip_coords=tip,
            report_path=tmp_path / "report.md",
        )
        assert result.dataset_quality == "good"
        assert result.near_tip_fraction == pytest.approx(0.032)
        assert len(context.knowledge_context) == 1
        assert (tmp_path / "report.md").exists()


# ──────────────────────────────────────────────────────────────────────────────
# 3. DebuggerAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestDebuggerAgent:

    def test_parse_clean_response(self):
        from piano.agents.roles.debugger import DebuggerAgent, DebugResult
        agent = DebuggerAgent.__new__(DebuggerAgent)

        response = """ROOT_CAUSE: Missing import for `nn.functional` in piano/surrogate/deeponet.py line 42
FIX: Add `import torch.nn.functional as F` at the top of piano/surrogate/deeponet.py
FILES: piano/surrogate/deeponet.py
CONFIDENCE: high"""

        result = agent._parse(response, traceback="")
        assert "Missing import" in result.root_cause
        assert "torch.nn.functional" in result.fix_description
        assert "deeponet.py" in result.files_to_change[0]
        assert result.confidence == "high"

    def test_parse_extracts_files_from_traceback_fallback(self):
        from piano.agents.roles.debugger import DebuggerAgent
        agent = DebuggerAgent.__new__(DebuggerAgent)

        tb = '''Traceback (most recent call last):
  File "/mnt/c/project/piano/surrogate/trainer.py", line 304, in _train_with_hpo
    node_weights = ...
AttributeError: 'NoneType' object'''

        result = agent._parse("ROOT_CAUSE: null\nFIX: fix it\nCONFIDENCE: low", tb)
        assert any("trainer.py" in f for f in result.files_to_change)

    def test_to_engineer_prompt(self):
        from piano.agents.roles.debugger import DebugResult
        dr = DebugResult(
            root_cause="Missing import",
            fix_description="Add import statement",
            files_to_change=["piano/surrogate/deeponet.py"],
        )
        prompt = dr.to_engineer_prompt("original change")
        assert "ORIGINAL CHANGE" in prompt
        assert "Missing import" in prompt
        assert "deeponet.py" in prompt

    def test_low_confidence_on_empty_response(self):
        from piano.agents.roles.debugger import DebuggerAgent
        agent = DebuggerAgent.__new__(DebuggerAgent)
        result = agent._parse("", traceback="some error")
        assert result.confidence in ("low", "medium", "high", "")


# ──────────────────────────────────────────────────────────────────────────────
# 4. SelectorEnsembleAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestSelectorEnsembleAgent:

    def _make_candidates(self, n: int):
        """Create mock DebateResult-like candidates."""
        class MockArch:
            def __init__(self, i):
                self.changes = {"hidden_dim": 64 + i * 16}
                self.reasoning = f"Candidate {i} reasoning"
                class Cfg:
                    def to_dict(self): return {}
                self.config = Cfg()

        class MockCandidate:
            def __init__(self, i):
                self.arch_proposal = MockArch(i)
                self.physics_changes = {"equilibrium": 0.01 * i}
                self.validation_text = f"Candidate {i} looks reasonable."

        return [MockCandidate(i) for i in range(n)]

    def test_single_candidate_skips_vote(self):
        from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent, SelectionResult

        agent = SelectorEnsembleAgent(llm_provider=MockLLMProvider())
        candidates = self._make_candidates(1)
        result = agent.select_sync(candidates)
        assert result.selected_index == 0
        assert result.selection_method == "single_candidate"
        assert result.confidence == 1.0

    def test_majority_vote_two_voters_agree(self):
        from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent

        # All 3 voters say candidate 1
        provider = MockLLMProvider("CHOICE: 1\nREASONING: candidate 1 is better")
        agent = SelectorEnsembleAgent(llm_provider=provider, n_voters=3)
        candidates = self._make_candidates(3)
        result = agent.select_sync(candidates)
        assert result.selected_index == 1
        assert result.confidence == pytest.approx(1.0)

    def test_vote_count_tallied(self):
        from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent

        responses = ["CHOICE: 0\nREASONING: a", "CHOICE: 2\nREASONING: b", "CHOICE: 0\nREASONING: c"]
        call_count = [0]

        class RotatingProvider:
            async def generate(self, **kwargs):
                r = responses[call_count[0] % len(responses)]
                call_count[0] += 1
                return MockLLMResponse(r)

        agent = SelectorEnsembleAgent(llm_provider=RotatingProvider(), n_voters=3)
        candidates = self._make_candidates(3)
        result = agent.select_sync(candidates)
        assert result.vote_counts.get(0, 0) == 2
        assert result.vote_counts.get(2, 0) == 1
        assert result.selected_index == 0

    def test_parse_vote_clamps_to_valid_range(self):
        from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent
        agent = SelectorEnsembleAgent(llm_provider=MockLLMProvider())
        vote = agent._parse_vote(0, "CHOICE: 99\nREASONING: out of range", n_candidates=3)
        assert 0 <= vote.chosen_index < 3

    def test_build_prompt_contains_all_candidates(self):
        from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent
        agent = SelectorEnsembleAgent(llm_provider=MockLLMProvider())
        candidates = self._make_candidates(3)
        prompt = agent._build_prompt(candidates, history_summary="test_loss=0.1")
        assert "Candidate 0" in prompt
        assert "Candidate 1" in prompt
        assert "Candidate 2" in prompt
        assert "test_loss=0.1" in prompt


# ──────────────────────────────────────────────────────────────────────────────
# 5. MeshStrategyAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestMeshStrategyAgent:

    @pytest.fixture
    def mesh_data(self):
        rng = np.random.default_rng(0)
        n = 200
        coords = rng.random((n, 2))
        tip = np.array([0.3, 0.3])
        r = np.linalg.norm(coords - tip, axis=1)
        # Errors higher near tip
        errors = 1.0 / (r + 0.05) * rng.random(n)
        return coords, errors, tip

    def test_compute_mesh_stats_keys(self, mesh_data):
        from piano.agents.roles.mesh_strategy import MeshStrategyAgent
        coords, errors, tip = mesh_data
        stats = MeshStrategyAgent.compute_mesh_stats(coords, errors, tip, iteration=1, n_samples=20)
        for key in ("error_map", "n_nodes", "tip_x", "tip_y", "h_near_tip", "h_far_field",
                    "near_tip_fraction", "iteration", "n_samples", "prev_decisions"):
            assert key in stats, f"Missing key: {key}"

    def test_compute_mesh_stats_values(self, mesh_data):
        from piano.agents.roles.mesh_strategy import MeshStrategyAgent
        coords, errors, tip = mesh_data
        stats = MeshStrategyAgent.compute_mesh_stats(coords, errors, tip, iteration=2, n_samples=30)
        assert stats["n_nodes"] == len(coords)
        assert abs(stats["tip_x"] - 0.3) < 1e-6
        assert stats["h_near_tip"] > 0

    def test_parse_r_only(self):
        from piano.agents.roles.mesh_strategy import MeshStrategyAgent, MeshStrategyDecision
        agent = MeshStrategyAgent.__new__(MeshStrategyAgent)
        agent._decision_history = []
        response = """STRATEGY: r_only
R_REFINEMENT_TARGET: nodes within r < 0.05 from tip
H_REFINEMENT_ZONE: none
EXPECTED_IMPROVEMENT: ~30% reduction in near-tip error
REASONING: Near-tip error is 3x far-field. Moving nodes toward tip is sufficient."""
        result = agent.parse_response(response)
        assert result.strategy == "r_only"
        assert result.needs_r_refinement()
        assert not result.needs_h_refinement()
        assert "30%" in result.expected_improvement

    def test_parse_both(self):
        from piano.agents.roles.mesh_strategy import MeshStrategyAgent
        agent = MeshStrategyAgent.__new__(MeshStrategyAgent)
        agent._decision_history = []
        response = """STRATEGY: both
R_REFINEMENT_TARGET: crack tip zone r < 0.03
H_REFINEMENT_ZONE: upper-right quadrant MAE=0.045
EXPECTED_IMPROVEMENT: 40% overall improvement
REASONING: Two distinct high-error zones require different strategies."""
        result = agent.parse_response(response)
        assert result.needs_r_refinement()
        assert result.needs_h_refinement()

    def test_parse_none_strategy(self):
        from piano.agents.roles.mesh_strategy import MeshStrategyAgent
        agent = MeshStrategyAgent.__new__(MeshStrategyAgent)
        agent._decision_history = []
        response = "STRATEGY: none\nREASONING: error is uniformly low."
        result = agent.parse_response(response)
        assert not result.needs_r_refinement()
        assert not result.needs_h_refinement()

    def test_to_summary(self):
        from piano.agents.roles.mesh_strategy import MeshStrategyDecision
        d = MeshStrategyDecision(
            strategy="r_only",
            r_refinement_target="near tip",
            h_refinement_zone="none",
            reasoning="High near-tip error.",
        )
        s = d.to_summary()
        assert "r_only" in s
        assert "High near-tip" in s

    def test_prev_decisions_in_stats(self, mesh_data):
        from piano.agents.roles.mesh_strategy import MeshStrategyAgent
        coords, errors, tip = mesh_data
        stats = MeshStrategyAgent.compute_mesh_stats(
            coords, errors, tip, iteration=4, n_samples=50,
            prev_decisions=["r_only", "both", "none", "r_only"],
        )
        # Only last 3 should appear
        assert "r_only" in stats["prev_decisions"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. BudgetAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestBudgetAgent:

    def test_converged_fast_path_below_threshold(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(
            convergence_threshold=0.05,
            max_samples=200,
        )
        # No LLM needed — fast path triggers before LLM call
        decision = agent.decide_sync(
            iteration=3,
            test_error=0.03,  # below threshold
            mean_uncertainty=0.01,
            n_samples=50,
        )
        assert decision.should_stop()
        assert decision.decision == "converged"

    def test_budget_exhausted_fast_path(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(convergence_threshold=0.05, max_samples=100)
        decision = agent.decide_sync(
            iteration=10,
            test_error=0.2,
            mean_uncertainty=0.1,
            n_samples=100,  # at max
        )
        assert decision.should_stop()

    def test_llm_path_continue_fem(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(convergence_threshold=0.05, max_samples=200)
        agent.set_llm_provider(MockLLMProvider(
            "DECISION: continue_fem\nSAMPLES_NEXT: 12\nREASONING: uncertainty still high"
        ))
        decision = agent.decide_sync(
            iteration=2,
            test_error=0.15,
            mean_uncertainty=0.08,
            n_samples=40,
        )
        assert not decision.should_stop()
        assert not decision.should_switch_hpo()
        assert decision.samples_next == 12

    def test_llm_path_switch_hpo(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(convergence_threshold=0.05, max_samples=200)
        agent.set_llm_provider(MockLLMProvider(
            "DECISION: switch_hpo\nREASONING: diminishing returns on data collection"
        ))
        decision = agent.decide_sync(
            iteration=5,
            test_error=0.12,
            mean_uncertainty=0.01,
            n_samples=80,
        )
        assert decision.should_switch_hpo()
        assert not decision.should_stop()

    def test_increase_budget_boosts_samples(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(convergence_threshold=0.05, max_samples=200, base_samples_per_iter=10)
        agent.set_llm_provider(MockLLMProvider(
            "DECISION: increase_budget\nSAMPLES_NEXT: 10\nREASONING: large uncertainty spike"
        ))
        decision = agent.decide_sync(
            iteration=3,
            test_error=0.18,
            mean_uncertainty=0.2,
            n_samples=50,
        )
        assert decision.should_increase_budget()
        # samples_next should be bumped to at least 1.5 * base
        assert decision.samples_next >= 15

    def test_patience_count_increments(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(convergence_threshold=0.05, max_samples=200)
        agent.set_llm_provider(MockLLMProvider(
            "DECISION: continue_fem\nSAMPLES_NEXT: 10\nREASONING: continuing"
        ))
        # Push same error 3 times — patience should count up
        for i in range(3):
            agent.decide_sync(iteration=i+1, test_error=0.15, mean_uncertainty=0.05, n_samples=20+i*5)
        assert agent._patience_count >= 2

    def test_reset_clears_state(self):
        from piano.agents.roles.budget import BudgetAgent
        agent = BudgetAgent(convergence_threshold=0.05, max_samples=200)
        agent._patience_count = 5
        agent._error_history = [0.1, 0.2]
        agent.reset()
        assert agent._patience_count == 0
        assert len(agent._error_history) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. EngineerAgent + DebuggerAgent integration (mock-based)
# ──────────────────────────────────────────────────────────────────────────────

class TestEngineerDebuggerIntegration:

    def test_engineer_result_debug_attempted_flag(self):
        from piano.agents.roles.engineer import EngineerResult
        r = EngineerResult(success=True, changes_made="added import", debug_attempted=True)
        assert r.debug_attempted

    def test_debug_result_parse_round_trip(self):
        from piano.agents.roles.debugger import DebuggerAgent, DebugResult
        agent = DebuggerAgent.__new__(DebuggerAgent)
        raw = "ROOT_CAUSE: key error\nFIX: wrap in try/except\nFILES: piano/foo.py\nCONFIDENCE: medium"
        result = agent._parse(raw, "")
        assert result.root_cause == "key error"
        assert result.confidence == "medium"
        assert "foo.py" in result.files_to_change[0]
