"""Integration tests for the LangGraph workflow."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")


# ── State tests ───────────────────────────────────────────────────────────────

class TestAgentState:
    def test_initial_state_structure(self):
        from app.agents.state import initial_state
        state = initial_state("What is AAPL stock price?")
        assert state["query"] == "What is AAPL stock price?"
        assert state["ticker"] is None
        assert state["selected_agents"] == []
        assert state["news_results"] == []
        assert state["rag_context"] == []
        assert state["agent_errors"] == []
        assert state["final_answer"] is None

    def test_merge_optional_dict_both_none(self):
        from app.agents.state import merge_optional_dict
        assert merge_optional_dict(None, None) is None

    def test_merge_optional_dict_one_none(self):
        from app.agents.state import merge_optional_dict
        assert merge_optional_dict(None, {"a": 1}) == {"a": 1}
        assert merge_optional_dict({"a": 1}, None) == {"a": 1}

    def test_merge_optional_dict_both(self):
        from app.agents.state import merge_optional_dict
        result = merge_optional_dict({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_optional_dict_overwrite(self):
        from app.agents.state import merge_optional_dict
        result = merge_optional_dict({"a": 1, "x": 0}, {"a": 99})
        assert result["a"] == 99


# ── Routing tests ─────────────────────────────────────────────────────────────

class TestRouting:
    def test_route_to_agents_returns_sends(self):
        from app.agents.graph import route_to_agents
        from app.agents.state import initial_state
        from langgraph.constants import Send

        state = initial_state("test query")
        state["selected_agents"] = ["market_data", "news"]

        sends = route_to_agents(state)
        assert len(sends) == 2
        assert all(isinstance(s, Send) for s in sends)
        assert {s.node for s in sends} == {"market_data", "news"}

    def test_route_to_agents_fallback_all(self):
        from app.agents.graph import route_to_agents
        from app.agents.state import initial_state

        state = initial_state("test query")
        state["selected_agents"] = []  # empty → fallback

        sends = route_to_agents(state)
        assert len(sends) == 4  # all agents

    def test_route_filters_invalid_agents(self):
        from app.agents.graph import route_to_agents
        from app.agents.state import initial_state

        state = initial_state("test query")
        state["selected_agents"] = ["market_data", "invalid_agent"]

        sends = route_to_agents(state)
        assert len(sends) == 1
        assert sends[0].node == "market_data"


# ── Supervisor node tests ─────────────────────────────────────────────────────

class TestSupervisorNode:
    @pytest.mark.asyncio
    async def test_supervisor_sets_state_fields(self):
        from app.agents.supervisor import SupervisorDecision

        mock_decision = SupervisorDecision(
            ticker="AAPL",
            company="Apple Inc.",
            selected_agents=["market_data", "news", "rag", "sentiment"],
            reasoning="Full company analysis requested.",
        )

        with patch("app.agents.supervisor._supervisor_llm") as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=mock_decision)
            from app.agents.supervisor import supervisor_node
            from app.agents.state import initial_state

            state = initial_state("Tell me everything about Apple Inc.")
            result = await supervisor_node(state)

        assert result["ticker"] == "AAPL"
        assert result["company"] == "Apple Inc."
        assert set(result["selected_agents"]) == {"market_data", "news", "rag", "sentiment"}

    @pytest.mark.asyncio
    async def test_supervisor_fallback_on_error(self):
        with patch("app.agents.supervisor._supervisor_llm") as mock_llm:
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
            from app.agents.supervisor import supervisor_node
            from app.agents.state import initial_state

            state = initial_state("test query")
            result = await supervisor_node(state)

        # Should fall back to all agents
        assert len(result["selected_agents"]) == 4
        assert len(result["agent_errors"]) == 1


# ── Graph compilation test ─────────────────────────────────────────────────────

class TestGraphCompilation:
    def test_graph_compiles(self):
        """Smoke test: graph builds without errors."""
        with (
            patch("app.agents.supervisor._supervisor_llm"),
            patch("app.agents.market_data._agent"),
            patch("app.agents.news._agent"),
            patch("app.agents.rag._agent"),
            patch("app.agents.sentiment._agent"),
            patch("app.agents.graph._synthesizer_llm"),
            patch("langchain_openai.ChatOpenAI"),
        ):
            from app.agents.graph import build_graph
            graph = build_graph()
            assert graph is not None
            assert "supervisor" in graph.nodes
            assert "synthesize" in graph.nodes
