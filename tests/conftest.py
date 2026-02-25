"""Shared pytest fixtures."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set minimal env vars so config loads without real API keys
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")


@pytest.fixture
def mock_openai_llm():
    """Patch ChatOpenAI so no real API calls are made in unit tests."""
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Test LLM response"))
        mock_instance.with_structured_output = MagicMock(return_value=mock_instance)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_agent_state():
    """Return a minimal AgentState for testing."""
    from app.agents.state import initial_state
    return initial_state("What is Apple's current stock price and recent news?")
