from __future__ import annotations

import pytest

from secmcp.config import DATA_DIR
from secmcp.data.loaders import agentdojo, agentharm, agenttraj_l, asb, injecagent


@pytest.mark.parametrize(
    ("name", "loader", "root"),
    [
        ("injecagent", injecagent.load, DATA_DIR / "InjecAgent"),
        ("agentharm", agentharm.load, DATA_DIR / "AgentHarm"),
        ("asb", asb.load, DATA_DIR / "ASB"),
        ("agenttraj_l", agenttraj_l.load, DATA_DIR / "AgentTraj-L"),
        ("agentdojo", agentdojo.load, DATA_DIR / "AgentDojo"),
    ],
)
def test_loader_returns_valid_samples(name, loader, root):
    samples = loader(root, max_samples=8)
    assert samples, name
    assert all(sample.source == name for sample in samples)
    assert all(sample.label in {0, 1} for sample in samples)
    assert all(sample.text.strip() for sample in samples)
    assert all(sample.kind for sample in samples)


def test_malicious_sources_have_positive_samples():
    loaders = [
        (injecagent.load, DATA_DIR / "InjecAgent"),
        (agentharm.load, DATA_DIR / "AgentHarm"),
        (asb.load, DATA_DIR / "ASB"),
        (agentdojo.load, DATA_DIR / "AgentDojo"),
    ]
    for loader, root in loaders:
        samples = loader(root, max_samples=200)
        assert any(sample.label == 1 for sample in samples)


def test_agentdojo_text_is_not_python_repr():
    """AgentDojo content is list[dict]; text must be plain text, not '[{...}]'."""
    samples = agentdojo.load(DATA_DIR / "AgentDojo", max_samples=10)
    assert samples
    for s in samples:
        assert "[{" not in s.text[:100], (
            f"AgentDojo text looks like a Python repr: {s.text[:120]!r}"
        )


def test_agenttraj_is_benign_only():
    samples = agenttraj_l.load(DATA_DIR / "AgentTraj-L", max_samples=20)
    assert samples
    assert {sample.label for sample in samples} == {0}
