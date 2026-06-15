"""Tests for global residual-memory storage and application."""

import csv

import pytest
import torch
from torch import nn

from src.jury.memory import (
    apply_global_memory_to_model,
    memory_to_metadata,
    save_memory_metadata,
    save_vote_stats,
    vote_stats_to_metadata,
)


class WrappedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs)


class MemoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(2, 2, bias=False)
        self.value = WrappedLinear()
        with torch.no_grad():
            self.query.weight.zero_()
            self.value.base_layer.weight.zero_()


def _memory_atom(
    atom_id: str,
    layer_name: str,
    lambda_g: float = 2.0,
    u: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
) -> dict:
    return {
        "memory_atom_id": atom_id,
        "source_atom_id": f"source_{atom_id}",
        "round_idx": 2,
        "layer_name": layer_name,
        "u": torch.tensor([1.0, 0.0]) if u is None else u,
        "v": torch.tensor([0.0, 1.0]) if v is None else v,
        "sigma": 3.0,
        "lambda_g": lambda_g,
        "p_accept": 0.75,
        "p_reject": 0.25,
        "p_abstain": 0.0,
        "accept_count": 3,
        "reject_count": 1,
        "abstain_count": 0,
        "importance": abs(lambda_g) * 0.5,
    }


def test_applies_memory_to_direct_and_wrapped_weights_with_scale() -> None:
    model = MemoryModel()
    memory = {
        "query": [_memory_atom("query_a", "query", 2.0)],
        "value": [
            _memory_atom("value_a", "value", 1.0),
            _memory_atom(
                "value_b",
                "value",
                2.0,
                u=torch.tensor([0.0, 1.0]),
                v=torch.tensor([1.0, 0.0]),
            ),
        ],
    }

    apply_global_memory_to_model(model, memory, scale=0.5)

    torch.testing.assert_close(
        model.query.weight,
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        model.value.base_layer.weight,
        torch.tensor([[0.0, 0.5], [1.0, 0.0]]),
    )


def test_application_validates_all_updates_before_mutating_model() -> None:
    model = MemoryModel()
    memory = {
        "query": [_memory_atom("valid", "query")],
        "missing": [_memory_atom("missing", "missing")],
    }

    with pytest.raises(ValueError, match="does not contain"):
        apply_global_memory_to_model(model, memory)

    torch.testing.assert_close(model.query.weight, torch.zeros_like(model.query.weight))


def test_rejects_bad_memory_vector_shape() -> None:
    model = MemoryModel()
    bad = _memory_atom("bad", "query", u=torch.ones(3))
    with pytest.raises(ValueError, match="outer.*shape"):
        apply_global_memory_to_model(model, {"query": [bad]})


def test_memory_metadata_and_csv_omit_vectors(tmp_path) -> None:
    memory = {"query": [_memory_atom("atom", "query")]}

    metadata = memory_to_metadata(memory)

    assert metadata[0]["u_shape"] == [2]
    assert metadata[0]["v_shape"] == [2]
    assert "u" not in metadata[0]
    assert "v" not in metadata[0]

    output_path = tmp_path / "nested" / "memory.csv"
    save_memory_metadata(memory, output_path)
    with output_path.open(newline="", encoding="utf-8") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)
        assert reader.fieldnames == list(metadata[0])
    assert rows[0]["memory_atom_id"] == "atom"
    assert "tensor" not in output_path.read_text(encoding="utf-8")


def test_vote_stats_metadata_and_empty_csv(tmp_path) -> None:
    stats = [{
        "round_idx": 1,
        "atom_id": "atom",
        "layer_name": "query",
        "num_clients": 4,
        "accept_count": 2,
        "reject_count": 1,
        "abstain_count": 1,
        "p_accept": 0.5,
        "p_reject": 0.25,
        "p_abstain": 0.25,
    }]
    assert vote_stats_to_metadata(stats) == stats

    output_path = tmp_path / "empty" / "votes.csv"
    save_vote_stats([], output_path)
    with output_path.open(newline="", encoding="utf-8") as output_file:
        reader = csv.DictReader(output_file)
        assert reader.fieldnames is not None
        assert list(reader) == []
