"""Tests for server-side global Jury memory updates."""

import pytest
import torch
from torch import nn

from src.jury.server_jury import (
    apply_global_memory_to_model,
    count_votes_for_atoms,
    update_global_memory,
)


def _atom(atom_id: str, layer: str = "query", sigma: float = 2.0) -> dict:
    return {
        "atom_id": atom_id,
        "round_idx": 3,
        "layer_name": layer,
        "u": torch.tensor([1.0, 0.0]),
        "v": torch.tensor([0.0, 1.0]),
        "sigma": torch.tensor(sigma),
    }


def _vote(atom_id: str, client_id: int, vote: str, layer: str = "query") -> dict:
    return {
        "client_id": client_id,
        "task_name": "sst2",
        "atom_id": atom_id,
        "layer_name": layer,
        "score": 0.0,
        "vote": vote,
    }


def _config(**jury_values: object) -> dict:
    jury = {
        "theta_high": 0.5,
        "eta_memory": 2.0,
        "alpha_memory": 0.5,
        "global_memory_budget_per_layer": 2,
    }
    jury.update(jury_values)
    return {"data": {"num_clients": 4}, "jury": jury}


def _existing(atom_id: str, importance: float, lambda_g: float = 4.0) -> dict:
    return {
        "memory_atom_id": atom_id,
        "source_atom_id": f"source_{atom_id}",
        "round_idx": 1,
        "layer_name": "query",
        "u": torch.tensor([0.0, 1.0]),
        "v": torch.tensor([1.0, 0.0]),
        "sigma": 1.0,
        "lambda_g": lambda_g,
        "p_accept": 0.75,
        "p_reject": 0.25,
        "p_abstain": 0.0,
        "accept_count": 3,
        "reject_count": 1,
        "abstain_count": 0,
        "importance": importance,
    }


def test_counts_votes_and_treats_missing_clients_as_abstentions() -> None:
    selected = {"query": [_atom("a"), _atom("b")]}
    records = [
        _vote("a", 0, "accept"),
        _vote("a", 1, "reject"),
        _vote("a", 2, "abstain"),
        _vote("b", 0, "accept"),
    ]

    stats = count_votes_for_atoms(selected, records, num_clients=4)

    assert [row["atom_id"] for row in stats] == ["a", "b"]
    assert stats[0]["accept_count"] == 1
    assert stats[0]["reject_count"] == 1
    assert stats[0]["abstain_count"] == 2
    assert stats[1]["p_accept"] == 0.25
    assert stats[1]["p_abstain"] == 0.75
    assert sum(stats[0][key] for key in ("p_accept", "p_reject", "p_abstain")) == 1


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_vote("unknown", 0, "accept")], "unknown atom_id"),
        ([_vote("a", 0, "maybe")], "must be accept"),
        ([_vote("a", 0, "accept"), _vote("a", 0, "reject")], "duplicate votes"),
        ([_vote("a", 0, "accept", layer="value")], "expected 'query'"),
    ],
)
def test_rejects_invalid_vote_records(records: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        count_votes_for_atoms({"query": [_atom("a")]}, records, 4)


def test_rejects_duplicate_selected_atom_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate selected atom_id"):
        count_votes_for_atoms({"query": [_atom("a"), _atom("a")]}, [], 4)


def test_rejects_too_many_distinct_clients_across_atoms() -> None:
    selected = {"query": [_atom("a"), _atom("b")]}
    records = [
        _vote("a", 0, "accept"),
        _vote("a", 1, "accept"),
        _vote("b", 2, "accept"),
    ]
    with pytest.raises(ValueError, match="more clients than num_clients"):
        count_votes_for_atoms(selected, records, num_clients=2)


def test_updates_memory_with_decay_consensus_and_budget_without_mutating_input() -> None:
    old_low = _existing("old_low", importance=999.0, lambda_g=2.0)
    old_high = _existing("old_high", importance=999.0, lambda_g=6.0)
    memory = {"query": [old_low, old_high]}
    original_u = old_low["u"].clone()
    selected = {
        "query": [
            _atom("accepted", sigma=2.0),
            _atom("rejected", sigma=10.0),
        ]
    }
    votes = [
        _vote("accepted", 0, "accept"),
        _vote("accepted", 1, "accept"),
        _vote("accepted", 2, "accept"),
        _vote("accepted", 3, "reject"),
        _vote("rejected", 0, "accept"),
        _vote("rejected", 1, "reject"),
        _vote("rejected", 2, "reject"),
        _vote("rejected", 3, "abstain"),
    ]

    updated, stats = update_global_memory(
        memory,
        selected,
        votes,
        _config(),
        round_idx=4,
    )

    assert len(stats) == 2
    assert [atom["memory_atom_id"] for atom in updated["query"]] == [
        "old_high",
        "memory_round_4:accepted",
    ]
    accepted = updated["query"][1]
    assert accepted["lambda_g"] == pytest.approx(2.0)
    assert accepted["importance"] == pytest.approx(1.0)
    assert updated["query"][0]["lambda_g"] == pytest.approx(3.0)
    assert updated["query"][0]["importance"] == pytest.approx(1.5)
    assert old_low["lambda_g"] == 2.0
    torch.testing.assert_close(old_low["u"], original_u)
    assert updated["query"][0]["u"] is not old_high["u"]


def test_skips_nonpositive_strength_and_preserves_stable_budget_ties() -> None:
    first = _existing("first", importance=1.0, lambda_g=2.0)
    second = _existing("second", importance=1.0, lambda_g=2.0)
    selected = {"query": [_atom("zero", sigma=0.0)]}
    votes = [_vote("zero", client, "accept") for client in range(4)]

    updated, _ = update_global_memory(
        {"query": [first, second]},
        selected,
        votes,
        _config(alpha_memory=1.0),
        round_idx=4,
    )

    assert [atom["memory_atom_id"] for atom in updated["query"]] == ["first", "second"]


@pytest.mark.parametrize(
    "jury_update",
    [
        {"theta_high": 1.1},
        {"eta_memory": -1.0},
        {"alpha_memory": 1.1},
        {"global_memory_budget_per_layer": 0},
    ],
)
def test_rejects_invalid_memory_config(jury_update: dict) -> None:
    with pytest.raises(ValueError, match="config"):
        update_global_memory({}, {}, [], _config(**jury_update), round_idx=1)


def test_server_apply_wrapper_uses_unit_scale() -> None:
    model = nn.Module()
    model.query = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.query.weight.zero_()
    atom = _existing("atom", importance=1.0, lambda_g=2.0)

    apply_global_memory_to_model(model, {"query": [atom]})

    torch.testing.assert_close(
        model.query.weight,
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
    )
