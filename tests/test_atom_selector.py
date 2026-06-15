"""Tests for singular-value-based atom selection."""

import csv

import pytest
import torch

from src.jury.atom_selector import (
    save_selected_atoms_metadata,
    select_topk_atoms_by_sigma,
    selected_atoms_to_metadata,
)


def _config(top_k_per_layer: int = 2) -> dict:
    return {"jury": {"top_k_per_layer": top_k_per_layer}}


def _atom(
    atom_id: str,
    layer_name: str,
    sigma: object,
    client_id: int,
    rank_id: int,
) -> dict:
    return {
        "atom_id": atom_id,
        "round_idx": 4,
        "client_id": client_id,
        "task_name": "qnli",
        "layer_name": layer_name,
        "rank_id": rank_id,
        "u": torch.ones(3),
        "v": torch.ones(5),
        "sigma": sigma,
        "source_lora_rank": 2,
    }


def test_selects_topk_per_layer_and_preserves_layer_order() -> None:
    first = _atom("first", "layer_b", torch.tensor(1.0), 2, 0)
    second = _atom("second", "layer_a", 4.0, 3, 0)
    third = _atom("third", "layer_b", 3.0, 1, 1)
    fourth = _atom("fourth", "layer_b", 2.0, 1, 0)

    selected = select_topk_atoms_by_sigma(
        [first, second, third, fourth],
        _config(2),
    )

    assert list(selected) == ["layer_b", "layer_a"]
    assert selected["layer_b"] == [third, fourth]
    assert selected["layer_a"] == [second]
    assert selected["layer_b"][0] is third
    assert selected["layer_b"][0]["u"] is third["u"]


def test_uses_client_and_rank_tie_breakers_with_stable_exact_ties() -> None:
    client_two = _atom("client_two", "layer", 2.0, 2, 0)
    rank_two = _atom("rank_two", "layer", 2.0, 1, 2)
    exact_first = _atom("exact_first", "layer", 2.0, 1, 1)
    exact_second = _atom("exact_second", "layer", 2.0, 1, 1)

    selected = select_topk_atoms_by_sigma(
        [client_two, rank_two, exact_first, exact_second],
        _config(4),
    )

    assert [atom["atom_id"] for atom in selected["layer"]] == [
        "exact_first",
        "exact_second",
        "rank_two",
        "client_two",
    ]


def test_metadata_is_flat_ordered_and_excludes_vectors() -> None:
    first = _atom("first", "layer_a", torch.tensor(3.5), 1, 0)
    second = _atom("second", "layer_b", 2, 2, 1)

    metadata = selected_atoms_to_metadata(
        {"layer_a": [first], "layer_b": [second]}
    )

    assert [row["atom_id"] for row in metadata] == ["first", "second"]
    assert metadata[0] == {
        "round_idx": 4,
        "atom_id": "first",
        "layer_name": "layer_a",
        "client_id": 1,
        "task_name": "qnli",
        "rank_id": 0,
        "sigma": 3.5,
        "source_lora_rank": 2,
        "u_shape": [3],
        "v_shape": [5],
        "selection_score": 3.5,
    }
    assert "u" not in metadata[0]
    assert "v" not in metadata[0]


def test_saves_csv_with_stable_columns_and_creates_parent(tmp_path) -> None:
    atom = _atom("selected", "layer", 1.25, 1, 0)
    output_path = tmp_path / "nested" / "selected_atoms.csv"

    save_selected_atoms_metadata({"layer": [atom]}, output_path)

    with output_path.open(newline="", encoding="utf-8") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)
        assert reader.fieldnames == [
            "round_idx",
            "atom_id",
            "layer_name",
            "client_id",
            "task_name",
            "rank_id",
            "sigma",
            "source_lora_rank",
            "u_shape",
            "v_shape",
            "selection_score",
        ]
    assert len(rows) == 1
    assert rows[0]["atom_id"] == "selected"
    assert rows[0]["u_shape"] == "[3]"
    assert "tensor" not in output_path.read_text(encoding="utf-8")


def test_empty_selection_writes_header_only_csv(tmp_path) -> None:
    output_path = tmp_path / "empty.csv"

    save_selected_atoms_metadata({}, output_path)

    with output_path.open(newline="", encoding="utf-8") as output_file:
        assert list(csv.DictReader(output_file)) == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_rejects_invalid_top_k_per_layer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_topk_atoms_by_sigma([], {"jury": {"top_k_per_layer": value}})


@pytest.mark.parametrize(
    "sigma",
    [torch.ones(2), "large", True],
)
def test_rejects_non_scalar_or_non_numeric_sigma(sigma: object) -> None:
    with pytest.raises(ValueError, match="sigma must be"):
        select_topk_atoms_by_sigma([_atom("bad", "layer", sigma, 1, 0)], _config())


def test_rejects_malformed_atoms_and_mismatched_metadata_groups() -> None:
    with pytest.raises(ValueError, match="Each atom must be a mapping"):
        select_topk_atoms_by_sigma([[]], _config())
    with pytest.raises(ValueError, match="missing required field.*atom_id"):
        select_topk_atoms_by_sigma([{}], _config())

    atom = _atom("wrong_layer", "actual", 1.0, 1, 0)
    with pytest.raises(ValueError, match="not group"):
        selected_atoms_to_metadata({"other": [atom]})
