"""Tests for residual LoRA atom extraction."""

import pytest
import torch

from src.jury.atom_extractor import (
    atoms_to_metadata,
    extract_atoms_from_client_update,
    extract_atoms_from_round,
)


def _config(svd_top_r: int = 2) -> dict:
    return {"jury": {"svd_top_r": svd_top_r}}


def _update(grouped_lora_ab: dict) -> dict:
    return {
        "round_idx": 3,
        "client_id": 7,
        "task_name": "sst2",
        "grouped_lora_ab": grouped_lora_ab,
    }


def test_extracts_cpu_atoms_and_reconstructs_delta() -> None:
    matrix_a = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
        requires_grad=True,
    )
    matrix_b = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 1.0]]
    )

    atoms = extract_atoms_from_client_update(
        _update({"encoder.query": {"A": matrix_a, "B": matrix_b}}),
        _config(10),
        device="cpu",
    )

    assert len(atoms) == 3
    assert [atom["rank_id"] for atom in atoms] == [0, 1, 2]
    assert atoms[0]["atom_id"] == "round_3:client_7:layer_encoder.query:rank_0"
    assert all(atom["u"].shape == (4,) for atom in atoms)
    assert all(atom["v"].shape == (3,) for atom in atoms)
    assert all(atom["source_lora_rank"] == 2 for atom in atoms)
    assert all(atom["u"].device.type == "cpu" for atom in atoms)
    assert all(atom["v"].device.type == "cpu" for atom in atoms)
    assert all(atom["sigma"].device.type == "cpu" for atom in atoms)
    assert all(not atom["u"].requires_grad for atom in atoms)
    assert all(not atom["v"].requires_grad for atom in atoms)

    reconstructed = sum(
        atom["sigma"] * torch.outer(atom["u"], atom["v"]) for atom in atoms
    )
    torch.testing.assert_close(reconstructed, matrix_b @ matrix_a)


def test_truncates_atoms_and_preserves_client_and_layer_order() -> None:
    identity = torch.eye(2)
    first = _update(
        {
            "layer_a": {"A": identity, "B": identity},
            "layer_b": {"A": identity, "B": identity},
        }
    )
    second = {
        **_update({"layer_c": {"A": identity, "B": identity}}),
        "client_id": 8,
    }

    atoms = extract_atoms_from_round([first, second], _config(1))

    assert [(atom["client_id"], atom["layer_name"]) for atom in atoms] == [
        (7, "layer_a"),
        (7, "layer_b"),
        (8, "layer_c"),
    ]
    assert extract_atoms_from_round([], _config(1)) == []


def test_atoms_to_metadata_omits_vectors_and_serializes_shapes() -> None:
    atoms = extract_atoms_from_client_update(
        _update({"layer": {"A": torch.eye(2), "B": torch.ones(3, 2)}}),
        _config(1),
    )

    metadata = atoms_to_metadata(atoms)

    assert metadata == [
        {
            "atom_id": "round_3:client_7:layer_layer:rank_0",
            "round_idx": 3,
            "client_id": 7,
            "task_name": "sst2",
            "layer_name": "layer",
            "rank_id": 0,
            "sigma": pytest.approx(atoms[0]["sigma"].item()),
            "source_lora_rank": 2,
            "u_shape": [3],
            "v_shape": [2],
        }
    ]
    assert "u" not in metadata[0]
    assert "v" not in metadata[0]


@pytest.mark.parametrize(
    ("grouped", "message"),
    [
        ({"layer": {"B": torch.eye(2)}}, "missing: A"),
        ({"layer": {"A": torch.eye(2)}}, "missing: B"),
        (
            {"layer": {"A": [[1.0]], "B": torch.ones(1, 1)}},
            "A.*must be a tensor",
        ),
        ({"layer": {"A": torch.ones(2), "B": torch.ones(1, 2)}}, "A.*must be 2D"),
        ({"layer": {"A": torch.ones(2, 1), "B": torch.ones(2)}}, "B.*must be 2D"),
        (
            {"layer": {"A": torch.ones(2, 3), "B": torch.ones(4, 1)}},
            "B.shape\\[1\\] must equal A.shape\\[0\\]",
        ),
    ],
)
def test_rejects_invalid_lora_pairs(grouped: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_atoms_from_client_update(_update(grouped), _config())


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_rejects_invalid_svd_top_r(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        extract_atoms_from_client_update(_update({}), {"jury": {"svd_top_r": value}})


def test_rejects_malformed_client_updates() -> None:
    with pytest.raises(ValueError, match="client_update must be a mapping"):
        extract_atoms_from_client_update([], _config())
    with pytest.raises(ValueError, match="missing required field.*round_idx"):
        extract_atoms_from_client_update({}, _config())
    with pytest.raises(ValueError, match="grouped_lora_ab must be a mapping"):
        extract_atoms_from_client_update(
            {
                "round_idx": 1,
                "client_id": 2,
                "task_name": "sst2",
                "grouped_lora_ab": [],
            },
            _config(),
        )
