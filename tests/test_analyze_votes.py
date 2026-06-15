"""Tests for Jury-LoRA vote-pattern analysis."""

import csv
import math

import numpy as np
import pandas as pd
import pytest

from src.eval.analyze_votes import (
    ATOM_METADATA_COLUMNS,
    CLIENT_METADATA_COLUMNS,
    RATIO_COLUMNS,
    SIMILARITY_COLUMNS,
    analyze_votes,
    build_vote_matrix,
    compute_similarity_summary,
    compute_task_vote_ratios,
    cosine_similarity_matrix,
    load_vote_records,
    parse_round,
)


def _vote(
    round_idx: int,
    client_id: int,
    task_name: str,
    atom_id: str,
    layer_name: str,
    vote: str,
) -> dict:
    return {
        "round_idx": round_idx,
        "client_id": client_id,
        "task_name": task_name,
        "atom_id": atom_id,
        "layer_name": layer_name,
        "score": 0.0,
        "vote": vote,
    }


def test_build_vote_matrix_sorts_and_fills_missing_votes() -> None:
    votes = pd.DataFrame(
        [
            _vote(1, 3, "task_b", "z", "layer_b", "reject"),
            _vote(1, 2, "task_a", "a", "layer_a", "accept"),
            _vote(1, 1, "task_a", "z", "layer_b", "abstain"),
        ]
    )
    selected = pd.DataFrame(
        [
            {"round_idx": 1, "atom_id": "m", "layer_name": "layer_a"},
            {"round_idx": 1, "atom_id": "a", "layer_name": "layer_a"},
        ]
    )

    matrix, clients, atoms = build_vote_matrix(votes, selected)

    assert clients.to_dict("records") == [
        {"row_idx": 0, "client_id": 1, "task_name": "task_a"},
        {"row_idx": 1, "client_id": 2, "task_name": "task_a"},
        {"row_idx": 2, "client_id": 3, "task_name": "task_b"},
    ]
    assert atoms.to_dict("records") == [
        {"col_idx": 0, "atom_id": "a", "layer_name": "layer_a"},
        {"col_idx": 1, "atom_id": "m", "layer_name": "layer_a"},
        {"col_idx": 2, "atom_id": "z", "layer_name": "layer_b"},
    ]
    np.testing.assert_array_equal(
        matrix,
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, -1]], dtype=np.int8),
    )


def test_cosine_similarity_and_pair_summary_handle_zero_vectors() -> None:
    matrix = np.array([[1, 0], [1, 0], [0, 0], [-1, 0]], dtype=np.int8)
    clients = pd.DataFrame(
        {
            "row_idx": range(4),
            "client_id": range(4),
            "task_name": ["a", "a", "b", "b"],
        }
    )

    similarity = cosine_similarity_matrix(matrix)
    summary = compute_similarity_summary(1, similarity, clients)

    np.testing.assert_allclose(
        similarity,
        np.array(
            [
                [1, 1, 0, -1],
                [1, 1, 0, -1],
                [0, 0, 0, 0],
                [-1, -1, 0, 1],
            ]
        ),
    )
    assert summary["same_task_mean"] == pytest.approx(0.5)
    assert summary["same_task_std"] == pytest.approx(0.5)
    assert summary["different_task_mean"] == pytest.approx(-0.5)
    assert summary["different_task_std"] == pytest.approx(0.5)
    assert summary["separation_gap"] == pytest.approx(1.0)


def test_similarity_summary_uses_nan_when_pair_category_is_absent() -> None:
    clients = pd.DataFrame(
        {"row_idx": [0, 1], "client_id": [1, 2], "task_name": ["a", "b"]}
    )

    summary = compute_similarity_summary(1, np.eye(2), clients)

    assert math.isnan(summary["same_task_mean"])
    assert math.isnan(summary["same_task_std"])
    assert summary["different_task_mean"] == 0.0
    assert math.isnan(summary["separation_gap"])


def test_task_ratios_include_dense_missing_votes_as_abstentions() -> None:
    matrix = np.array([[1, 0], [-1, 0], [1, 1]], dtype=np.int8)
    clients = pd.DataFrame(
        {
            "row_idx": [0, 1, 2],
            "client_id": [0, 1, 2],
            "task_name": ["a", "a", "b"],
        }
    )

    rows = compute_task_vote_ratios(2, matrix, clients)

    assert rows == [
        {
            "round_idx": 2,
            "task_name": "a",
            "accept_ratio": 0.25,
            "reject_ratio": 0.25,
            "abstain_ratio": 0.5,
            "num_votes": 4,
        },
        {
            "round_idx": 2,
            "task_name": "b",
            "accept_ratio": 1.0,
            "reject_ratio": 0.0,
            "abstain_ratio": 0.0,
            "num_votes": 2,
        },
    ]


def test_analyze_all_rounds_writes_requested_artifacts(tmp_path) -> None:
    votes = pd.DataFrame(
        [
            _vote(1, 1, "a", "x", "layer", "accept"),
            _vote(1, 2, "a", "x", "layer", "accept"),
            _vote(1, 3, "b", "x", "layer", "reject"),
            _vote(2, 1, "a", "y", "layer", "abstain"),
            _vote(2, 3, "b", "y", "layer", "accept"),
        ]
    )
    votes.to_csv(tmp_path / "vote_records.csv", index=False)
    pd.DataFrame(
        [
            {"round_idx": 1, "atom_id": "x", "layer_name": "layer"},
            {"round_idx": 2, "atom_id": "y", "layer_name": "layer"},
            {"round_idx": 2, "atom_id": "z", "layer_name": "layer"},
        ]
    ).to_csv(tmp_path / "selected_atoms.csv", index=False)

    analyze_votes(tmp_path, "all")

    assert np.load(tmp_path / "vote_matrix_round_2.npy").shape == (2, 2)
    for round_idx in (1, 2):
        assert (tmp_path / f"vote_matrix_round_{round_idx}.npy").is_file()
        assert (tmp_path / f"vote_agreement_heatmap_round_{round_idx}.png").stat().st_size > 0
        with (tmp_path / f"vote_matrix_clients_round_{round_idx}.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            assert csv.DictReader(file).fieldnames == list(CLIENT_METADATA_COLUMNS)
        with (tmp_path / f"vote_matrix_atoms_round_{round_idx}.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            assert csv.DictReader(file).fieldnames == list(ATOM_METADATA_COLUMNS)
    assert list(pd.read_csv(tmp_path / "vote_similarity.csv").columns) == list(
        SIMILARITY_COLUMNS
    )
    assert list(pd.read_csv(tmp_path / "vote_ratios_by_task.csv").columns) == list(
        RATIO_COLUMNS
    )
    assert pd.read_csv(tmp_path / "vote_similarity.csv")["round_idx"].tolist() == [1, 2]


def test_specific_round_overwrites_aggregate_files_with_only_that_round(tmp_path) -> None:
    pd.DataFrame(
        [
            _vote(1, 1, "a", "x", "layer", "accept"),
            _vote(2, 1, "a", "y", "layer", "reject"),
        ]
    ).to_csv(tmp_path / "vote_records.csv", index=False)

    analyze_votes(tmp_path, 2)

    assert pd.read_csv(tmp_path / "vote_similarity.csv")["round_idx"].tolist() == [2]
    assert pd.read_csv(tmp_path / "vote_ratios_by_task.csv")["round_idx"].tolist() == [2]
    assert not (tmp_path / "vote_matrix_round_1.npy").exists()
    assert (tmp_path / "vote_matrix_round_2.npy").exists()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"round_idx": 1}], "missing required"),
        (
            [
                _vote(1, 1, "a", "x", "layer", "accept"),
                _vote(1, 1, "a", "x", "layer", "reject"),
            ],
            "duplicate",
        ),
        ([_vote(1, 1, "a", "x", "layer", "maybe")], "invalid votes"),
        (
            [
                _vote(1, 1, "a", "x", "layer", "accept"),
                _vote(1, 1, "b", "y", "layer", "accept"),
            ],
            "multiple task_name",
        ),
    ],
)
def test_vote_record_validation(tmp_path, rows, message) -> None:
    pd.DataFrame(rows).to_csv(tmp_path / "vote_records.csv", index=False)
    with pytest.raises(ValueError, match=message):
        load_vote_records(tmp_path / "vote_records.csv")


def test_round_parser_and_unavailable_round(tmp_path) -> None:
    assert parse_round("all") == "all"
    assert parse_round("3") == 3
    with pytest.raises(Exception):
        parse_round("0")

    pd.DataFrame([_vote(1, 1, "a", "x", "layer", "accept")]).to_csv(
        tmp_path / "vote_records.csv", index=False
    )
    with pytest.raises(ValueError, match="unavailable"):
        analyze_votes(tmp_path, 2)
