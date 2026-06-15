"""Analyze whether Jury-LoRA voting patterns cluster by client task."""

import argparse
import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VOTE_RECORD_COLUMNS = (
    "round_idx",
    "client_id",
    "task_name",
    "atom_id",
    "layer_name",
    "score",
    "vote",
)
SELECTED_ATOM_COLUMNS = ("round_idx", "atom_id", "layer_name")
CLIENT_METADATA_COLUMNS = ("row_idx", "client_id", "task_name")
ATOM_METADATA_COLUMNS = ("col_idx", "atom_id", "layer_name")
SIMILARITY_COLUMNS = (
    "round_idx",
    "same_task_mean",
    "same_task_std",
    "different_task_mean",
    "different_task_std",
    "separation_gap",
)
RATIO_COLUMNS = (
    "round_idx",
    "task_name",
    "accept_ratio",
    "reject_ratio",
    "abstain_ratio",
    "num_votes",
)

_VOTE_VALUES = {"accept": 1, "reject": -1, "abstain": 0}


def parse_round(value: str) -> int | str:
    """Parse a positive round index or the literal ``all``."""
    if value == "all":
        return value
    try:
        round_idx = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--round must be a positive integer or 'all'."
        ) from error
    if round_idx <= 0:
        raise argparse.ArgumentTypeError(
            "--round must be a positive integer or 'all'."
        )
    return round_idx


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse vote-analysis command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze task-wise similarity of Jury-LoRA client votes."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing vote_records.csv.",
    )
    parser.add_argument(
        "--round",
        type=parse_round,
        default="all",
        help="Positive round index or 'all'.",
    )
    return parser.parse_args(argv)


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    filename: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing required column(s): {', '.join(missing)}."
        )


def _integer_column(
    frame: pd.DataFrame,
    column: str,
    filename: str,
) -> pd.Series:
    converted = pd.to_numeric(frame[column], errors="coerce")
    if converted.isna().any() or not np.all(converted == np.floor(converted)):
        raise ValueError(f"{filename} column {column!r} must contain integers.")
    return converted.astype(np.int64)


def load_vote_records(path: str | Path) -> pd.DataFrame:
    """Load and validate vote records from CSV."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Vote record file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    _require_columns(frame, VOTE_RECORD_COLUMNS, csv_path.name)
    frame = frame.loc[:, VOTE_RECORD_COLUMNS].copy()
    frame["round_idx"] = _integer_column(frame, "round_idx", csv_path.name)
    frame["client_id"] = _integer_column(frame, "client_id", csv_path.name)
    if (frame["round_idx"] <= 0).any():
        raise ValueError("vote_records.csv round_idx values must be positive.")
    for column in ("task_name", "atom_id", "layer_name", "vote"):
        if frame[column].isna().any() or (
            frame[column].astype(str).str.len() == 0
        ).any():
            raise ValueError(f"vote_records.csv column {column!r} cannot be empty.")
        frame[column] = frame[column].astype(str)
    invalid_votes = sorted(set(frame["vote"]) - set(_VOTE_VALUES))
    if invalid_votes:
        raise ValueError(
            "vote_records.csv contains invalid votes: "
            + ", ".join(invalid_votes)
        )
    duplicate_mask = frame.duplicated(
        subset=["round_idx", "client_id", "layer_name", "atom_id"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError("vote_records.csv contains duplicate client/atom votes.")
    task_counts = frame.groupby(["round_idx", "client_id"])["task_name"].nunique()
    if (task_counts > 1).any():
        raise ValueError(
            "A client_id maps to multiple task_name values within a round."
        )
    return frame


def load_selected_atoms(path: str | Path) -> pd.DataFrame:
    """Load optional selected-atom metadata."""
    csv_path = Path(path)
    if not csv_path.is_file():
        return pd.DataFrame(columns=SELECTED_ATOM_COLUMNS)
    frame = pd.read_csv(csv_path)
    _require_columns(frame, SELECTED_ATOM_COLUMNS, csv_path.name)
    frame = frame.loc[:, SELECTED_ATOM_COLUMNS].copy()
    frame["round_idx"] = _integer_column(frame, "round_idx", csv_path.name)
    if (frame["round_idx"] <= 0).any():
        raise ValueError("selected_atoms.csv round_idx values must be positive.")
    for column in ("atom_id", "layer_name"):
        if frame[column].isna().any() or (
            frame[column].astype(str).str.len() == 0
        ).any():
            raise ValueError(f"selected_atoms.csv column {column!r} cannot be empty.")
        frame[column] = frame[column].astype(str)
    duplicates = frame.duplicated(
        subset=["round_idx", "layer_name", "atom_id"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError("selected_atoms.csv contains duplicate atoms within a round.")
    return frame


def build_vote_matrix(
    round_votes: pd.DataFrame,
    round_selected_atoms: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build one sorted dense client-by-atom vote matrix."""
    if round_votes.empty:
        raise ValueError("Cannot build a vote matrix for a round with no vote records.")
    client_metadata = (
        round_votes.loc[:, ["client_id", "task_name"]]
        .drop_duplicates()
        .sort_values(["task_name", "client_id"], kind="stable")
        .reset_index(drop=True)
    )
    client_metadata.insert(0, "row_idx", np.arange(len(client_metadata)))

    atom_frames = [round_votes.loc[:, ["atom_id", "layer_name"]]]
    if round_selected_atoms is not None and not round_selected_atoms.empty:
        atom_frames.append(round_selected_atoms.loc[:, ["atom_id", "layer_name"]])
    atom_metadata = (
        pd.concat(atom_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["layer_name", "atom_id"], kind="stable")
        .reset_index(drop=True)
    )
    atom_metadata.insert(0, "col_idx", np.arange(len(atom_metadata)))

    row_lookup = {
        client_id: row_idx
        for row_idx, client_id in enumerate(client_metadata["client_id"])
    }
    col_lookup = {
        (layer_name, atom_id): col_idx
        for col_idx, (layer_name, atom_id) in enumerate(
            zip(atom_metadata["layer_name"], atom_metadata["atom_id"])
        )
    }
    matrix = np.zeros((len(client_metadata), len(atom_metadata)), dtype=np.int8)
    for row in round_votes.itertuples(index=False):
        matrix[
            row_lookup[row.client_id],
            col_lookup[(row.layer_name, row.atom_id)],
        ] = _VOTE_VALUES[row.vote]
    return matrix, client_metadata, atom_metadata


def cosine_similarity_matrix(vote_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity with safe zero-vector handling."""
    matrix = np.asarray(vote_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vote_matrix must be two-dimensional.")
    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.zeros_like(matrix, dtype=np.float64)
    nonzero = norms > 0
    normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
    # Avoid a BLAS call here: some mixed Torch/NumPy Windows environments load
    # incompatible OpenMP runtimes in the same process.
    similarity = np.sum(
        normalized[:, None, :] * normalized[None, :, :],
        axis=2,
    )
    similarity[np.abs(similarity) < 1e-15] = 0.0
    return np.clip(similarity, -1.0, 1.0)


def compute_similarity_summary(
    round_idx: int,
    similarity: np.ndarray,
    client_metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate unique client-pair similarities by task agreement."""
    same_task: list[float] = []
    different_task: list[float] = []
    tasks = client_metadata["task_name"].tolist()
    for first in range(len(tasks)):
        for second in range(first + 1, len(tasks)):
            target = same_task if tasks[first] == tasks[second] else different_task
            target.append(float(similarity[first, second]))

    def mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(array.std(ddof=0))

    same_mean, same_std = mean_std(same_task)
    different_mean, different_std = mean_std(different_task)
    gap = same_mean - different_mean
    return {
        "round_idx": round_idx,
        "same_task_mean": same_mean,
        "same_task_std": same_std,
        "different_task_mean": different_mean,
        "different_task_std": different_std,
        "separation_gap": gap,
    }


def compute_task_vote_ratios(
    round_idx: int,
    vote_matrix: np.ndarray,
    client_metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compute task ratios over the completed dense vote matrix."""
    rows: list[dict[str, Any]] = []
    for task_name in sorted(client_metadata["task_name"].unique()):
        task_indices = client_metadata.index[
            client_metadata["task_name"] == task_name
        ].to_numpy()
        task_votes = vote_matrix[task_indices, :]
        num_votes = int(task_votes.size)
        rows.append(
            {
                "round_idx": round_idx,
                "task_name": task_name,
                "accept_ratio": float(np.count_nonzero(task_votes == 1) / num_votes),
                "reject_ratio": float(np.count_nonzero(task_votes == -1) / num_votes),
                "abstain_ratio": float(np.count_nonzero(task_votes == 0) / num_votes),
                "num_votes": num_votes,
            }
        )
    return rows


def _render_heatmap_worker(
    similarity: np.ndarray,
    labels: list[str],
    output_path: str,
    round_idx: int,
) -> None:
    """Render one heatmap in a process isolated from Torch runtimes."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    size = max(6.0, min(16.0, 0.45 * len(labels) + 3.0))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_title(f"Client Vote Agreement - Round {round_idx}")
    axis.set_xlabel("Client (task:id)")
    axis.set_ylabel("Client (task:id)")
    positions = np.arange(len(labels))
    axis.set_xticks(positions, labels=labels, rotation=90)
    axis.set_yticks(positions, labels=labels)
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_heatmap(
    similarity: np.ndarray,
    client_metadata: pd.DataFrame,
    output_path: str | Path,
    round_idx: int,
) -> None:
    """Save a matplotlib heatmap of client vote agreement."""
    labels = [
        f"{task}:{client_id}"
        for task, client_id in zip(
            client_metadata["task_name"],
            client_metadata["client_id"],
        )
    ]
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_render_heatmap_worker,
        args=(similarity, labels, str(output_path), round_idx),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"Heatmap rendering failed for round {round_idx} with exit code "
            f"{process.exitcode}."
        )


def analyze_votes(output_dir: str | Path, round_selection: int | str = "all") -> None:
    """Analyze vote records and write all requested artifacts."""
    directory = Path(output_dir)
    vote_records = load_vote_records(directory / "vote_records.csv")
    selected_atoms = load_selected_atoms(directory / "selected_atoms.csv")
    available_rounds = sorted(vote_records["round_idx"].unique().tolist())
    if round_selection == "all":
        rounds = available_rounds
    else:
        if round_selection not in available_rounds:
            raise ValueError(
                f"Round {round_selection} is unavailable; available rounds: "
                + ", ".join(str(value) for value in available_rounds)
                + "."
            )
        rounds = [round_selection]
    if not rounds:
        raise ValueError("vote_records.csv does not contain any rounds to analyze.")

    similarity_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for round_idx in rounds:
        round_votes = vote_records[vote_records["round_idx"] == round_idx]
        round_selected = selected_atoms[selected_atoms["round_idx"] == round_idx]
        matrix, clients, atoms = build_vote_matrix(round_votes, round_selected)
        similarity = cosine_similarity_matrix(matrix)
        summary = compute_similarity_summary(round_idx, similarity, clients)
        task_ratios = compute_task_vote_ratios(round_idx, matrix, clients)
        similarity_rows.append(summary)
        ratio_rows.extend(task_ratios)

        np.save(directory / f"vote_matrix_round_{round_idx}.npy", matrix)
        clients.to_csv(
            directory / f"vote_matrix_clients_round_{round_idx}.csv",
            index=False,
            columns=CLIENT_METADATA_COLUMNS,
        )
        atoms.to_csv(
            directory / f"vote_matrix_atoms_round_{round_idx}.csv",
            index=False,
            columns=ATOM_METADATA_COLUMNS,
        )
        save_heatmap(
            similarity,
            clients,
            directory / f"vote_agreement_heatmap_round_{round_idx}.png",
            round_idx,
        )
        print(
            f"Round {round_idx}: clients={len(clients)}, atoms={len(atoms)}, "
            f"same_task_mean={summary['same_task_mean']:.6g}, "
            f"different_task_mean={summary['different_task_mean']:.6g}, "
            f"separation_gap={summary['separation_gap']:.6g}"
        )

    pd.DataFrame(similarity_rows, columns=SIMILARITY_COLUMNS).to_csv(
        directory / "vote_similarity.csv",
        index=False,
    )
    pd.DataFrame(ratio_rows, columns=RATIO_COLUMNS).to_csv(
        directory / "vote_ratios_by_task.csv",
        index=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run vote analysis from the command line."""
    args = parse_args(argv)
    analyze_votes(args.output_dir, args.round)


if __name__ == "__main__":
    main()
