"""Server-side vote counting and global Jury memory updates."""

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch
from torch import nn

from src.jury.memory import (
    apply_global_memory_to_model as _apply_global_memory_to_model,
    clone_memory_atom,
    scalar_as_float,
)


_REQUIRED_SELECTED_ATOM_FIELDS = (
    "atom_id",
    "round_idx",
    "layer_name",
    "u",
    "v",
    "sigma",
)

_REQUIRED_VOTE_FIELDS = (
    "client_id",
    "task_name",
    "atom_id",
    "layer_name",
    "score",
    "vote",
)

_VALID_VOTES = {"accept", "reject", "abstain"}


def _validate_selected_atoms(
    selected_atoms_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[Mapping[str, Any]], dict[Any, Mapping[str, Any]]]:
    if not isinstance(selected_atoms_by_layer, Mapping):
        raise ValueError("selected_atoms_by_layer must be a mapping by layer.")
    ordered_atoms: list[Mapping[str, Any]] = []
    atoms_by_id: dict[Any, Mapping[str, Any]] = {}
    for layer_name, layer_atoms in selected_atoms_by_layer.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("Selected atom layer names must be nonempty strings.")
        if isinstance(layer_atoms, (str, bytes)) or not isinstance(
            layer_atoms, Sequence
        ):
            raise ValueError(
                f"Selected atoms for layer {layer_name!r} must be a sequence."
            )
        for atom in layer_atoms:
            if not isinstance(atom, Mapping):
                raise ValueError("Each selected atom must be a mapping.")
            missing = [
                field for field in _REQUIRED_SELECTED_ATOM_FIELDS if field not in atom
            ]
            if missing:
                raise ValueError(
                    "Selected atom is missing required field(s): "
                    + ", ".join(missing)
                    + "."
                )
            if atom["layer_name"] != layer_name:
                raise ValueError(
                    f"Selected atom {atom['atom_id']!r} belongs to layer "
                    f"{atom['layer_name']!r}, not group {layer_name!r}."
                )
            atom_id = atom["atom_id"]
            if atom_id in atoms_by_id:
                raise ValueError(f"Duplicate selected atom_id {atom_id!r}.")
            for vector_name in ("u", "v"):
                vector = atom[vector_name]
                if not torch.is_tensor(vector) or vector.ndim != 1:
                    raise ValueError(
                        f"Selected atom {atom_id!r} {vector_name} must be a 1D tensor."
                    )
            scalar_as_float(atom["sigma"], f"Selected atom {atom_id!r} sigma")
            ordered_atoms.append(atom)
            atoms_by_id[atom_id] = atom
    return ordered_atoms, atoms_by_id


def count_votes_for_atoms(
    selected_atoms_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
    vote_records: Sequence[Mapping[str, Any]],
    num_clients: int,
) -> list[dict[str, Any]]:
    """Count one vote per client and treat missing votes as abstentions."""
    if (
        isinstance(num_clients, bool)
        or not isinstance(num_clients, int)
        or num_clients <= 0
    ):
        raise ValueError("num_clients must be a positive integer.")
    ordered_atoms, atoms_by_id = _validate_selected_atoms(selected_atoms_by_layer)
    if isinstance(vote_records, (str, bytes)) or not isinstance(vote_records, Sequence):
        raise ValueError("vote_records must be a sequence.")

    explicit_counts = {
        atom_id: {"accept": 0, "reject": 0, "abstain": 0}
        for atom_id in atoms_by_id
    }
    voters_by_atom: dict[Any, set[Any]] = {atom_id: set() for atom_id in atoms_by_id}
    all_voters: set[Any] = set()
    for record in vote_records:
        if not isinstance(record, Mapping):
            raise ValueError("Each vote record must be a mapping.")
        missing = [field for field in _REQUIRED_VOTE_FIELDS if field not in record]
        if missing:
            raise ValueError(
                "Vote record is missing required field(s): "
                + ", ".join(missing)
                + "."
            )
        atom_id = record["atom_id"]
        if atom_id not in atoms_by_id:
            raise ValueError(f"Vote record references unknown atom_id {atom_id!r}.")
        atom = atoms_by_id[atom_id]
        if record["layer_name"] != atom["layer_name"]:
            raise ValueError(
                f"Vote for atom {atom_id!r} uses layer {record['layer_name']!r}, "
                f"expected {atom['layer_name']!r}."
            )
        vote = record["vote"]
        if vote not in _VALID_VOTES:
            raise ValueError(
                f"Vote for atom {atom_id!r} must be accept, reject, or abstain."
            )
        client_id = record["client_id"]
        if client_id in voters_by_atom[atom_id]:
            raise ValueError(
                f"Client {client_id!r} has duplicate votes for atom {atom_id!r}."
            )
        voters_by_atom[atom_id].add(client_id)
        all_voters.add(client_id)
        if len(all_voters) > num_clients:
            raise ValueError("Vote records reference more clients than num_clients.")
        if len(voters_by_atom[atom_id]) > num_clients:
            raise ValueError(
                f"Atom {atom_id!r} has more unique voters than num_clients."
            )
        explicit_counts[atom_id][vote] += 1

    vote_stats: list[dict[str, Any]] = []
    for atom in ordered_atoms:
        atom_id = atom["atom_id"]
        counts = explicit_counts[atom_id]
        missing_votes = num_clients - len(voters_by_atom[atom_id])
        abstain_count = counts["abstain"] + missing_votes
        vote_stats.append(
            {
                "round_idx": atom["round_idx"],
                "atom_id": atom_id,
                "layer_name": atom["layer_name"],
                "num_clients": num_clients,
                "accept_count": counts["accept"],
                "reject_count": counts["reject"],
                "abstain_count": abstain_count,
                "p_accept": counts["accept"] / num_clients,
                "p_reject": counts["reject"] / num_clients,
                "p_abstain": abstain_count / num_clients,
            }
        )
    return vote_stats


def _finite_number(
    value: Any,
    config_name: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"config value {config_name} must be numeric.")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise ValueError(f"config value {config_name} is out of range.")
    if maximum is not None and converted > maximum:
        raise ValueError(f"config value {config_name} is out of range.")
    return converted


def _memory_config(config: Mapping[str, Any]) -> tuple[int, float, float, float, int]:
    try:
        num_clients = config["data"]["num_clients"]
        jury = config["jury"]
        theta_high = jury["theta_high"]
        eta_memory = jury["eta_memory"]
        alpha_memory = jury["alpha_memory"]
        budget = jury["global_memory_budget_per_layer"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Missing required global Jury memory configuration."
        ) from error
    if (
        isinstance(num_clients, bool)
        or not isinstance(num_clients, int)
        or num_clients <= 0
    ):
        raise ValueError("config['data']['num_clients'] must be a positive integer.")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError(
            "config['jury']['global_memory_budget_per_layer'] must be a positive "
            "integer."
        )
    return (
        num_clients,
        _finite_number(theta_high, "jury.theta_high", 0.0, 1.0),
        _finite_number(eta_memory, "jury.eta_memory", 0.0),
        _finite_number(alpha_memory, "jury.alpha_memory", 0.0, 1.0),
        budget,
    )


def update_global_memory(
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_atoms_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
    vote_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    round_idx: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Decay, extend, and budget global residual memory for one round."""
    if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
        raise ValueError("round_idx must be a nonnegative integer.")
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping by layer.")
    num_clients, theta_high, eta_memory, alpha_memory, budget = _memory_config(config)
    ordered_atoms, atoms_by_id = _validate_selected_atoms(selected_atoms_by_layer)
    vote_stats = count_votes_for_atoms(
        selected_atoms_by_layer,
        vote_records,
        num_clients,
    )

    updated_memory: dict[str, list[dict[str, Any]]] = {}
    for layer_name, layer_atoms in memory.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("Memory layer names must be nonempty strings.")
        if isinstance(layer_atoms, (str, bytes)) or not isinstance(
            layer_atoms, Sequence
        ):
            raise ValueError(
                f"Memory atoms for layer {layer_name!r} must be a sequence."
            )
        copied_atoms: list[dict[str, Any]] = []
        for atom in layer_atoms:
            copied = clone_memory_atom(atom, layer_name)
            copied["lambda_g"] *= alpha_memory
            margin = max(copied["p_accept"] - copied["p_reject"], 0.0)
            copied["importance"] = abs(copied["lambda_g"]) * margin
            copied_atoms.append(copied)
        updated_memory[layer_name] = copied_atoms

    stats_by_id = {stats["atom_id"]: stats for stats in vote_stats}
    for atom in ordered_atoms:
        stats = stats_by_id[atom["atom_id"]]
        if stats["p_accept"] < theta_high:
            continue
        sigma = scalar_as_float(atom["sigma"], f"Atom {atom['atom_id']!r} sigma")
        margin = stats["p_accept"] - stats["p_reject"]
        lambda_g = eta_memory * sigma * margin
        if lambda_g <= 0:
            continue
        layer_name = atom["layer_name"]
        updated_memory.setdefault(layer_name, []).append(
            {
                "memory_atom_id": f"memory_round_{round_idx}:{atom['atom_id']}",
                "source_atom_id": atom["atom_id"],
                "round_idx": round_idx,
                "layer_name": layer_name,
                "u": atom["u"].detach().cpu().clone(),
                "v": atom["v"].detach().cpu().clone(),
                "sigma": sigma,
                "lambda_g": lambda_g,
                "p_accept": stats["p_accept"],
                "p_reject": stats["p_reject"],
                "p_abstain": stats["p_abstain"],
                "accept_count": stats["accept_count"],
                "reject_count": stats["reject_count"],
                "abstain_count": stats["abstain_count"],
                "importance": abs(lambda_g) * max(margin, 0.0),
            }
        )

    for layer_name, layer_atoms in updated_memory.items():
        updated_memory[layer_name] = sorted(
            layer_atoms,
            key=lambda atom: -atom["importance"],
        )[:budget]
    return updated_memory, vote_stats


def apply_global_memory_to_model(
    model: nn.Module,
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Apply global memory once to a freshly loaded model in-place."""
    _apply_global_memory_to_model(model, memory, scale=1.0)
