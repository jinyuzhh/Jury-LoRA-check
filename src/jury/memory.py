"""Global residual-memory storage, persistence, and model application."""

import csv
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn


_MEMORY_FIELDS = (
    "round_idx",
    "memory_atom_id",
    "source_atom_id",
    "layer_name",
    "sigma",
    "lambda_g",
    "p_accept",
    "p_reject",
    "p_abstain",
    "importance",
    "u_shape",
    "v_shape",
)

_VOTE_STATS_FIELDS = (
    "round_idx",
    "atom_id",
    "layer_name",
    "num_clients",
    "accept_count",
    "reject_count",
    "abstain_count",
    "p_accept",
    "p_reject",
    "p_abstain",
)

_REQUIRED_MEMORY_ATOM_FIELDS = (
    "memory_atom_id",
    "source_atom_id",
    "round_idx",
    "layer_name",
    "u",
    "v",
    "sigma",
    "lambda_g",
    "p_accept",
    "p_reject",
    "p_abstain",
    "accept_count",
    "reject_count",
    "abstain_count",
    "importance",
)


def scalar_as_float(value: Any, field_name: str) -> float:
    """Convert a finite scalar tensor or real number to float."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"{field_name} must be scalar, but has shape {tuple(value.shape)}."
            )
        converted = float(value.detach().cpu().item())
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a numeric scalar.")
    else:
        converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


def validate_memory_atom(
    atom: Any,
    expected_layer: str | None = None,
) -> Mapping[str, Any]:
    """Validate one memory atom and return its mapping."""
    if not isinstance(atom, Mapping):
        raise ValueError("Each memory atom must be a mapping.")
    missing = [field for field in _REQUIRED_MEMORY_ATOM_FIELDS if field not in atom]
    if missing:
        raise ValueError(
            "Memory atom is missing required field(s): " + ", ".join(missing) + "."
        )
    layer_name = atom["layer_name"]
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError("Memory atom layer_name must be a nonempty string.")
    if expected_layer is not None and layer_name != expected_layer:
        raise ValueError(
            f"Memory atom {atom['memory_atom_id']!r} belongs to layer "
            f"{layer_name!r}, not group {expected_layer!r}."
        )
    for vector_name in ("u", "v"):
        vector = atom[vector_name]
        if not torch.is_tensor(vector):
            raise ValueError(
                f"Memory atom {atom['memory_atom_id']!r} {vector_name} must be "
                "a tensor."
            )
        if vector.ndim != 1:
            raise ValueError(
                f"Memory atom {atom['memory_atom_id']!r} {vector_name} must be 1D, "
                f"but has shape {tuple(vector.shape)}."
            )
    for field in (
        "sigma",
        "lambda_g",
        "p_accept",
        "p_reject",
        "p_abstain",
        "importance",
    ):
        scalar_as_float(atom[field], f"Memory atom {atom['memory_atom_id']!r} {field}")
    return atom


def clone_memory_atom(atom: Mapping[str, Any], layer_name: str) -> dict[str, Any]:
    """Return an independent CPU copy of a validated memory atom."""
    validated = validate_memory_atom(atom, layer_name)
    copied = dict(validated)
    copied["u"] = validated["u"].detach().cpu().clone()
    copied["v"] = validated["v"].detach().cpu().clone()
    for field in (
        "sigma",
        "lambda_g",
        "p_accept",
        "p_reject",
        "p_abstain",
        "importance",
    ):
        copied[field] = scalar_as_float(
            validated[field],
            f"Memory atom {validated['memory_atom_id']!r} {field}",
        )
    return copied


def _find_module(model: nn.Module, layer_name: str) -> nn.Module:
    for module_name, module in model.named_modules():
        if module_name == layer_name:
            return module
    raise ValueError(f"Model does not contain memory target layer {layer_name!r}.")


def _target_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight):
        base_layer = getattr(module, "base_layer", None)
        weight = getattr(base_layer, "weight", None)
    if not torch.is_tensor(weight):
        raise ValueError(
            f"Target module {module.__class__.__name__} does not expose a weight "
            "tensor or base_layer.weight tensor."
        )
    if weight.ndim != 2:
        raise ValueError(
            f"Target layer weight must be 2D, but has shape {tuple(weight.shape)}."
        )
    return weight


def apply_global_memory_to_model(
    model: nn.Module,
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
    scale: float = 1.0,
) -> None:
    """Apply residual memory to model weights in-place.

    Updates are cumulative. Call this on a freshly loaded base/global model, or
    explicitly reload that model before applying the same memory again.
    """
    scale_value = scalar_as_float(scale, "scale")
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping by layer.")

    prepared_updates: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_name, layer_atoms in memory.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("Memory layer names must be nonempty strings.")
        if isinstance(layer_atoms, (str, bytes)) or not isinstance(
            layer_atoms, Sequence
        ):
            raise ValueError(
                f"Memory atoms for layer {layer_name!r} must be a sequence."
            )
        if not layer_atoms:
            continue
        module = _find_module(model, layer_name)
        weight = _target_weight(module)
        layer_update = torch.zeros_like(weight)
        for atom_value in layer_atoms:
            atom = validate_memory_atom(atom_value, layer_name)
            vector_u = atom["u"].detach().to(weight.device, dtype=weight.dtype)
            vector_v = atom["v"].detach().to(weight.device, dtype=weight.dtype)
            residual = torch.outer(vector_u, vector_v)
            if residual.shape != weight.shape:
                raise ValueError(
                    f"Memory atom {atom['memory_atom_id']!r} outer(u, v) has shape "
                    f"{tuple(residual.shape)}, but target layer weight has shape "
                    f"{tuple(weight.shape)}."
                )
            lambda_g = scalar_as_float(
                atom["lambda_g"],
                f"Memory atom {atom['memory_atom_id']!r} lambda_g",
            )
            layer_update.add_(residual, alpha=scale_value * lambda_g)
        prepared_updates.append((weight, layer_update))

    with torch.no_grad():
        for weight, update in prepared_updates:
            weight.add_(update)


def memory_to_metadata(
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten memory into tensor-free metadata rows."""
    metadata: list[dict[str, Any]] = []
    for layer_name, layer_atoms in memory.items():
        for atom_value in layer_atoms:
            atom = validate_memory_atom(atom_value, layer_name)
            metadata.append(
                {
                    "round_idx": atom["round_idx"],
                    "memory_atom_id": atom["memory_atom_id"],
                    "source_atom_id": atom["source_atom_id"],
                    "layer_name": atom["layer_name"],
                    "sigma": scalar_as_float(atom["sigma"], "sigma"),
                    "lambda_g": scalar_as_float(atom["lambda_g"], "lambda_g"),
                    "p_accept": scalar_as_float(atom["p_accept"], "p_accept"),
                    "p_reject": scalar_as_float(atom["p_reject"], "p_reject"),
                    "p_abstain": scalar_as_float(atom["p_abstain"], "p_abstain"),
                    "importance": scalar_as_float(atom["importance"], "importance"),
                    "u_shape": list(atom["u"].shape),
                    "v_shape": list(atom["v"].shape),
                }
            )
    return metadata


def _save_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    fields: Sequence[str],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_memory_metadata(
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
) -> None:
    """Write tensor-free global-memory metadata to CSV."""
    _save_rows(memory_to_metadata(memory), output_path, _MEMORY_FIELDS)


def vote_stats_to_metadata(
    vote_stats: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable, CSV-friendly vote-stat rows."""
    metadata: list[dict[str, Any]] = []
    for stats in vote_stats:
        missing = [field for field in _VOTE_STATS_FIELDS if field not in stats]
        if missing:
            raise ValueError(
                "Vote stats are missing required field(s): "
                + ", ".join(missing)
                + "."
            )
        metadata.append(
            {
                "round_idx": stats["round_idx"],
                "atom_id": stats["atom_id"],
                "layer_name": stats["layer_name"],
                "num_clients": stats["num_clients"],
                "accept_count": stats["accept_count"],
                "reject_count": stats["reject_count"],
                "abstain_count": stats["abstain_count"],
                "p_accept": scalar_as_float(stats["p_accept"], "p_accept"),
                "p_reject": scalar_as_float(stats["p_reject"], "p_reject"),
                "p_abstain": scalar_as_float(stats["p_abstain"], "p_abstain"),
            }
        )
    return metadata


def save_vote_stats(
    vote_stats: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    """Write per-atom vote statistics to CSV."""
    _save_rows(vote_stats_to_metadata(vote_stats), output_path, _VOTE_STATS_FIELDS)
