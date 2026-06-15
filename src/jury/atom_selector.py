"""Select residual LoRA atoms by singular-value magnitude."""

import csv
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import torch


_REQUIRED_ATOM_FIELDS = (
    "atom_id",
    "round_idx",
    "client_id",
    "task_name",
    "layer_name",
    "rank_id",
    "u",
    "v",
    "sigma",
    "source_lora_rank",
)

_METADATA_FIELDS = (
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
)


def _get_top_k_per_layer(config: Mapping[str, Any]) -> int:
    """Return the configured per-layer atom limit."""
    try:
        top_k_per_layer = config["jury"]["top_k_per_layer"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "config must contain jury.top_k_per_layer as a positive integer."
        ) from error

    if isinstance(top_k_per_layer, bool) or not isinstance(top_k_per_layer, int):
        raise ValueError(
            "config['jury']['top_k_per_layer'] must be a positive integer."
        )
    if top_k_per_layer <= 0:
        raise ValueError(
            "config['jury']['top_k_per_layer'] must be a positive integer."
        )
    return top_k_per_layer


def _sigma_as_float(sigma: Any, atom_id: Any) -> float:
    """Convert a scalar atom score to a Python float."""
    if torch.is_tensor(sigma):
        if sigma.numel() != 1:
            raise ValueError(
                f"Atom {atom_id!r} sigma must be scalar, but has shape "
                f"{tuple(sigma.shape)}."
            )
        return float(sigma.detach().cpu().item())
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise ValueError(f"Atom {atom_id!r} sigma must be a numeric scalar.")
    return float(sigma)


def _validate_atom(atom: Any) -> Mapping[str, Any]:
    if not isinstance(atom, Mapping):
        raise ValueError("Each atom must be a mapping.")

    missing_fields = [field for field in _REQUIRED_ATOM_FIELDS if field not in atom]
    if missing_fields:
        raise ValueError(
            "Atom is missing required field(s): "
            + ", ".join(missing_fields)
            + "."
        )
    if not isinstance(atom["layer_name"], str):
        raise ValueError(f"Atom {atom['atom_id']!r} layer_name must be a string.")
    for field in ("client_id", "rank_id"):
        value = atom[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Atom {atom['atom_id']!r} {field} must be an integer.")
    if not torch.is_tensor(atom["u"]):
        raise ValueError(f"Atom {atom['atom_id']!r} u must be a tensor.")
    if not torch.is_tensor(atom["v"]):
        raise ValueError(f"Atom {atom['atom_id']!r} v must be a tensor.")
    _sigma_as_float(atom["sigma"], atom["atom_id"])
    return atom


def select_topk_atoms_by_sigma(
    atoms: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group atoms by layer and retain the highest-sigma atoms in each layer."""
    top_k_per_layer = _get_top_k_per_layer(config)
    atoms_by_layer: dict[str, list[Mapping[str, Any]]] = {}

    for atom_value in atoms:
        atom = _validate_atom(atom_value)
        atoms_by_layer.setdefault(atom["layer_name"], []).append(atom)

    selected_atoms: dict[str, list[Mapping[str, Any]]] = {}
    for layer_name, layer_atoms in atoms_by_layer.items():
        ranked_atoms = sorted(
            layer_atoms,
            key=lambda atom: (
                -_sigma_as_float(atom["sigma"], atom["atom_id"]),
                atom["client_id"],
                atom["rank_id"],
            ),
        )
        selected_atoms[layer_name] = ranked_atoms[:top_k_per_layer]

    return selected_atoms


def selected_atoms_to_metadata(
    selected_atoms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten selected atoms into tensor-free metadata rows."""
    metadata: list[dict[str, Any]] = []
    for layer_name, layer_atoms in selected_atoms.items():
        for atom_value in layer_atoms:
            atom = _validate_atom(atom_value)
            if atom["layer_name"] != layer_name:
                raise ValueError(
                    f"Selected atom {atom['atom_id']!r} belongs to layer "
                    f"{atom['layer_name']!r}, not group {layer_name!r}."
                )
            sigma = _sigma_as_float(atom["sigma"], atom["atom_id"])
            metadata.append(
                {
                    "round_idx": atom["round_idx"],
                    "atom_id": atom["atom_id"],
                    "layer_name": atom["layer_name"],
                    "client_id": atom["client_id"],
                    "task_name": atom["task_name"],
                    "rank_id": atom["rank_id"],
                    "sigma": sigma,
                    "source_lora_rank": atom["source_lora_rank"],
                    "u_shape": list(atom["u"].shape),
                    "v_shape": list(atom["v"].shape),
                    "selection_score": sigma,
                }
            )
    return metadata


def save_selected_atoms_metadata(
    selected_atoms: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
) -> None:
    """Write selected atom metadata to CSV without serializing atom tensors."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = selected_atoms_to_metadata(selected_atoms)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
