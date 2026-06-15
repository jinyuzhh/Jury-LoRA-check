"""Extract residual rank-one atoms from grouped client LoRA updates."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch


_REQUIRED_UPDATE_FIELDS = (
    "round_idx",
    "client_id",
    "task_name",
    "grouped_lora_ab",
)


def _get_svd_top_r(config: Mapping[str, Any]) -> int:
    """Return the configured number of singular components to extract."""
    try:
        svd_top_r = config["jury"]["svd_top_r"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "config must contain jury.svd_top_r as a positive integer."
        ) from error

    if isinstance(svd_top_r, bool) or not isinstance(svd_top_r, int):
        raise ValueError("config['jury']['svd_top_r'] must be a positive integer.")
    if svd_top_r <= 0:
        raise ValueError("config['jury']['svd_top_r'] must be a positive integer.")
    return svd_top_r


def _validate_client_update(client_update: Mapping[str, Any]) -> None:
    missing_fields = [
        field for field in _REQUIRED_UPDATE_FIELDS if field not in client_update
    ]
    if missing_fields:
        raise ValueError(
            "Client update is missing required field(s): "
            + ", ".join(missing_fields)
            + "."
        )

    if not isinstance(client_update["grouped_lora_ab"], Mapping):
        raise ValueError("Client update grouped_lora_ab must be a mapping by layer.")


def _validate_lora_pair(
    layer_name: str,
    matrices: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(matrices, Mapping):
        raise ValueError(
            f"LoRA matrices for layer {layer_name!r} must be a mapping containing "
            "'A' and 'B'."
        )
    if "A" not in matrices or "B" not in matrices:
        missing = [name for name in ("A", "B") if name not in matrices]
        raise ValueError(
            f"LoRA matrices for layer {layer_name!r} are missing: "
            + ", ".join(missing)
            + "."
        )

    matrix_a = matrices["A"]
    matrix_b = matrices["B"]
    if not torch.is_tensor(matrix_a):
        raise ValueError(f"LoRA A for layer {layer_name!r} must be a tensor.")
    if not torch.is_tensor(matrix_b):
        raise ValueError(f"LoRA B for layer {layer_name!r} must be a tensor.")
    if matrix_a.ndim != 2:
        raise ValueError(
            f"LoRA A for layer {layer_name!r} must be 2D, but has shape "
            f"{tuple(matrix_a.shape)}."
        )
    if matrix_b.ndim != 2:
        raise ValueError(
            f"LoRA B for layer {layer_name!r} must be 2D, but has shape "
            f"{tuple(matrix_b.shape)}."
        )
    if matrix_b.shape[1] != matrix_a.shape[0]:
        raise ValueError(
            f"LoRA rank mismatch for layer {layer_name!r}: A has shape "
            f"{tuple(matrix_a.shape)} and B has shape {tuple(matrix_b.shape)}; "
            "B.shape[1] must equal A.shape[0]."
        )
    return matrix_a, matrix_b


def extract_atoms_from_client_update(
    client_update: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    """Extract compact-SVD atoms from one client's grouped LoRA update."""
    if not isinstance(client_update, Mapping):
        raise ValueError("client_update must be a mapping.")
    _validate_client_update(client_update)
    svd_top_r = _get_svd_top_r(config)
    configured_device = torch.device(device) if device is not None else None

    round_idx = client_update["round_idx"]
    client_id = client_update["client_id"]
    task_name = client_update["task_name"]
    grouped_lora_ab = client_update["grouped_lora_ab"]
    atoms: list[dict[str, Any]] = []

    with torch.no_grad():
        for layer_name, matrices in grouped_lora_ab.items():
            if not isinstance(layer_name, str):
                raise ValueError("Each grouped_lora_ab layer name must be a string.")
            matrix_a, matrix_b = _validate_lora_pair(layer_name, matrices)
            svd_device = configured_device or matrix_a.device
            matrix_a_for_svd = matrix_a.to(svd_device)
            matrix_b_for_svd = matrix_b.to(svd_device)

            delta = matrix_b_for_svd @ matrix_a_for_svd
            u_matrix, singular_values, vh_matrix = torch.linalg.svd(
                delta,
                full_matrices=False,
            )
            atom_count = min(svd_top_r, singular_values.numel())
            source_lora_rank = matrix_a.shape[0]

            for rank_id in range(atom_count):
                atoms.append(
                    {
                        "atom_id": (
                            f"round_{round_idx}:client_{client_id}:"
                            f"layer_{layer_name}:rank_{rank_id}"
                        ),
                        "round_idx": round_idx,
                        "client_id": client_id,
                        "task_name": task_name,
                        "layer_name": layer_name,
                        "rank_id": rank_id,
                        "u": u_matrix[:, rank_id].detach().cpu().clone(),
                        "v": vh_matrix[rank_id, :].detach().cpu().clone(),
                        "sigma": singular_values[rank_id].detach().cpu().clone(),
                        "source_lora_rank": source_lora_rank,
                    }
                )

    return atoms


def extract_atoms_from_round(
    client_updates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    """Extract atoms from all client updates, preserving their input order."""
    atoms: list[dict[str, Any]] = []
    for client_update in client_updates:
        atoms.extend(extract_atoms_from_client_update(client_update, config, device))
    return atoms


def atoms_to_metadata(atoms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-friendly atom metadata without singular-vector tensors."""
    metadata: list[dict[str, Any]] = []
    for atom in atoms:
        sigma = atom["sigma"]
        sigma_value = (
            sigma.detach().cpu().item()
            if torch.is_tensor(sigma)
            else float(sigma)
        )
        metadata.append(
            {
                "atom_id": atom["atom_id"],
                "round_idx": atom["round_idx"],
                "client_id": atom["client_id"],
                "task_name": atom["task_name"],
                "layer_name": atom["layer_name"],
                "rank_id": atom["rank_id"],
                "sigma": sigma_value,
                "source_lora_rank": atom["source_lora_rank"],
                "u_shape": list(atom["u"].shape),
                "v_shape": list(atom["v"].shape),
            }
        )
    return metadata
