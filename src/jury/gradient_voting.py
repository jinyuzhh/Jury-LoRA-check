"""Compute client votes for selected atoms using temporary gradient gates."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle


LOGGER = logging.getLogger(__name__)

_REQUIRED_ATOM_FIELDS = (
    "atom_id",
    "layer_name",
    "u",
    "v",
    "sigma",
    "client_id",
    "rank_id",
    "task_name",
)


@dataclass
class AtomGateHandle:
    """Own a layer hook and its temporary atom-gate parameters."""

    hook: RemovableHandle
    atoms: list[Mapping[str, Any]]
    gates: list[nn.Parameter]


def find_module_by_layer_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Return the module whose full name exactly matches ``layer_name``."""
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError("layer_name must be a nonempty string.")
    for module_name, module in model.named_modules():
        if module_name == layer_name:
            return module
    raise ValueError(f"Model does not contain target layer {layer_name!r}.")


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


def _validate_atom(atom: Any, layer_name: str) -> Mapping[str, Any]:
    if not isinstance(atom, Mapping):
        raise ValueError(f"Each atom for layer {layer_name!r} must be a mapping.")
    missing_fields = [field for field in _REQUIRED_ATOM_FIELDS if field not in atom]
    if missing_fields:
        raise ValueError(
            f"Atom for layer {layer_name!r} is missing required field(s): "
            + ", ".join(missing_fields)
            + "."
        )
    if atom["layer_name"] != layer_name:
        raise ValueError(
            f"Atom {atom['atom_id']!r} belongs to layer {atom['layer_name']!r}, "
            f"not {layer_name!r}."
        )
    for vector_name in ("u", "v"):
        vector = atom[vector_name]
        if not torch.is_tensor(vector):
            raise ValueError(
                f"Atom {atom['atom_id']!r} {vector_name} must be a tensor."
            )
        if vector.ndim != 1:
            raise ValueError(
                f"Atom {atom['atom_id']!r} {vector_name} must be 1D, but has "
                f"shape {tuple(vector.shape)}."
            )
    return atom


def create_atom_gates_for_layer(
    atoms: Sequence[Mapping[str, Any]],
    module: nn.Module,
    device: torch.device | str,
) -> AtomGateHandle:
    """Attach zero-initialized atom gates to one target layer via a hook."""
    if not atoms:
        raise ValueError("At least one atom is required to create layer gates.")

    weight = _target_weight(module)
    voting_device = weight.device
    if weight.device != voting_device:
        raise ValueError(
            f"Target layer weight is on {weight.device}, but voting device is "
            f"{voting_device}."
        )

    first_atom = atoms[0]
    if not isinstance(first_atom, Mapping) or "layer_name" not in first_atom:
        raise ValueError("Each atom must contain layer_name.")
    layer_name = first_atom["layer_name"]
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError("Atom layer_name must be a nonempty string.")

    validated_atoms: list[Mapping[str, Any]] = []
    residuals: list[torch.Tensor] = []
    gates: list[nn.Parameter] = []
    for atom_value in atoms:
        atom = _validate_atom(atom_value, layer_name)
        vector_u = atom["u"].detach().to(device=weight.device, dtype=weight.dtype)
        vector_v = atom["v"].detach().to(device=weight.device, dtype=weight.dtype)
        residual = torch.outer(vector_u, vector_v)
        if residual.shape != weight.shape:
            raise ValueError(
                f"Atom {atom['atom_id']!r} outer(u, v) has shape "
                f"{tuple(residual.shape)}, but target layer weight has shape "
                f"{tuple(weight.shape)}."
            )
        validated_atoms.append(atom)
        residuals.append(residual)
        gates.append(
            nn.Parameter(
                torch.zeros((), device=weight.device, dtype=weight.dtype),
                requires_grad=True,
            )
        )

    def add_gated_residual(
        _module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> torch.Tensor:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise ValueError(
                f"Target layer {layer_name!r} must receive its input tensor as "
                "the first positional argument."
            )
        if not torch.is_tensor(output):
            raise ValueError(
                f"Target layer {layer_name!r} must return a tensor for voting."
            )
        gated_weight = torch.stack(
            [gate * residual for gate, residual in zip(gates, residuals)]
        ).sum(dim=0)
        return output + F.linear(inputs[0], gated_weight)

    hook = module.register_forward_hook(add_gated_residual)
    return AtomGateHandle(hook=hook, atoms=validated_atoms, gates=gates)


def remove_hooks_and_cleanup(handles: Sequence[AtomGateHandle]) -> None:
    """Remove voting hooks and clear all temporary gate gradients."""
    for handle in handles:
        handle.hook.remove()
        for gate in handle.gates:
            gate.grad = None
        handle.atoms.clear()
        handle.gates.clear()


def _get_voting_config(config: Mapping[str, Any]) -> tuple[float, bool, bool]:
    try:
        jury_config = config["jury"]
        threshold_value = jury_config["gamma_threshold"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "config must contain jury.gamma_threshold as a nonnegative number."
        ) from error

    if isinstance(threshold_value, bool) or not isinstance(threshold_value, Real):
        raise ValueError("config['jury']['gamma_threshold'] must be nonnegative.")
    threshold = float(threshold_value)
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("config['jury']['gamma_threshold'] must be nonnegative.")

    vote_eval_mode = jury_config.get("vote_eval_mode", False)
    debug_voting = jury_config.get("debug_voting", False)
    if not isinstance(vote_eval_mode, bool):
        raise ValueError("config['jury']['vote_eval_mode'] must be a boolean.")
    if not isinstance(debug_voting, bool):
        raise ValueError("config['jury']['debug_voting'] must be a boolean.")
    return threshold, vote_eval_mode, debug_voting


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be a mapping.")
    moved_batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    if "labels" not in moved_batch and "label" in moved_batch:
        moved_batch["labels"] = moved_batch.pop("label")
    return moved_batch


def _extract_loss(outputs: Any) -> torch.Tensor:
    loss = (
        outputs.get("loss")
        if isinstance(outputs, Mapping)
        else getattr(outputs, "loss", None)
    )
    if not torch.is_tensor(loss):
        raise ValueError("Voting model output must contain a tensor loss.")
    if loss.numel() != 1:
        raise ValueError(
            f"Voting loss must be scalar, but has shape {tuple(loss.shape)}."
        )
    if not loss.requires_grad:
        raise ValueError("Voting loss is not connected to the atom gates.")
    return loss


def compute_gradient_votes(
    model: nn.Module,
    batch: Mapping[str, Any],
    selected_atoms_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    client_id: Any,
    task_name: Any,
    device: torch.device | str,
) -> list[dict[str, Any]]:
    """Compute one client's gradient vote for every selected atom."""
    threshold, vote_eval_mode, debug_voting = _get_voting_config(config)
    if not isinstance(selected_atoms_by_layer, Mapping):
        raise ValueError("selected_atoms_by_layer must be a mapping by layer.")

    voting_layers: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    selected_atom_count = 0
    for layer_name, atoms in selected_atoms_by_layer.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("Selected atom layer names must be nonempty strings.")
        if isinstance(atoms, (str, bytes)) or not isinstance(atoms, Sequence):
            raise ValueError(
                f"Selected atoms for layer {layer_name!r} must be a sequence."
            )
        if atoms:
            voting_layers.append((layer_name, atoms))
            selected_atom_count += len(atoms)

    if selected_atom_count == 0:
        if debug_voting:
            LOGGER.info("Gradient voting: 0 selected atoms; no model pass run.")
        return []

    voting_device = torch.device(device)
    moved_batch = _move_batch_to_device(batch, voting_device)
    module_training_states = {
        module: module.training for module in model.modules()
    }
    parameter_grad_states = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    handles: list[AtomGateHandle] = []

    try:
        model.zero_grad(set_to_none=True)
        for parameter in parameter_grad_states:
            parameter.requires_grad_(False)

        if vote_eval_mode:
            model.eval()
        else:
            model.train()

        for layer_name, atoms in voting_layers:
            module = find_module_by_layer_name(model, layer_name)
            handles.append(create_atom_gates_for_layer(atoms, module, voting_device))

        outputs = model(**moved_batch)
        loss = _extract_loss(outputs)
        loss.backward()

        votes: list[dict[str, Any]] = []
        scores: list[float] = []
        vote_counts = {"accept": 0, "reject": 0, "abstain": 0}
        for handle in handles:
            for atom, gate in zip(handle.atoms, handle.gates):
                if gate.grad is None:
                    raise ValueError(
                        f"Atom {atom['atom_id']!r} gate did not receive a gradient."
                    )
                score = float(gate.grad.detach().cpu().item())
                if score < -threshold:
                    vote = "accept"
                elif score > threshold:
                    vote = "reject"
                else:
                    vote = "abstain"
                scores.append(score)
                vote_counts[vote] += 1
                votes.append(
                    {
                        "client_id": client_id,
                        "task_name": task_name,
                        "atom_id": atom["atom_id"],
                        "layer_name": atom["layer_name"],
                        "score": score,
                        "vote": vote,
                    }
                )

        if debug_voting:
            mean_score = sum(scores) / len(scores)
            LOGGER.info(
                "Gradient voting: atoms=%d matched_layers=%d accept=%d reject=%d "
                "abstain=%d score_min=%.6g score_mean=%.6g score_max=%.6g",
                selected_atom_count,
                len(handles),
                vote_counts["accept"],
                vote_counts["reject"],
                vote_counts["abstain"],
                min(scores),
                mean_score,
                max(scores),
            )
        return votes
    finally:
        remove_hooks_and_cleanup(handles)
        for parameter, requires_grad in parameter_grad_states.items():
            parameter.requires_grad_(requires_grad)
        for module, training in module_training_states.items():
            module.training = training
        model.zero_grad(set_to_none=True)
