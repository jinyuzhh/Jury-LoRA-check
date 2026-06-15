"""Utilities for handling LoRA-only model state."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"


def _is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return detached CPU copies of only the model's LoRA A/B parameters."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if _is_lora_parameter(name)
    }


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = False,
) -> Any:
    """Load a state dictionary containing only LoRA A/B parameters."""
    invalid_keys = [name for name in state_dict if not _is_lora_parameter(name)]
    if invalid_keys:
        formatted_keys = ", ".join(sorted(invalid_keys))
        raise ValueError(
            "LoRA state dictionary contains non-LoRA parameters: "
            f"{formatted_keys}."
        )

    return model.load_state_dict(dict(state_dict), strict=strict)


def extract_lora_A_B(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Group default-adapter LoRA A and B matrices by adapted layer name."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in state_dict.items():
        matrix_name: str | None = None
        layer_name: str | None = None
        if name.endswith(_LORA_A_SUFFIX):
            matrix_name = "A"
            layer_name = name[: -len(_LORA_A_SUFFIX)]
        elif name.endswith(_LORA_B_SUFFIX):
            matrix_name = "B"
            layer_name = name[: -len(_LORA_B_SUFFIX)]

        if matrix_name is None or layer_name is None:
            continue
        if tensor.ndim != 2:
            raise ValueError(
                f"LoRA {matrix_name} matrix for layer {layer_name!r} must be "
                f"2D, but has shape {tuple(tensor.shape)}."
            )
        if matrix_name in grouped.setdefault(layer_name, {}):
            raise ValueError(
                f"Duplicate LoRA {matrix_name} matrix for layer {layer_name!r}."
            )
        grouped[layer_name][matrix_name] = tensor

    for layer_name, matrices in grouped.items():
        if "A" not in matrices:
            raise ValueError(
                f"Layer {layer_name!r} has a LoRA B matrix but no LoRA A matrix."
            )
        if "B" not in matrices:
            raise ValueError(
                f"Layer {layer_name!r} has a LoRA A matrix but no LoRA B matrix."
            )

        matrix_a = matrices["A"]
        matrix_b = matrices["B"]
        if matrix_a.shape[0] != matrix_b.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for layer {layer_name!r}: "
                f"A has shape {tuple(matrix_a.shape)} and B has shape "
                f"{tuple(matrix_b.shape)}."
            )

    return grouped


def count_trainable_parameters(model: nn.Module) -> None:
    """Print the model's trainable, total, and percentage parameter counts."""
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_percent = (
        100.0 * trainable_params / total_params if total_params else 0.0
    )
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_percent:.2f}%)"
    )


def print_lora_layers(model: nn.Module) -> None:
    """Print LoRA A/B parameter names and shapes in model order."""
    for name, parameter in model.named_parameters():
        if _is_lora_parameter(name):
            print(f"{name}: shape={tuple(parameter.shape)}")
