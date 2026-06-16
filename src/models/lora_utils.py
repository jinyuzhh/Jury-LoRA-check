"""Utilities for handling LoRA-only model state."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
_HEAD_NAME_MARKERS = ("classifier", "score", "classification_head", "lm_head")
_ADAPTER_NAME_MARKERS = ("lora", "pissa", "svd")


def _is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def is_head_parameter(name: str) -> bool:
    """Return whether a parameter name belongs to a task-specific head."""
    lowered = name.lower()
    return any(marker in lowered for marker in _HEAD_NAME_MARKERS)


def is_adapter_parameter(name: str) -> bool:
    """Return whether a parameter name belongs to a trainable adapter."""
    lowered = name.lower()
    return any(marker in lowered for marker in _ADAPTER_NAME_MARKERS)


def freeze_head_parameters(model: nn.Module) -> None:
    """Freeze classifier/task-head parameters in-place."""
    for name, parameter in model.named_parameters():
        if is_head_parameter(name):
            parameter.requires_grad_(False)


def assert_no_trainable_heads(model: nn.Module) -> None:
    """Raise if any classifier/task-head parameter is still trainable."""
    trainable_heads = [
        name
        for name, parameter in model.named_parameters()
        if is_head_parameter(name) and parameter.requires_grad
    ]
    if trainable_heads:
        raise AssertionError(
            "Classifier/head parameters must be frozen, but these remain "
            "trainable: "
            + ", ".join(trainable_heads)
        )


def print_trainable_parameter_names(model: nn.Module) -> None:
    """Print trainable parameter names and assert task heads are frozen."""
    assert_no_trainable_heads(model)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print("Trainable parameters:")
    if not trainable_names:
        print("  <none>")
    for name in trainable_names:
        print(f"  {name}")


def trainable_adapter_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    """Return only trainable adapter parameters for optimizer construction."""
    assert_no_trainable_heads(model)
    invalid_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not is_adapter_parameter(name)
    ]
    if invalid_trainable:
        raise AssertionError(
            "Only adapter parameters may be trainable, but these non-adapter "
            "parameters require gradients: "
            + ", ".join(invalid_trainable)
        )
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and is_adapter_parameter(name)
    ]
    if not adapter_parameters:
        raise ValueError("Model does not expose any trainable adapter parameters.")
    return adapter_parameters


def get_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return detached CPU copies of classifier/task-head tensors."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if is_head_parameter(name)
    }


def assert_head_state_unchanged(
    before: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> None:
    """Raise if any classifier/task-head tensor changed."""
    after = get_head_state_dict(model)
    if set(after) != set(before):
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        raise AssertionError("Classifier/head state keys changed; " + "; ".join(details))
    for name, before_tensor in before.items():
        torch.testing.assert_close(after[name], before_tensor)


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

    result = model.load_state_dict(dict(state_dict), strict=strict)
    assert_no_trainable_heads(model)
    return result


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
