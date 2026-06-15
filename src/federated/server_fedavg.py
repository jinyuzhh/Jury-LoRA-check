"""FedAvg aggregation for LoRA-only client updates."""

from typing import Any

import torch


def _is_lora_key(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def fedavg_lora_states(
    client_updates: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Compute a sample-weighted average of matching client LoRA states."""
    if not client_updates:
        raise ValueError("FedAvg requires at least one client update.")

    reference_state = client_updates[0].get("lora_state_dict")
    if not isinstance(reference_state, dict) or not reference_state:
        raise ValueError("Client updates must contain a nonempty lora_state_dict.")
    reference_keys = set(reference_state)
    non_lora_keys = sorted(key for key in reference_keys if not _is_lora_key(key))
    if non_lora_keys:
        raise ValueError(
            "Client LoRA state contains non-LoRA parameters: "
            + ", ".join(non_lora_keys)
        )

    total_samples = 0
    validated_updates: list[tuple[int, dict[str, torch.Tensor]]] = []
    for update in client_updates:
        num_samples = update.get("num_train_samples")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise ValueError("Client num_train_samples must be a nonnegative integer.")
        if num_samples < 0:
            raise ValueError("Client num_train_samples must be a nonnegative integer.")

        state = update.get("lora_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Each client update must contain a lora_state_dict.")
        state_keys = set(state)
        if state_keys != reference_keys:
            missing = sorted(reference_keys - state_keys)
            extra = sorted(state_keys - reference_keys)
            details = []
            if missing:
                details.append("missing keys: " + ", ".join(missing))
            if extra:
                details.append("extra keys: " + ", ".join(extra))
            raise ValueError("Client LoRA keys do not match; " + "; ".join(details))

        for key in reference_keys:
            if not _is_lora_key(key):
                raise ValueError(f"Uploaded key {key!r} is not a LoRA A/B parameter.")
            if not torch.is_tensor(state[key]):
                raise ValueError(f"Uploaded LoRA value {key!r} must be a tensor.")
            if state[key].shape != reference_state[key].shape:
                raise ValueError(
                    f"Shape mismatch for {key!r}: expected "
                    f"{tuple(reference_state[key].shape)}, got {tuple(state[key].shape)}."
                )

        total_samples += num_samples
        validated_updates.append((num_samples, state))

    if total_samples <= 0:
        raise ValueError("FedAvg requires a positive total client sample count.")

    averaged_state: dict[str, torch.Tensor] = {}
    for key, reference_tensor in reference_state.items():
        accumulator = torch.zeros(
            reference_tensor.shape,
            dtype=torch.float64,
            device="cpu",
        )
        for num_samples, state in validated_updates:
            if num_samples == 0:
                continue
            accumulator.add_(
                state[key].detach().to(device="cpu", dtype=torch.float64),
                alpha=num_samples,
            )
        averaged_state[key] = (accumulator / total_samples).to(
            dtype=reference_tensor.dtype
        )

    return averaged_state
