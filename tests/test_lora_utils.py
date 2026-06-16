"""Tests for LoRA-only state and private-head freezing utilities."""

import pytest
import torch
from torch import nn

from src.federated.server_fedavg import fedavg_lora_states
from src.models.lora_utils import (
    assert_head_state_unchanged,
    assert_no_trainable_heads,
    freeze_head_parameters,
    get_head_state_dict,
    get_lora_state_dict,
    load_lora_state_dict,
    trainable_adapter_parameters,
)


class HeadAndAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(1, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 1))
        self.backbone = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2)
        self.score = nn.Linear(2, 1)
        self.classification_head = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 3)


def test_freezes_all_known_head_names_and_keeps_adapters_trainable() -> None:
    model = HeadAndAdapterModel()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)

    freeze_head_parameters(model)

    assert_no_trainable_heads(model)
    assert model.lora_A.requires_grad
    assert model.lora_B.requires_grad
    assert all(not parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(not parameter.requires_grad for parameter in model.score.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.classification_head.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.lm_head.parameters())
    trainable_parameters = trainable_adapter_parameters(model)
    assert [id(parameter) for parameter in trainable_parameters] == [
        id(model.lora_A),
        id(model.lora_B),
    ]


def test_rejects_trainable_heads_and_non_adapter_optimizer_parameters() -> None:
    model = HeadAndAdapterModel()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(AssertionError, match="Classifier/head"):
        assert_no_trainable_heads(model)

    freeze_head_parameters(model)
    model.backbone.weight.requires_grad_(True)

    with pytest.raises(AssertionError, match="non-adapter"):
        trainable_adapter_parameters(model)


def test_lora_state_excludes_heads_and_shared_load_rejects_head_keys() -> None:
    model = HeadAndAdapterModel()

    state = get_lora_state_dict(model)

    assert set(state) == {"lora_A", "lora_B"}
    assert all("classifier" not in key for key in state)
    with pytest.raises(ValueError, match="non-LoRA"):
        load_lora_state_dict(model, {"classifier.weight": torch.ones(2, 2)})


def test_fedavg_rejects_classifier_keys_in_uploaded_lora_state() -> None:
    with pytest.raises(ValueError, match="non-LoRA"):
        fedavg_lora_states(
            [
                {
                    "num_train_samples": 1,
                    "lora_state_dict": {
                        "lora_A": torch.ones(1, 2),
                        "classifier.weight": torch.ones(2, 2),
                    },
                }
            ]
        )


def test_head_state_unchanged_assertion_detects_weight_updates() -> None:
    model = HeadAndAdapterModel()
    before = get_head_state_dict(model)

    assert_head_state_unchanged(before, model)
    with torch.no_grad():
        model.classifier.weight.add_(1.0)

    with pytest.raises(AssertionError):
        assert_head_state_unchanged(before, model)
