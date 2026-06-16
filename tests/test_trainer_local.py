"""Tests for single-client local diagnostics."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

import src.federated.trainer_local as trainer_local


class MappingLoader:
    def __init__(self, batches: list[dict[str, torch.Tensor]], size: int) -> None:
        self.batches = batches
        self.dataset = list(range(size))

    def __iter__(self):
        return iter(self.batches)


class Tokenizer:
    @staticmethod
    def from_pretrained(name: str) -> object:
        return object()


class LocalClient:
    trained_ids: list[int] = []
    loaded_ids: list[int] = []

    def __init__(self, client_id, task_name, train_dataloader, config, device):
        self.client_id = client_id
        self.task_name = task_name
        self.train_dataloader = train_dataloader
        self.config = config
        self.device = device

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataloader.dataset)

    def train(self, initial_lora_state):
        self.trained_ids.append(self.client_id)
        assert set(initial_lora_state) == {"lora_A", "lora_B"}
        return {
            "client_id": self.client_id,
            "task_name": self.task_name,
            "num_train_samples": self.num_train_samples,
            "average_train_loss": 0.125,
            "lora_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in initial_lora_state.items()
            },
        }

    def load_private_classifier(self, model: nn.Module) -> None:
        self.loaded_ids.append(self.client_id)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(1, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 1))
        self.classifier = nn.Linear(2, 2, bias=False)
        self.classifier.weight.requires_grad_(False)

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor):
        logits = input_values + self.classifier(input_values) * 0
        return SimpleNamespace(
            logits=logits,
            loss=nn.functional.cross_entropy(logits, labels),
        )


def _config(tmp_path) -> dict:
    return {
        "seed": 1,
        "model": {"name": "toy"},
        "data": {"tasks": ["sst2"]},
        "federated": {},
        "output": {"dir": str(tmp_path)},
    }


def _client_rows() -> tuple[list[dict], dict[str, MappingLoader]]:
    client_zero_loader = MappingLoader([{"input_values": torch.eye(2)}], size=2)
    client_four_loader = MappingLoader([{"input_values": torch.ones(1, 2)}], size=1)
    eval_loader = MappingLoader(
        [{"input_values": torch.eye(2), "labels": torch.tensor([0, 1])}],
        size=2,
    )
    return (
        [
            {
                "client_id": 0,
                "task_name": "sst2",
                "train_dataloader": client_zero_loader,
            },
            {
                "client_id": 4,
                "task_name": "sst2",
                "train_dataloader": client_four_loader,
            },
        ],
        {"sst2": eval_loader},
    )


def test_local_diagnostic_trains_only_selected_client_and_writes_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    LocalClient.trained_ids = []
    LocalClient.loaded_ids = []
    monkeypatch.setattr(trainer_local, "load_config", lambda path: _config(tmp_path))
    monkeypatch.setattr(trainer_local, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        trainer_local,
        "_load_runtime_dependencies",
        lambda: (
            Tokenizer,
            lambda config, tokenizer: _client_rows(),
            LocalClient,
        ),
    )

    result = trainer_local.run_local_client_training(
        "config.yaml",
        client_id=4,
        model_builder=lambda task_name, config: ToyModel(),
    )

    assert LocalClient.trained_ids == [4]
    assert LocalClient.loaded_ids == [4]
    assert result["train_log"]["client_id"] == 4
    assert result["metrics"]["accuracy"] == 1.0

    train_log = pd.read_csv(tmp_path / "local_client_train_log.csv")
    metrics = pd.read_csv(tmp_path / "local_client_metrics.csv")
    assert train_log.to_dict("records") == [
        {
            "client_id": 4,
            "task_name": "sst2",
            "num_train_samples": 1,
            "train_loss": 0.125,
        }
    ]
    assert metrics.loc[0, "client_id"] == 4
    assert metrics.loc[0, "task_name"] == "sst2"
    assert metrics.loc[0, "accuracy"] == 1.0


def test_local_diagnostic_rejects_unknown_client_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trainer_local, "load_config", lambda path: _config(tmp_path))
    monkeypatch.setattr(trainer_local, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        trainer_local,
        "_load_runtime_dependencies",
        lambda: (
            Tokenizer,
            lambda config, tokenizer: _client_rows(),
            LocalClient,
        ),
    )

    with pytest.raises(ValueError, match="client_id 99"):
        trainer_local.run_local_client_training(
            "config.yaml",
            client_id=99,
            model_builder=lambda task_name, config: ToyModel(),
        )
