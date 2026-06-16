"""Tests for global-only Jury-LoRA training orchestration."""

import csv
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.federated.trainer_jury as trainer_jury


class MappingLoader:
    def __init__(self, batches: list[dict[str, torch.Tensor]], size: int) -> None:
        self.batches = batches
        self.dataset = list(range(size))

    def __iter__(self):
        return iter(self.batches)


class ToyClient:
    def __init__(self, loader: MappingLoader) -> None:
        self.client_id = 3
        self.task_name = "sst2"
        self.train_dataloader = loader
        self.loaded_classifier = 0
        self.saved_classifier = 0

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataloader.dataset)

    @property
    def has_private_classifier(self) -> bool:
        return True

    def load_private_classifier(self, model: nn.Module) -> None:
        self.loaded_classifier += 1
        model.classifier_loaded = True

    def save_private_classifier(self, model: nn.Module) -> None:
        self.saved_classifier += 1


class ToyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(1, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 1))
        self.classifier = nn.Linear(2, 2, bias=False)
        self.initial_classifier_weight = self.classifier.weight.detach().clone()
        self.classifier_loaded = False

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        delta = self.lora_B @ self.lora_A
        logits = input_values @ delta.T + self.classifier(input_values)
        return SimpleNamespace(loss=nn.functional.cross_entropy(logits, labels))


def _training_config() -> dict:
    return {
        "federated": {
            "local_epochs": 1,
            "lr": 0.1,
            "weight_decay": 0.0,
        },
        "jury": {"vote_batch_size": 2},
        "data": {"tasks": ["sst2"]},
    }


def test_local_training_uses_one_fresh_memory_application_and_lora_upload() -> None:
    loader = MappingLoader(
        [{"input_values": torch.eye(2), "label": torch.tensor([0, 1])}],
        size=2,
    )
    client = ToyClient(loader)
    built_models: list[ToyTrainModel] = []
    applied: list[tuple[nn.Module, object]] = []

    def build_model(task_name: str, config: object) -> ToyTrainModel:
        assert task_name == "sst2"
        model = ToyTrainModel()
        built_models.append(model)
        return model

    def apply_memory(model: nn.Module, memory: object) -> None:
        applied.append((model, memory))

    memory = {"layer": []}
    upload = trainer_jury.train_client_with_memory(
        client,
        memory,
        _training_config(),
        "cpu",
        model_builder=build_model,
        memory_applier=apply_memory,
    )

    assert len(built_models) == 1
    assert applied == [(built_models[0], memory)]
    assert client.loaded_classifier == 1
    assert client.saved_classifier == 1
    assert set(upload["lora_state_dict"]) == {"lora_A", "lora_B"}
    assert all(
        tensor.device.type == "cpu"
        and not tensor.requires_grad
        for tensor in upload["lora_state_dict"].values()
    )
    assert built_models[0].classifier.weight.requires_grad is False
    torch.testing.assert_close(
        built_models[0].classifier.weight,
        built_models[0].initial_classifier_weight,
    )
    assert upload["average_train_loss"] > 0


def test_voting_uses_fresh_model_private_classifier_and_truncated_batch() -> None:
    loader = MappingLoader(
        [{"input_values": torch.arange(10).reshape(5, 2), "label": torch.arange(5)}],
        size=5,
    )
    client = ToyClient(loader)
    selected = {"query": [{"atom_id": "a"}]}
    built_models: list[ToyTrainModel] = []
    applied: list[object] = []
    observed: dict[str, object] = {}

    def build_model(task_name: str, config: object) -> ToyTrainModel:
        model = ToyTrainModel()
        built_models.append(model)
        return model

    def vote_computer(
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        selected_atoms: object,
        config: object,
        client_id: int,
        task_name: str,
        device: object,
    ) -> list[dict]:
        observed.update(
            model=model,
            batch=batch,
            selected=selected_atoms,
            client_id=client_id,
            task_name=task_name,
        )
        return [{"atom_id": "a", "layer_name": "query", "score": -1.0, "vote": "accept", "client_id": client_id, "task_name": task_name}]

    records = trainer_jury.vote_for_selected_atoms(
        client,
        {"old": []},
        selected,
        _training_config(),
        "cpu",
        model_builder=build_model,
        memory_applier=lambda model, memory: applied.append(memory),
        vote_computer=vote_computer,
    )

    assert len(built_models) == 1
    assert applied == [{"old": []}]
    assert client.loaded_classifier == 1
    assert built_models[0].classifier_loaded is True
    assert observed["batch"]["input_values"].shape[0] == 2
    assert observed["batch"]["label"].shape[0] == 2
    assert records[0]["vote"] == "accept"


def test_evaluation_uses_each_client_private_classifier() -> None:
    class EvalModel(nn.Module):
        def __init__(self, task_name: str) -> None:
            super().__init__()
            self.task_name = task_name
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, input_values, labels):
            logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]]) + self.anchor
            return SimpleNamespace(
                logits=logits,
                loss=nn.functional.cross_entropy(logits, labels),
            )

    built: list[EvalModel] = []
    applied: list[tuple[EvalModel, object]] = []
    config = {"data": {"tasks": ["sst2", "mrpc"]}}
    loaders = {
        task: MappingLoader(
            [{"input_values": torch.zeros(2, 1), "labels": torch.tensor([0, 1])}],
            size=2,
        )
        for task in config["data"]["tasks"]
    }

    def builder(task_name: str, config: object) -> EvalModel:
        model = EvalModel(task_name)
        built.append(model)
        return model

    clients = [
        ToyClient(MappingLoader([], size=1)),
        ToyClient(MappingLoader([], size=3)),
        ToyClient(MappingLoader([], size=2)),
    ]
    clients[0].task_name = "sst2"
    clients[1].task_name = "sst2"
    clients[2].task_name = "mrpc"
    memory = {"query": []}
    metrics = trainer_jury.evaluate_global_memory(
        memory,
        loaders,
        clients,
        config,
        "cpu",
        model_builder=builder,
        memory_applier=lambda model, current: applied.append((model, current)),
    )

    assert [model.task_name for model in built] == ["sst2", "sst2", "mrpc"]
    assert applied == [
        (built[0], memory),
        (built[1], memory),
        (built[2], memory),
    ]
    assert [client.loaded_classifier for client in clients] == [1, 1, 1]
    assert all(model.classifier_loaded for model in built)
    assert [row["accuracy"] for row in metrics] == [1.0, 1.0]
    assert metrics[0]["f1"] is None
    assert metrics[1]["f1"] == 1.0


def test_memory_and_vote_summary_rows_track_surviving_atoms() -> None:
    previous = {"query": [{"source_atom_id": "old", "round_idx": 1, "lambda_g": 2.0}]}
    updated = {
        "query": [
            {"source_atom_id": "old", "round_idx": 1, "lambda_g": 1.0},
            {"source_atom_id": "new", "round_idx": 2, "lambda_g": 3.0},
        ]
    }
    selected = {"query": [{"atom_id": "new", "sigma": 2.0}, {"atom_id": "pruned", "sigma": 1.0}]}
    stats = [
        {"atom_id": "new", "p_accept": 1.0, "p_reject": 0.0},
        {"atom_id": "pruned", "p_accept": 1.0, "p_reject": 0.0},
    ]
    config = {"jury": {"theta_high": 0.5, "eta_memory": 1.0}}

    vote_rows = trainer_jury.build_vote_stats_rows(stats, updated)
    memory_rows = trainer_jury.build_memory_stats_rows(
        previous,
        updated,
        selected,
        stats,
        config,
        round_idx=2,
    )

    assert [row["accepted_to_memory"] for row in vote_rows] == [True, False]
    assert memory_rows == [
        {
            "round_idx": 2,
            "layer_name": "query",
            "memory_size": 2,
            "added_atoms": 1,
            "pruned_atoms": 1,
            "mean_lambda_g": 2.0,
            "max_lambda_g": 3.0,
        }
    ]


def test_mocked_full_round_writes_exact_outputs(tmp_path, monkeypatch) -> None:
    config = {
        "seed": 4,
        "model": {"name": "toy"},
        "data": {"tasks": ["sst2"], "num_clients": 1},
        "federated": {"rounds": 1},
        "jury": {
            "enabled": True,
            "mode": "global_only",
            "top_k_mode": "sigma",
            "vote_batch_size": 1,
            "theta_high": 0.5,
            "eta_memory": 1.0,
            "alpha_memory": 1.0,
            "global_memory_budget_per_layer": 2,
        },
        "output": {"dir": str(tmp_path)},
    }
    atom = {
        "atom_id": "atom",
        "round_idx": 1,
        "client_id": 0,
        "task_name": "sst2",
        "layer_name": "query",
        "rank_id": 0,
        "u": torch.tensor([1.0, 0.0]),
        "v": torch.tensor([0.0, 1.0]),
        "sigma": torch.tensor(2.0),
        "source_lora_rank": 1,
    }
    call_order: list[str] = []

    class Tokenizer:
        @staticmethod
        def from_pretrained(name: str) -> object:
            return object()

    class Client:
        def __init__(self, client_id, task_name, train_dataloader, config, device):
            self.client_id = client_id
            self.task_name = task_name
            self.train_dataloader = train_dataloader
            self.num_train_samples = 1

    loader = MappingLoader([{"labels": torch.tensor([0])}], size=1)
    monkeypatch.setattr(trainer_jury, "load_config", lambda path: config)
    monkeypatch.setattr(trainer_jury, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        trainer_jury,
        "_load_runtime_dependencies",
        lambda: (
            Tokenizer,
            lambda cfg, tokenizer: (
                [{"client_id": 0, "task_name": "sst2", "train_dataloader": loader}],
                {"sst2": loader},
            ),
            Client,
        ),
    )

    def train(client, memory, config, device):
        call_order.append(f"train:{bool(memory)}")
        return {
            "client_id": 0,
            "task_name": "sst2",
            "num_train_samples": 1,
            "average_train_loss": 0.25,
            "lora_state_dict": {
                "query.lora_A.default.weight": torch.ones(1, 2),
                "query.lora_B.default.weight": torch.ones(2, 1),
            },
        }

    monkeypatch.setattr(trainer_jury, "train_client_with_memory", train)
    monkeypatch.setattr(
        trainer_jury,
        "extract_atoms_from_round",
        lambda updates, cfg, device: call_order.append("extract") or [atom],
    )
    monkeypatch.setattr(
        trainer_jury,
        "select_topk_atoms_by_sigma",
        lambda atoms, cfg: call_order.append("select") or {"query": [atom]},
    )

    def vote(client, memory, selected, config, device):
        call_order.append(f"vote:{bool(memory)}")
        return [{
            "client_id": 0,
            "task_name": "sst2",
            "atom_id": "atom",
            "layer_name": "query",
            "score": -1.0,
            "vote": "accept",
        }]

    monkeypatch.setattr(trainer_jury, "vote_for_selected_atoms", vote)

    def evaluate(memory, loaders, clients, config, device):
        call_order.append(f"evaluate:{bool(memory)}")
        return [{"task_name": "sst2", "accuracy": 0.75, "f1": None, "eval_loss": 0.5}]

    monkeypatch.setattr(trainer_jury, "evaluate_global_memory", evaluate)

    trainer_jury.run_jury_global_training("config.yaml")

    assert call_order == [
        "train:False",
        "extract",
        "select",
        "vote:False",
        "evaluate:True",
    ]
    for filename, columns in trainer_jury.OUTPUT_SCHEMAS.items():
        with (tmp_path / filename).open(newline="", encoding="utf-8") as output_file:
            reader = csv.DictReader(output_file)
            rows = list(reader)
            assert reader.fieldnames == columns
            assert rows
    with (tmp_path / "vote_stats.csv").open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file))[0]["accepted_to_memory"] == "True"
    with (tmp_path / "memory_stats.csv").open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file))[0]["added_atoms"] == "1"


def test_initialize_output_files_overwrites_existing_rows(tmp_path) -> None:
    path = tmp_path / "round_metrics.csv"
    path.write_text("old data", encoding="utf-8")

    trainer_jury.initialize_output_files(tmp_path)

    assert path.read_text(encoding="utf-8").splitlines() == [
        ",".join(trainer_jury.ROUND_METRIC_COLUMNS)
    ]


@pytest.mark.parametrize(
    "jury_update",
    [
        {"top_k_mode": "entropy"},
        {"vote_batch_size": 0},
    ],
)
def test_run_rejects_unsupported_selection_or_vote_batch(
    tmp_path,
    monkeypatch,
    jury_update,
) -> None:
    config = {
        "seed": 1,
        "jury": {
            "enabled": True,
            "mode": "global_only",
            "top_k_mode": "sigma",
            "vote_batch_size": 1,
            **jury_update,
        },
        "output": {"dir": str(tmp_path)},
    }
    monkeypatch.setattr(trainer_jury, "load_config", lambda path: config)
    with pytest.raises(ValueError):
        trainer_jury.run_jury_global_training("config.yaml")
