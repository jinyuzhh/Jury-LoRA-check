"""Tests for temporary gradient-gated atom voting."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.jury.gradient_voting import (
    compute_gradient_votes,
    create_atom_gates_for_layer,
    find_module_by_layer_name,
    remove_hooks_and_cleanup,
)


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(2, 2, bias=False)
        self.value = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.query.weight.zero_()
            self.value.weight.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.query(inputs) + self.value(inputs)


class ToyVotingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = ToyAttention()
        self.classifier = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.classifier.weight.fill_(1.0)
        self.forward_count = 0

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        self.forward_count += 1
        prediction = self.classifier(self.attention(input_values)).squeeze(-1)
        return SimpleNamespace(loss=((prediction - labels) ** 2).mean())


class BaseLayerWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs)


def _atom(
    atom_id: str,
    layer_name: str,
    u: list[float],
    v: list[float],
) -> dict:
    return {
        "atom_id": atom_id,
        "layer_name": layer_name,
        "u": torch.tensor(u),
        "v": torch.tensor(v),
        "sigma": torch.tensor(1.0),
        "client_id": 99,
        "rank_id": 0,
        "task_name": "source",
    }


def _config(threshold: float = 0.0, **jury_values: object) -> dict:
    return {"jury": {"gamma_threshold": threshold, **jury_values}}


def test_finds_exact_nested_module() -> None:
    model = ToyVotingModel()

    module = find_module_by_layer_name(model, "attention.query")

    assert module is model.attention.query
    with pytest.raises(ValueError, match="does not contain"):
        find_module_by_layer_name(model, "query")


def test_computes_analytical_votes_in_one_pass_and_restores_model() -> None:
    model = ToyVotingModel()
    model.train()
    model.attention.value.eval()
    model.attention.query.weight.requires_grad_(False)
    original_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    original_training = {module: module.training for module in model.modules()}
    original_requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    selected = {
        "attention.query": [
            _atom("accept", "attention.query", [1.0, 0.0], [1.0, 0.0]),
            _atom("reject", "attention.query", [-1.0, 0.0], [1.0, 0.0]),
            _atom("abstain", "attention.query", [0.0, 1.0], [0.0, 1.0]),
        ]
    }
    batch = {
        "input_values": torch.tensor([[1.0, 0.0]]),
        "label": torch.tensor([1.0]),
    }

    votes = compute_gradient_votes(
        model,
        batch,
        selected,
        _config(),
        client_id=7,
        task_name="sst2",
        device="cpu",
    )

    assert model.forward_count == 1
    assert [vote["vote"] for vote in votes] == ["accept", "reject", "abstain"]
    assert [vote["score"] for vote in votes] == pytest.approx([-2.0, 2.0, 0.0])
    assert [vote["client_id"] for vote in votes] == [7, 7, 7]
    assert [vote["task_name"] for vote in votes] == ["sst2", "sst2", "sst2"]
    assert len(model.attention.query._forward_hooks) == 0
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        module.training == original_training[module] for module in model.modules()
    )
    assert all(
        parameter.requires_grad == original_requires_grad[parameter]
        for parameter in model.parameters()
    )


def test_vote_threshold_uses_strict_comparisons() -> None:
    model = ToyVotingModel()
    selected = {
        "attention.query": [
            _atom("positive", "attention.query", [-1.0, 0.0], [1.0, 0.0]),
            _atom("negative", "attention.query", [1.0, 0.0], [1.0, 0.0]),
        ]
    }

    votes = compute_gradient_votes(
        model,
        {"input_values": torch.tensor([[1.0, 0.0]]), "labels": torch.tensor([1.0])},
        selected,
        _config(2.0),
        client_id=1,
        task_name="task",
        device="cpu",
    )

    assert [vote["vote"] for vote in votes] == ["abstain", "abstain"]


def test_vote_eval_mode_is_used_during_forward_and_restored() -> None:
    class ModeModel(ToyVotingModel):
        seen_training: bool | None = None

        def forward(self, **kwargs) -> SimpleNamespace:
            self.seen_training = self.training
            return super().forward(**kwargs)

    model = ModeModel()
    model.train()

    compute_gradient_votes(
        model,
        {"input_values": torch.ones(1, 2), "labels": torch.zeros(1)},
        {"attention.query": [_atom("atom", "attention.query", [1, 0], [1, 0])]},
        _config(vote_eval_mode=True),
        client_id=1,
        task_name="task",
        device="cpu",
    )

    assert model.seen_training is False
    assert model.training is True


def test_empty_selection_skips_model_pass() -> None:
    model = ToyVotingModel()

    votes = compute_gradient_votes(
        model,
        {},
        {},
        _config(),
        client_id=1,
        task_name="task",
        device="cpu",
    )

    assert votes == []
    assert model.forward_count == 0


def test_supports_base_layer_weight_and_manual_cleanup() -> None:
    module = BaseLayerWrapper()
    atom = _atom("wrapped", "wrapped", [1, 0], [1, 0])

    handle = create_atom_gates_for_layer([atom], module, "cpu")

    assert len(module._forward_hooks) == 1
    assert handle.gates[0].requires_grad
    remove_hooks_and_cleanup([handle])
    assert len(module._forward_hooks) == 0
    assert handle.gates == []


def test_rejects_missing_layer_and_weight_device_mismatch() -> None:
    model = ToyVotingModel()
    atom = _atom("missing", "attention.missing", [1, 0], [1, 0])
    with pytest.raises(ValueError, match="does not contain"):
        compute_gradient_votes(
            model,
            {"input_values": torch.ones(1, 2), "labels": torch.zeros(1)},
            {"attention.missing": [atom]},
            _config(),
            client_id=1,
            task_name="task",
            device="cpu",
        )
    assert all(parameter.grad is None for parameter in model.parameters())

    meta_module = nn.Linear(2, 2, bias=False, device="meta")
    with pytest.raises(ValueError, match="voting device"):
        create_atom_gates_for_layer(
            [_atom("meta", "meta", [1, 0], [1, 0])],
            meta_module,
            "cpu",
        )


@pytest.mark.parametrize(
    ("atom", "message"),
    [
        ({}, "layer_name"),
        (
            _atom("bad_u", "attention.query", [1, 0], [1, 0]) | {"u": torch.ones(1, 2)},
            "u must be 1D",
        ),
        (
            _atom("bad_shape", "attention.query", [1, 0, 0], [1, 0]),
            "outer.*shape",
        ),
    ],
)
def test_rejects_malformed_or_mismatched_atoms(atom: dict, message: str) -> None:
    model = ToyVotingModel()
    with pytest.raises(ValueError, match=message):
        create_atom_gates_for_layer([atom], model.attention.query, "cpu")


@pytest.mark.parametrize("threshold", [-1.0, True, "0", float("nan")])
def test_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="gamma_threshold"):
        compute_gradient_votes(
            ToyVotingModel(),
            {},
            {},
            {"jury": {"gamma_threshold": threshold}},
            client_id=1,
            task_name="task",
            device="cpu",
        )


def test_cleanup_runs_when_model_has_no_loss() -> None:
    class NoLossModel(ToyVotingModel):
        def forward(self, **kwargs) -> SimpleNamespace:
            self.forward_count += 1
            self.attention(kwargs["input_values"])
            return SimpleNamespace(loss=None)

    model = NoLossModel()
    with pytest.raises(ValueError, match="tensor loss"):
        compute_gradient_votes(
            model,
            {"input_values": torch.ones(1, 2), "labels": torch.zeros(1)},
            {"attention.query": [_atom("atom", "attention.query", [1, 0], [1, 0])]},
            _config(),
            client_id=1,
            task_name="task",
            device="cpu",
        )
    assert len(model.attention.query._forward_hooks) == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_rejects_gate_without_gradient() -> None:
    class DisconnectedModel(ToyVotingModel):
        def forward(self, input_values: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
            self.forward_count += 1
            ignored = self.attention(input_values)
            loss = torch.zeros((), device=input_values.device, requires_grad=True)
            return SimpleNamespace(loss=loss + ignored.detach().sum() * 0)

    model = DisconnectedModel()
    with pytest.raises(ValueError, match="did not receive a gradient"):
        compute_gradient_votes(
            model,
            {"input_values": torch.ones(1, 2), "labels": torch.zeros(1)},
            {"attention.query": [_atom("atom", "attention.query", [1, 0], [1, 0])]},
            _config(),
            client_id=1,
            task_name="task",
            device="cpu",
        )
