"""Tests for Jury/FedAvg command dispatch."""

import sys
from types import ModuleType

import pytest

import src.federated.run_jury_lora as run_jury_lora


def _base_config(tmp_path) -> dict:
    return {
        "seed": 1,
        "method": "fedavg_lora",
        "output": {"dir": str(tmp_path)},
        "jury": {"enabled": False, "mode": "global_only"},
    }


def test_jury_enabled_takes_precedence(tmp_path, monkeypatch) -> None:
    config = _base_config(tmp_path)
    config["jury"]["enabled"] = True
    calls: list[object] = []
    monkeypatch.setattr(run_jury_lora, "load_config", lambda path: config)
    monkeypatch.setattr(run_jury_lora, "set_seed", lambda seed: None)
    import src.federated.trainer_jury as trainer_jury
    monkeypatch.setattr(
        trainer_jury,
        "run_jury_global_training",
        lambda path: calls.append(path),
    )

    run_jury_lora.main(["--config", "config.yaml"])

    assert len(calls) == 1
    assert calls[0].name == "config.yaml"


def test_fedavg_fallback_when_jury_disabled(tmp_path, monkeypatch) -> None:
    config = _base_config(tmp_path)
    calls: list[object] = []
    fake_trainer = ModuleType("src.federated.trainer")
    fake_trainer.run_fedavg_training = lambda path: calls.append(path)
    monkeypatch.setitem(sys.modules, "src.federated.trainer", fake_trainer)
    monkeypatch.setattr(run_jury_lora, "load_config", lambda path: config)
    monkeypatch.setattr(run_jury_lora, "set_seed", lambda seed: None)

    run_jury_lora.main(["--config", "config.yaml"])

    assert len(calls) == 1


def test_local_client_mode_takes_precedence_over_jury(tmp_path, monkeypatch) -> None:
    config = _base_config(tmp_path)
    config["jury"]["enabled"] = True
    calls: list[tuple[object, int]] = []
    monkeypatch.setattr(run_jury_lora, "load_config", lambda path: config)
    monkeypatch.setattr(run_jury_lora, "set_seed", lambda seed: None)

    fake_local = ModuleType("src.federated.trainer_local")
    fake_local.run_local_client_training = (
        lambda path, client_id: calls.append((path, client_id))
    )
    monkeypatch.setitem(sys.modules, "src.federated.trainer_local", fake_local)

    run_jury_lora.main(
        ["--config", "config.yaml", "--local-client-id", "0"]
    )

    assert len(calls) == 1
    assert calls[0][0].name == "config.yaml"
    assert calls[0][1] == 0


def test_rejects_unsupported_jury_mode(tmp_path, monkeypatch) -> None:
    config = _base_config(tmp_path)
    config["jury"] = {"enabled": True, "mode": "personalized"}
    monkeypatch.setattr(run_jury_lora, "load_config", lambda path: config)
    monkeypatch.setattr(run_jury_lora, "set_seed", lambda seed: None)

    with pytest.raises(ValueError, match="Unsupported Jury mode"):
        run_jury_lora.main(["--config", "config.yaml"])
