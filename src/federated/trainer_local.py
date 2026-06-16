"""Single-client local training diagnostics without federated aggregation."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.models.lora_utils import (
    assert_no_trainable_heads,
    get_lora_state_dict,
    load_lora_state_dict,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


LOCAL_TRAIN_LOG_COLUMNS = [
    "client_id",
    "task_name",
    "num_train_samples",
    "train_loss",
]
LOCAL_METRIC_COLUMNS = [
    "client_id",
    "task_name",
    "accuracy",
    "f1",
    "eval_loss",
]


def _default_model_builder(task_name: str, config: Mapping[str, Any]) -> nn.Module:
    from src.models.roberta_lora import build_lora_model

    return build_lora_model(task_name, dict(config))


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    """Load optional runtime dependencies only for a real local run."""
    from transformers import AutoTokenizer

    from src.data.glue_clients import build_glue_clients
    from src.federated.client import FederatedClient

    return AutoTokenizer, build_glue_clients, FederatedClient


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    if "labels" not in moved and "label" in moved:
        moved["labels"] = moved.pop("label")
    return moved


def evaluate_single_client_lora_state(
    lora_state: Mapping[str, torch.Tensor],
    eval_loader: Any,
    client: Any,
    config: Mapping[str, Any],
    device: torch.device | str,
    model_builder: Callable[[str, Mapping[str, Any]], nn.Module] | None = None,
) -> dict[str, Any]:
    """Evaluate one client's private head with its locally trained LoRA state."""
    build_model = model_builder or _default_model_builder
    evaluation_device = torch.device(device)
    model = build_model(client.task_name, config)
    load_lora_state_dict(model, lora_state, strict=False)
    client.load_private_classifier(model)
    assert_no_trainable_heads(model)
    model.to(evaluation_device)
    model.eval()

    predictions: list[int] = []
    references: list[int] = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in eval_loader:
            moved_batch = _move_batch_to_device(batch, evaluation_device)
            labels = moved_batch.get("labels")
            if labels is None:
                raise ValueError(
                    f"Evaluation batch for {client.task_name!r} does not contain "
                    "labels."
                )
            outputs = model(**moved_batch)
            loss = getattr(outputs, "loss", None)
            logits = getattr(outputs, "logits", None)
            if not torch.is_tensor(loss) or not torch.is_tensor(logits):
                raise ValueError(
                    f"Evaluation model for {client.task_name!r} must return loss "
                    "and logits."
                )

            batch_size = labels.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_examples += batch_size
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels.cpu().tolist())

    del model
    if total_examples == 0:
        raise ValueError(f"Evaluation loader for {client.task_name!r} is empty.")

    f1_value = None
    if client.task_name in {"mrpc", "qqp"}:
        f1_value = f1_score(
            references,
            predictions,
            average="binary",
            zero_division=0,
        )
    return {
        "client_id": client.client_id,
        "task_name": client.task_name,
        "accuracy": accuracy_score(references, predictions),
        "f1": f1_value,
        "eval_loss": total_loss / total_examples,
    }


def _select_client_data(
    client_data: list[dict[str, Any]],
    client_id: int,
) -> dict[str, Any]:
    for client in client_data:
        if client.get("client_id") == client_id:
            return client
    available_ids = ", ".join(str(client.get("client_id")) for client in client_data)
    raise ValueError(
        f"Local diagnostic client_id {client_id} was not found. "
        f"Available client ids: {available_ids}."
    )


def run_local_client_training(
    config_path: str | Path,
    client_id: int,
    model_builder: Callable[[str, Mapping[str, Any]], nn.Module] | None = None,
) -> dict[str, Any]:
    """Train and evaluate one client locally without server aggregation."""
    if isinstance(client_id, bool) or not isinstance(client_id, int):
        raise ValueError("local client_id must be an integer.")

    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    AutoTokenizer, build_glue_clients, FederatedClient = _load_runtime_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    client_data, eval_loaders = build_glue_clients(config, tokenizer)
    selected_data = _select_client_data(client_data, client_id)
    selected_client = FederatedClient(
        client_id=selected_data["client_id"],
        task_name=selected_data["task_name"],
        train_dataloader=selected_data["train_dataloader"],
        config=config,
        device=device,
    )
    if selected_client.task_name not in eval_loaders:
        raise ValueError(
            f"Missing evaluation loader for task {selected_client.task_name!r}."
        )

    build_model = model_builder or _default_model_builder
    initial_model = build_model(selected_client.task_name, config)
    initial_lora_state = get_lora_state_dict(initial_model)
    del initial_model

    print(
        "Running local single-client diagnostic: "
        f"client_id={selected_client.client_id}, "
        f"task={selected_client.task_name}, "
        f"train_samples={selected_client.num_train_samples}"
    )
    update = selected_client.train(initial_lora_state)
    metrics = evaluate_single_client_lora_state(
        update["lora_state_dict"],
        eval_loaders[selected_client.task_name],
        selected_client,
        config,
        device,
        model_builder=build_model,
    )

    train_row = {
        "client_id": update["client_id"],
        "task_name": update["task_name"],
        "num_train_samples": update["num_train_samples"],
        "train_loss": update["average_train_loss"],
    }
    pd.DataFrame([train_row], columns=LOCAL_TRAIN_LOG_COLUMNS).to_csv(
        output_dir / "local_client_train_log.csv",
        index=False,
    )
    pd.DataFrame([metrics], columns=LOCAL_METRIC_COLUMNS).to_csv(
        output_dir / "local_client_metrics.csv",
        index=False,
    )
    print(
        f"Local client {metrics['client_id']} eval: "
        f"task={metrics['task_name']}, accuracy={metrics['accuracy']:.6f}, "
        f"f1={metrics['f1']}, loss={metrics['eval_loss']:.6f}"
    )
    return {"train_log": train_row, "metrics": metrics}
