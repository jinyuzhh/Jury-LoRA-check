"""Training orchestration for the FedAvg-LoRA baseline."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.glue_clients import build_glue_clients
from src.federated.client import FederatedClient
from src.federated.server_fedavg import fedavg_lora_states
from src.models.lora_utils import get_lora_state_dict, load_lora_state_dict
from src.models.roberta_lora import build_lora_model
from src.utils.config import load_config
from src.utils.seed import set_seed


ROUND_METRIC_COLUMNS = ["round", "task_name", "accuracy", "f1", "eval_loss"]
CLIENT_LOG_COLUMNS = [
    "round",
    "client_id",
    "task_name",
    "num_train_samples",
    "train_loss",
]


def _move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    moved_batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    if "labels" not in moved_batch and "label" in moved_batch:
        moved_batch["labels"] = moved_batch.pop("label")
    return moved_batch


def evaluate_lora_state(
    global_state: Mapping[str, torch.Tensor],
    eval_loaders: dict[str, DataLoader],
    clients: list[FederatedClient],
    config: dict[str, Any],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    """Evaluate private client classifiers with the shared global LoRA state."""
    evaluation_device = torch.device(device)
    results: list[dict[str, Any]] = []

    for task_name in config["data"]["tasks"]:
        if task_name not in eval_loaders:
            raise ValueError(f"Missing evaluation loader for task {task_name!r}.")

        task_clients = [client for client in clients if client.task_name == task_name]
        if not task_clients:
            raise ValueError(f"No federated clients found for task {task_name!r}.")

        weighted_accuracy = 0.0
        weighted_f1 = 0.0
        weighted_eval_loss = 0.0
        total_client_samples = 0
        for client in task_clients:
            client_samples = client.num_train_samples
            if client_samples == 0:
                continue
            if not client.has_private_classifier:
                raise ValueError(
                    f"Client {client.client_id} has no persisted classifier state."
                )

            model = build_lora_model(task_name, config)
            load_lora_state_dict(model, global_state, strict=False)
            client.load_private_classifier(model)
            model.to(evaluation_device)
            model.eval()

            predictions: list[int] = []
            references: list[int] = []
            total_loss = 0.0
            total_examples = 0
            with torch.no_grad():
                for batch in eval_loaders[task_name]:
                    batch = _move_batch_to_device(batch, evaluation_device)
                    labels = batch.get("labels")
                    if labels is None:
                        raise ValueError(
                            f"Evaluation batch for {task_name!r} does not contain labels."
                        )
                    outputs = model(**batch)
                    if outputs.loss is None:
                        raise ValueError(
                            f"Evaluation model for {task_name!r} did not return a loss."
                        )

                    batch_size = labels.shape[0]
                    total_loss += outputs.loss.detach().item() * batch_size
                    total_examples += batch_size
                    predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                    references.extend(labels.cpu().tolist())

            if total_examples == 0:
                raise ValueError(f"Evaluation loader for {task_name!r} is empty.")

            client_accuracy = accuracy_score(references, predictions)
            client_eval_loss = total_loss / total_examples
            weighted_accuracy += client_accuracy * client_samples
            weighted_eval_loss += client_eval_loss * client_samples
            if task_name in {"mrpc", "qqp"}:
                client_f1 = f1_score(
                    references,
                    predictions,
                    average="binary",
                    zero_division=0,
                )
                weighted_f1 += client_f1 * client_samples
            total_client_samples += client_samples
            del model

        if total_client_samples == 0:
            raise ValueError(f"Task {task_name!r} has no nonempty clients to evaluate.")

        results.append(
            {
                "task_name": task_name,
                "accuracy": weighted_accuracy / total_client_samples,
                "f1": (
                    weighted_f1 / total_client_samples
                    if task_name in {"mrpc", "qqp"}
                    else None
                ),
                "eval_loss": weighted_eval_loss / total_client_samples,
            }
        )

    return results


def run_fedavg_training(config_path: str | Path) -> None:
    """Run all configured rounds of the FedAvg-LoRA baseline."""
    config = load_config(config_path)
    if config.get("method") != "fedavg_lora":
        raise ValueError("FedAvg trainer requires method: fedavg_lora.")

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "round_metrics.csv"
    client_logs_path = output_dir / "client_train_logs.csv"

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    client_data, eval_loaders = build_glue_clients(config, tokenizer)
    clients = [
        FederatedClient(
            client_id=client["client_id"],
            task_name=client["task_name"],
            train_dataloader=client["train_dataloader"],
            config=config,
            device=device,
        )
        for client in client_data
    ]

    initial_model = build_lora_model(config["data"]["tasks"][0], config)
    global_lora_state = get_lora_state_dict(initial_model)
    del initial_model

    pd.DataFrame(columns=ROUND_METRIC_COLUMNS).to_csv(metrics_path, index=False)
    pd.DataFrame(columns=CLIENT_LOG_COLUMNS).to_csv(client_logs_path, index=False)

    for round_number in range(1, config["federated"]["rounds"] + 1):
        print(f"Starting federated round {round_number}")
        client_updates = []
        round_client_log_rows: list[dict[str, Any]] = []
        for client in clients:
            update = client.train(global_lora_state)
            client_updates.append(update)
            round_client_log_rows.append(
                {
                    "round": round_number,
                    "client_id": update["client_id"],
                    "task_name": update["task_name"],
                    "num_train_samples": update["num_train_samples"],
                    "train_loss": update["average_train_loss"],
                }
            )

        global_lora_state = fedavg_lora_states(client_updates)
        task_metrics = evaluate_lora_state(
            global_lora_state,
            eval_loaders,
            clients,
            config,
            device,
        )
        round_metric_rows = [
            {"round": round_number, **task_metrics_row}
            for task_metrics_row in task_metrics
        ]

        pd.DataFrame(round_metric_rows, columns=ROUND_METRIC_COLUMNS).to_csv(
            metrics_path,
            mode="a",
            header=False,
            index=False,
        )
        pd.DataFrame(
            round_client_log_rows,
            columns=CLIENT_LOG_COLUMNS,
        ).to_csv(
            client_logs_path,
            mode="a",
            header=False,
            index=False,
        )
        print(f"Completed federated round {round_number}")
