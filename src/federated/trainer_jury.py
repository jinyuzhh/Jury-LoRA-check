"""Training orchestration for global-only Jury-LoRA."""

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from src.jury.atom_extractor import extract_atoms_from_round
from src.jury.atom_selector import (
    select_topk_atoms_by_sigma,
    selected_atoms_to_metadata,
)
from src.jury.gradient_voting import compute_gradient_votes
from src.jury.server_jury import (
    apply_global_memory_to_model,
    update_global_memory,
)
from src.models.lora_utils import (
    assert_head_state_unchanged,
    assert_no_trainable_heads,
    extract_lora_A_B,
    freeze_head_parameters,
    get_head_state_dict,
    get_lora_state_dict,
    print_trainable_parameter_names,
    trainable_adapter_parameters,
)
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
SELECTED_ATOM_COLUMNS = [
    "round_idx",
    "atom_id",
    "layer_name",
    "client_id",
    "task_name",
    "rank_id",
    "sigma",
    "source_lora_rank",
    "selection_score",
]
VOTE_RECORD_COLUMNS = [
    "round_idx",
    "client_id",
    "task_name",
    "atom_id",
    "layer_name",
    "score",
    "vote",
]
VOTE_STATS_COLUMNS = [
    "round_idx",
    "atom_id",
    "layer_name",
    "p_accept",
    "p_reject",
    "p_abstain",
    "accept_count",
    "reject_count",
    "abstain_count",
    "accepted_to_memory",
]
MEMORY_STATS_COLUMNS = [
    "round_idx",
    "layer_name",
    "memory_size",
    "added_atoms",
    "pruned_atoms",
    "mean_lambda_g",
    "max_lambda_g",
]

OUTPUT_SCHEMAS = {
    "round_metrics.csv": ROUND_METRIC_COLUMNS,
    "client_train_logs.csv": CLIENT_LOG_COLUMNS,
    "selected_atoms.csv": SELECTED_ATOM_COLUMNS,
    "vote_records.csv": VOTE_RECORD_COLUMNS,
    "vote_stats.csv": VOTE_STATS_COLUMNS,
    "memory_stats.csv": MEMORY_STATS_COLUMNS,
}


def _default_model_builder(task_name: str, config: Mapping[str, Any]) -> nn.Module:
    from src.models.roberta_lora import build_lora_model

    return build_lora_model(task_name, dict(config))


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    """Load optional training dependencies only for a real Jury run."""
    from transformers import AutoTokenizer

    from src.data.glue_clients import build_glue_clients
    from src.federated.client import FederatedClient

    return AutoTokenizer, build_glue_clients, FederatedClient


def initialize_output_files(output_dir: str | Path) -> dict[str, Path]:
    """Overwrite all Jury CSV outputs with stable headers."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, columns in OUTPUT_SCHEMAS.items():
        path = directory / filename
        with path.open("w", newline="", encoding="utf-8") as output_file:
            csv.DictWriter(output_file, fieldnames=columns).writeheader()
        paths[filename] = path
    return paths


def append_csv_rows(
    output_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    """Append rows using exactly the requested output columns."""
    if not rows:
        return
    with Path(output_path).open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="ignore")
        writer.writerows(
            {column: row.get(column) for column in columns} for row in rows
        )


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


def train_client_with_memory(
    client: Any,
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    device: torch.device | str,
    model_builder: Callable[[str, Mapping[str, Any]], nn.Module] | None = None,
    memory_applier: Callable[[nn.Module, Mapping[str, Any]], None] = (
        apply_global_memory_to_model
    ),
) -> dict[str, Any]:
    """Train one client from a fresh W0 + global-memory model."""
    build_model = model_builder or _default_model_builder
    training_device = torch.device(device)
    model = build_model(client.task_name, config)
    freeze_head_parameters(model)
    memory_applier(model, memory)
    client.load_private_classifier(model)
    assert_no_trainable_heads(model)
    model.to(training_device)

    num_train_samples = client.num_train_samples
    if num_train_samples == 0:
        lora_state = get_lora_state_dict(model)
        del model
        return {
            "client_id": client.client_id,
            "task_name": client.task_name,
            "num_train_samples": 0,
            "lora_state_dict": lora_state,
            "average_train_loss": 0.0,
        }

    model.train()
    print_trainable_parameter_names(model)
    classifier_state_before_training = get_head_state_dict(model)
    federated_config = config["federated"]
    optimizer = AdamW(
        trainable_adapter_parameters(model),
        lr=federated_config["lr"],
        weight_decay=federated_config["weight_decay"],
    )

    total_loss = 0.0
    total_examples = 0
    for _ in range(federated_config["local_epochs"]):
        for batch in client.train_dataloader:
            moved_batch = _move_batch_to_device(batch, training_device)
            labels = moved_batch.get("labels")
            if labels is None:
                raise ValueError(
                    f"Client {client.client_id} received a batch without labels."
                )
            optimizer.zero_grad()
            outputs = model(**moved_batch)
            loss = getattr(outputs, "loss", None)
            if not torch.is_tensor(loss):
                raise ValueError(
                    f"Client {client.client_id} model output did not contain a loss."
                )
            loss.backward()
            optimizer.step()
            batch_size = labels.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_examples += batch_size

    assert_no_trainable_heads(model)
    assert_head_state_unchanged(classifier_state_before_training, model)
    lora_state = get_lora_state_dict(model)
    client.save_private_classifier(model)
    average_loss = total_loss / total_examples if total_examples else 0.0
    del model
    return {
        "client_id": client.client_id,
        "task_name": client.task_name,
        "num_train_samples": num_train_samples,
        "lora_state_dict": lora_state,
        "average_train_loss": average_loss,
    }


def get_voting_batch(
    train_dataloader: Any,
    vote_batch_size: int,
) -> dict[str, Any] | None:
    """Return the first local batch truncated to the configured vote size."""
    if (
        isinstance(vote_batch_size, bool)
        or not isinstance(vote_batch_size, int)
        or vote_batch_size <= 0
    ):
        raise ValueError("jury.vote_batch_size must be a positive integer.")
    try:
        batch = next(iter(train_dataloader))
    except StopIteration:
        return None
    if not isinstance(batch, Mapping):
        raise ValueError("Voting dataloader batches must be mappings.")
    return {
        name: (
            value[:vote_batch_size]
            if torch.is_tensor(value) and value.ndim > 0
            else value
        )
        for name, value in batch.items()
    }


def vote_for_selected_atoms(
    client: Any,
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_atoms_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    device: torch.device | str,
    model_builder: Callable[[str, Mapping[str, Any]], nn.Module] | None = None,
    memory_applier: Callable[[nn.Module, Mapping[str, Any]], None] = (
        apply_global_memory_to_model
    ),
    vote_computer: Callable[..., list[dict[str, Any]]] = compute_gradient_votes,
) -> list[dict[str, Any]]:
    """Vote with a fresh W0 + M model and the client's trained classifier."""
    if client.num_train_samples == 0 or not any(selected_atoms_by_layer.values()):
        return []
    vote_batch = get_voting_batch(
        client.train_dataloader,
        config["jury"]["vote_batch_size"],
    )
    if vote_batch is None:
        return []

    build_model = model_builder or _default_model_builder
    model = build_model(client.task_name, config)
    freeze_head_parameters(model)
    memory_applier(model, memory)
    client.load_private_classifier(model)
    assert_no_trainable_heads(model)
    model.to(torch.device(device))
    records = vote_computer(
        model,
        vote_batch,
        selected_atoms_by_layer,
        config,
        client.client_id,
        client.task_name,
        device,
    )
    assert_no_trainable_heads(model)
    del model
    return records


def evaluate_global_memory(
    memory: Mapping[str, Sequence[Mapping[str, Any]]],
    eval_loaders: Mapping[str, Any],
    clients: Sequence[Any],
    config: Mapping[str, Any],
    device: torch.device | str,
    model_builder: Callable[[str, Mapping[str, Any]], nn.Module] | None = None,
    memory_applier: Callable[[nn.Module, Mapping[str, Any]], None] = (
        apply_global_memory_to_model
    ),
) -> list[dict[str, Any]]:
    """Evaluate W0 + M with each client's trained private classifier."""
    from sklearn.metrics import accuracy_score, f1_score

    build_model = model_builder or _default_model_builder
    evaluation_device = torch.device(device)
    metrics: list[dict[str, Any]] = []
    for task_name in config["data"]["tasks"]:
        if task_name not in eval_loaders:
            raise ValueError(f"Missing evaluation loader for task {task_name!r}.")
        task_clients = [
            client
            for client in clients
            if client.task_name == task_name and client.num_train_samples > 0
        ]
        if not task_clients:
            raise ValueError(
                f"Task {task_name!r} has no nonempty clients to evaluate."
            )

        weighted_accuracy = 0.0
        weighted_f1 = 0.0
        weighted_eval_loss = 0.0
        total_client_samples = 0
        for client in task_clients:
            if not client.has_private_classifier:
                raise ValueError(
                    f"Client {client.client_id} has no persisted classifier state."
                )
            model = build_model(task_name, config)
            freeze_head_parameters(model)
            memory_applier(model, memory)
            client.load_private_classifier(model)
            assert_no_trainable_heads(model)
            model.to(evaluation_device)
            model.eval()

            predictions: list[int] = []
            references: list[int] = []
            total_loss = 0.0
            total_examples = 0
            with torch.no_grad():
                for batch in eval_loaders[task_name]:
                    moved_batch = _move_batch_to_device(batch, evaluation_device)
                    labels = moved_batch.get("labels")
                    if labels is None:
                        raise ValueError(
                            f"Evaluation batch for {task_name!r} does not contain "
                            "labels."
                        )
                    outputs = model(**moved_batch)
                    loss = getattr(outputs, "loss", None)
                    logits = getattr(outputs, "logits", None)
                    if not torch.is_tensor(loss) or not torch.is_tensor(logits):
                        raise ValueError(
                            f"Evaluation model for {task_name!r} must return loss "
                            "and logits."
                        )
                    batch_size = labels.shape[0]
                    total_loss += loss.detach().item() * batch_size
                    total_examples += batch_size
                    predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                    references.extend(labels.cpu().tolist())
            if total_examples == 0:
                raise ValueError(f"Evaluation loader for {task_name!r} is empty.")

            client_samples = client.num_train_samples
            weighted_accuracy += (
                accuracy_score(references, predictions) * client_samples
            )
            weighted_eval_loss += total_loss / total_examples * client_samples
            if task_name in {"mrpc", "qqp"}:
                weighted_f1 += (
                    f1_score(
                        references,
                        predictions,
                        average="binary",
                        zero_division=0,
                    )
                    * client_samples
                )
            total_client_samples += client_samples
            del model

        metrics.append(
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
    return metrics


def build_vote_stats_rows(
    vote_stats: Sequence[Mapping[str, Any]],
    updated_memory: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Add final memory-admission status to per-atom vote statistics."""
    surviving_sources = {
        atom["source_atom_id"]
        for layer_atoms in updated_memory.values()
        for atom in layer_atoms
    }
    return [
        {
            **dict(stats),
            "accepted_to_memory": stats["atom_id"] in surviving_sources,
        }
        for stats in vote_stats
    ]


def build_memory_stats_rows(
    previous_memory: Mapping[str, Sequence[Mapping[str, Any]]],
    updated_memory: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_atoms_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
    vote_stats: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    round_idx: int,
) -> list[dict[str, Any]]:
    """Summarize additions, pruning, and strengths for each observed layer."""
    stats_by_id = {stats["atom_id"]: stats for stats in vote_stats}
    theta_high = float(config["jury"]["theta_high"])
    eta_memory = float(config["jury"]["eta_memory"])
    eligible_by_layer: dict[str, int] = {}
    for layer_name, atoms in selected_atoms_by_layer.items():
        eligible = 0
        for atom in atoms:
            stats = stats_by_id[atom["atom_id"]]
            margin = stats["p_accept"] - stats["p_reject"]
            sigma = float(atom["sigma"].detach().cpu().item()) if torch.is_tensor(
                atom["sigma"]
            ) else float(atom["sigma"])
            if stats["p_accept"] >= theta_high and eta_memory * sigma * margin > 0:
                eligible += 1
        eligible_by_layer[layer_name] = eligible

    layers = list(previous_memory)
    for source in (selected_atoms_by_layer, updated_memory):
        for layer_name in source:
            if layer_name not in layers:
                layers.append(layer_name)

    rows: list[dict[str, Any]] = []
    for layer_name in layers:
        final_atoms = list(updated_memory.get(layer_name, []))
        lambdas = [float(atom["lambda_g"]) for atom in final_atoms]
        added_atoms = sum(
            atom["round_idx"] == round_idx for atom in final_atoms
        )
        candidate_size = len(
            previous_memory.get(layer_name, [])
        ) + eligible_by_layer.get(layer_name, 0)
        rows.append(
            {
                "round_idx": round_idx,
                "layer_name": layer_name,
                "memory_size": len(final_atoms),
                "added_atoms": added_atoms,
                "pruned_atoms": max(candidate_size - len(final_atoms), 0),
                "mean_lambda_g": sum(lambdas) / len(lambdas) if lambdas else 0.0,
                "max_lambda_g": max(lambdas) if lambdas else 0.0,
            }
        )
    return rows


def _selected_metadata_rows(
    selected_atoms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {column: row[column] for column in SELECTED_ATOM_COLUMNS}
        for row in selected_atoms_to_metadata(selected_atoms)
    ]


def _print_round_summary(
    round_idx: int,
    atom_count: int,
    selected_count: int,
    vote_records: Sequence[Mapping[str, Any]],
    memory_stats: Sequence[Mapping[str, Any]],
    task_metrics: Sequence[Mapping[str, Any]],
) -> None:
    vote_counts = {"accept": 0, "reject": 0, "abstain": 0}
    for record in vote_records:
        vote_counts[record["vote"]] += 1
    added_atoms = sum(row["added_atoms"] for row in memory_stats)
    print(
        f"Round {round_idx}: extracted={atom_count}, selected={selected_count}, "
        f"accept={vote_counts['accept']}, reject={vote_counts['reject']}, "
        f"abstain={vote_counts['abstain']}, added_to_memory={added_atoms}"
    )
    for row in memory_stats:
        print(f"  Memory {row['layer_name']}: size={row['memory_size']}")
    for row in task_metrics:
        print(
            f"  Eval {row['task_name']}: accuracy={row['accuracy']:.6f}, "
            f"f1={row['f1']}, loss={row['eval_loss']:.6f}"
        )


def run_jury_global_training(config_path: str | Path) -> None:
    """Run all rounds of global-only Jury-LoRA training."""
    config = load_config(config_path)
    jury_config = config.get("jury")
    if not isinstance(jury_config, Mapping) or jury_config.get("enabled") is not True:
        raise ValueError("Jury trainer requires jury.enabled: true.")
    if jury_config.get("mode") != "global_only":
        raise ValueError("Jury trainer currently supports only jury.mode: global_only.")
    if jury_config.get("top_k_mode", "sigma") != "sigma":
        raise ValueError("Global Jury trainer supports only jury.top_k_mode: sigma.")
    vote_batch_size = jury_config.get("vote_batch_size")
    if (
        isinstance(vote_batch_size, bool)
        or not isinstance(vote_batch_size, int)
        or vote_batch_size <= 0
    ):
        raise ValueError("jury.vote_batch_size must be a positive integer.")

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_paths = initialize_output_files(config["output"]["dir"])
    AutoTokenizer, build_glue_clients, FederatedClient = _load_runtime_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    client_data, eval_loaders = build_glue_clients(config, tokenizer)
    clients = [
        FederatedClient(
            client_id=client_data_row["client_id"],
            task_name=client_data_row["task_name"],
            train_dataloader=client_data_row["train_dataloader"],
            config=config,
            device=device,
        )
        for client_data_row in client_data
    ]

    memory: dict[str, list[dict[str, Any]]] = {}
    for round_idx in range(1, config["federated"]["rounds"] + 1):
        print(f"Starting Jury-LoRA round {round_idx}")
        client_uploads: list[dict[str, Any]] = []
        client_log_rows: list[dict[str, Any]] = []
        for client in clients:
            upload = train_client_with_memory(client, memory, config, device)
            client_uploads.append(upload)
            client_log_rows.append(
                {
                    "round": round_idx,
                    "client_id": upload["client_id"],
                    "task_name": upload["task_name"],
                    "num_train_samples": upload["num_train_samples"],
                    "train_loss": upload["average_train_loss"],
                }
            )

        extraction_updates = [
            {
                "round_idx": round_idx,
                "client_id": upload["client_id"],
                "task_name": upload["task_name"],
                "grouped_lora_ab": extract_lora_A_B(upload["lora_state_dict"]),
            }
            for upload in client_uploads
        ]
        atoms = extract_atoms_from_round(extraction_updates, config, device)
        selected_atoms = select_topk_atoms_by_sigma(atoms, config)

        vote_records: list[dict[str, Any]] = []
        for client in clients:
            client_votes = vote_for_selected_atoms(
                client,
                memory,
                selected_atoms,
                config,
                device,
            )
            vote_records.extend(
                {"round_idx": round_idx, **record} for record in client_votes
            )

        previous_memory = memory
        memory, vote_stats = update_global_memory(
            previous_memory,
            selected_atoms,
            vote_records,
            config,
            round_idx,
        )
        vote_stats_rows = build_vote_stats_rows(vote_stats, memory)
        memory_stats_rows = build_memory_stats_rows(
            previous_memory,
            memory,
            selected_atoms,
            vote_stats,
            config,
            round_idx,
        )
        task_metrics = evaluate_global_memory(
            memory,
            eval_loaders,
            clients,
            config,
            device,
        )
        round_metric_rows = [
            {"round": round_idx, **task_metric} for task_metric in task_metrics
        ]

        append_csv_rows(
            output_paths["client_train_logs.csv"],
            client_log_rows,
            CLIENT_LOG_COLUMNS,
        )
        append_csv_rows(
            output_paths["selected_atoms.csv"],
            _selected_metadata_rows(selected_atoms),
            SELECTED_ATOM_COLUMNS,
        )
        append_csv_rows(
            output_paths["vote_records.csv"],
            vote_records,
            VOTE_RECORD_COLUMNS,
        )
        append_csv_rows(
            output_paths["vote_stats.csv"],
            vote_stats_rows,
            VOTE_STATS_COLUMNS,
        )
        append_csv_rows(
            output_paths["memory_stats.csv"],
            memory_stats_rows,
            MEMORY_STATS_COLUMNS,
        )
        append_csv_rows(
            output_paths["round_metrics.csv"],
            round_metric_rows,
            ROUND_METRIC_COLUMNS,
        )
        _print_round_summary(
            round_idx,
            len(atoms),
            sum(len(layer_atoms) for layer_atoms in selected_atoms.values()),
            vote_records,
            memory_stats_rows,
            task_metrics,
        )
