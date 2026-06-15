"""GLUE dataset preparation for fixed-task federated clients."""

from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


_TASK_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "sst2": ("sentence",),
    "qnli": ("question", "sentence"),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
}

_TASK_CLIENT_START: dict[str, int] = {
    "sst2": 0,
    "qnli": 4,
    "mrpc": 8,
    "qqp": 12,
}


def get_glue_text_fields(task_name: str) -> tuple[str, ...]:
    """Return the text column names used by a supported GLUE task."""
    try:
        return _TASK_TEXT_FIELDS[task_name]
    except KeyError as error:
        supported = ", ".join(_TASK_TEXT_FIELDS)
        raise ValueError(
            f"Unsupported GLUE task {task_name!r}. Supported tasks: {supported}."
        ) from error


def tokenize_glue_dataset(
    dataset: Dataset,
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize a GLUE dataset while preserving its label column."""
    text_fields = get_glue_text_fields(task_name)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [batch[field] for field in text_fields]
        return tokenizer(*texts, truncation=True, max_length=max_length)

    columns_to_remove = [
        column for column in dataset.column_names if column != "label"
    ]
    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
    )


def _get_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Configuration value {key!r} must be a positive integer.")
    return value


def _validate_data_config(config: dict[str, Any]) -> None:
    tasks = config.get("tasks")
    expected_tasks = set(_TASK_TEXT_FIELDS)
    if (
        not isinstance(tasks, list)
        or len(tasks) != len(expected_tasks)
        or set(tasks) != expected_tasks
    ):
        raise ValueError(
            "data.tasks must contain exactly: sst2, qnli, mrpc, qqp."
        )

    if config.get("num_clients") != 16:
        raise ValueError("data.num_clients must be 16 for the fixed GLUE layout.")
    if config.get("clients_per_task") != 4:
        raise ValueError(
            "data.clients_per_task must be 4 for the fixed GLUE layout."
        )


def build_glue_clients(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[dict[str, Any]], dict[str, DataLoader]]:
    """Build the fixed 16-client GLUE training and evaluation loaders."""
    data_config = config.get("data")
    federated_config = config.get("federated")
    if not isinstance(data_config, dict):
        raise ValueError("Configuration must contain a data mapping.")
    if not isinstance(federated_config, dict):
        raise ValueError("Configuration must contain a federated mapping.")

    _validate_data_config(data_config)

    max_train_samples = _get_positive_int(
        data_config, "max_train_samples_per_client"
    )
    max_eval_samples = _get_positive_int(
        data_config, "max_eval_samples_per_task"
    )
    train_batch_size = _get_positive_int(
        federated_config, "train_batch_size"
    )
    eval_batch_size = _get_positive_int(federated_config, "eval_batch_size")
    max_length = data_config.get("max_length", 128)
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length <= 0
    ):
        raise ValueError("Configuration value 'max_length' must be a positive integer.")

    split_seed = data_config.get("split_seed", config.get("seed"))
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError(
            "data.split_seed or the top-level seed must be an integer."
        )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    clients: list[dict[str, Any]] = []
    eval_loaders: dict[str, DataLoader] = {}

    for task_name in _TASK_TEXT_FIELDS:
        task_dataset = load_dataset("nyu-mll/glue", task_name)

        shuffled_train = task_dataset["train"].shuffle(seed=split_seed)
        train_sample_count = min(
            len(shuffled_train),
            data_config["clients_per_task"] * max_train_samples,
        )
        train_pool = shuffled_train.select(range(train_sample_count))
        tokenized_train = tokenize_glue_dataset(
            train_pool, task_name, tokenizer, max_length
        )

        base_size, remainder = divmod(
            train_sample_count, data_config["clients_per_task"]
        )
        shard_start = 0
        for task_client_index in range(data_config["clients_per_task"]):
            shard_size = base_size + (task_client_index < remainder)
            shard_end = shard_start + shard_size
            client_dataset = tokenized_train.select(range(shard_start, shard_end))
            client_id = _TASK_CLIENT_START[task_name] + task_client_index

            generator = torch.Generator()
            generator.manual_seed(split_seed + client_id)
            train_dataloader = DataLoader(
                client_dataset,
                batch_size=train_batch_size,
                shuffle=len(client_dataset) > 0,
                collate_fn=collator,
                generator=generator,
            )
            clients.append(
                {
                    "client_id": client_id,
                    "task_name": task_name,
                    "train_dataset": client_dataset,
                    "train_dataloader": train_dataloader,
                }
            )
            print(
                f"Client {client_id}: task={task_name}, "
                f"train_samples={len(client_dataset)}"
            )
            shard_start = shard_end

        validation = task_dataset["validation"]
        if len(validation) > max_eval_samples:
            validation = validation.shuffle(seed=split_seed).select(
                range(max_eval_samples)
            )
        tokenized_validation = tokenize_glue_dataset(
            validation, task_name, tokenizer, max_length
        )
        eval_loaders[task_name] = DataLoader(
            tokenized_validation,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        print(f"Eval task {task_name}: samples={len(tokenized_validation)}")

    clients.sort(key=lambda client: client["client_id"])
    return clients, eval_loaders
