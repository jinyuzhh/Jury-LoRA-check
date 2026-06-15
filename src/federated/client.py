"""Federated client implementation for local LoRA training."""

from collections.abc import Mapping
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.lora_utils import get_lora_state_dict, load_lora_state_dict
from src.models.roberta_lora import build_lora_model


class FederatedClient:
    """Train a task-specific model and upload only its LoRA A/B parameters."""

    def __init__(
        self,
        client_id: int,
        task_name: str,
        train_dataloader: DataLoader,
        config: dict[str, Any],
        device: torch.device | str | None = None,
    ) -> None:
        self.client_id = client_id
        self.task_name = task_name
        self.train_dataloader = train_dataloader
        self.config = config
        self._classifier_state_dict: dict[str, torch.Tensor] | None = None
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @property
    def num_train_samples(self) -> int:
        """Return the number of examples assigned to this client."""
        return len(self.train_dataloader.dataset)

    @property
    def has_private_classifier(self) -> bool:
        """Return whether this client has trained classifier state to restore."""
        return self._classifier_state_dict is not None

    @staticmethod
    def _is_classifier_parameter(name: str) -> bool:
        return "classifier" in name.split(".")

    def load_private_classifier(self, model: torch.nn.Module) -> None:
        """Restore this client's private classifier without touching LoRA state."""
        if self._classifier_state_dict is None:
            return
        model.load_state_dict(self._classifier_state_dict, strict=False)

    def save_private_classifier(self, model: torch.nn.Module) -> None:
        """Persist this client's classifier on CPU for future rounds."""
        classifier_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
            if self._is_classifier_parameter(name)
        }
        if not classifier_state:
            raise ValueError(
                f"Client {self.client_id} model does not expose classifier parameters."
            )
        self._classifier_state_dict = classifier_state

    def train(
        self,
        global_lora_state: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run local training from the supplied global LoRA state."""
        num_train_samples = self.num_train_samples
        if num_train_samples == 0:
            return {
                "client_id": self.client_id,
                "task_name": self.task_name,
                "num_train_samples": 0,
                "lora_state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in global_lora_state.items()
                },
                "average_train_loss": 0.0,
            }

        model = build_lora_model(self.task_name, self.config)
        load_lora_state_dict(model, global_lora_state, strict=False)
        self.load_private_classifier(model)
        model.to(self.device)
        model.train()

        federated_config = self.config["federated"]
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=federated_config["lr"],
            weight_decay=federated_config["weight_decay"],
        )

        total_loss = 0.0
        total_examples = 0
        for _ in range(federated_config["local_epochs"]):
            for batch in self.train_dataloader:
                batch = {
                    name: tensor.to(self.device) if torch.is_tensor(tensor) else tensor
                    for name, tensor in batch.items()
                }
                if "labels" not in batch and "label" in batch:
                    batch["labels"] = batch.pop("label")
                labels = batch.get("labels")
                if labels is None:
                    raise ValueError(
                        f"Client {self.client_id} received a batch without labels."
                    )
                batch_size = labels.shape[0]

                optimizer.zero_grad()
                outputs = model(**batch)
                if outputs.loss is None:
                    raise ValueError(
                        f"Client {self.client_id} model output did not contain a loss."
                    )
                outputs.loss.backward()
                optimizer.step()

                total_loss += outputs.loss.detach().item() * batch_size
                total_examples += batch_size

        average_train_loss = total_loss / total_examples if total_examples else 0.0
        lora_state_dict = get_lora_state_dict(model)
        self.save_private_classifier(model)
        return {
            "client_id": self.client_id,
            "task_name": self.task_name,
            "num_train_samples": num_train_samples,
            "lora_state_dict": lora_state_dict,
            "average_train_loss": average_train_loss,
        }
