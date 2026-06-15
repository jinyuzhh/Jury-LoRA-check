"""RoBERTa sequence-classification models with LoRA adapters."""

from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification


_SUPPORTED_TASKS = {"sst2", "qnli", "mrpc", "qqp"}


def build_lora_model(task_name: str, config: dict[str, Any]) -> PeftModel:
    """Build a binary RoBERTa classifier with PEFT LoRA adapters."""
    if task_name not in _SUPPORTED_TASKS:
        supported = ", ".join(sorted(_SUPPORTED_TASKS))
        raise ValueError(
            f"Unsupported GLUE task {task_name!r}. Supported tasks: {supported}."
        )

    model_config = config["model"]
    lora_config = config["lora"]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        num_labels=2,
    )
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
    )
    return get_peft_model(model, peft_config)
