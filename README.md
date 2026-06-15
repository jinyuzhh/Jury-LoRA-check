# Jury-LoRA Federated Fine-Tuning

This repository contains the initial project scaffold for Jury-LoRA federated
fine-tuning with PyTorch and Hugging Face libraries. Training logic is not yet
implemented; the current entry point loads configuration, initializes random
seeds, and prepares the output directory.

## Requirements

- Python 3.10

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Configuration

The default experiment configuration is located at
`configs/roberta_glue_4task_jury_global.yaml`. It defines the model, GLUE tasks,
LoRA settings, federated schedule, Jury settings, and output paths.

## Run

Run the Python module directly:

```bash
python -m src.federated.run_jury_lora --config configs/roberta_glue_4task_jury_global.yaml
```

Alternatively, on a POSIX-compatible shell:

```bash
./scripts/run_jury_lora_global.sh
```
