#!/usr/bin/env bash
set -euo pipefail

python -m src.federated.run_jury_lora --config configs/roberta_glue_4task_jury_global.yaml
