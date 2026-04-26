# IntentDrift

IntentDrift is an experimental repository for detecting prompt-injection risk in tool-using agents from model activations. The project moves away from a whole-context last-token classifier and instead implements **task-drift detection**: after a tool response is appended to an agent trajectory, the detector compares the model state against the original user task and the pre-tool history before the agent is allowed to take the next action.

The main detector is designed for the AgentDojo-style execution point:

```python
ToolsExecutionLoop([ToolsExecutor(), SecMCPTaskDriftDetector(...), llm])
```

This lets the system score a tool observation before the next LLM decision, while still preserving the whole-context RF/logistic baselines for comparison.

## What Is Implemented

- Unified dataset schema for agent-style trajectories.
- Loaders for AgentDojo, ASB, InjecAgent, AgentHarm, and AgentTraj-L.
- Train/validation/test split generation with AgentDojo held-out grouping.
- Hugging Face model loading, chat rendering, system-role compatibility handling, and head-tail truncation.
- Whole-context last-token activation extraction for legacy baselines.
- Task-drift activation extraction at every `role == "tool"` step.
- Drift features based on global drift, incremental drift, history progress, relative norm, per-layer L2 norms, per-layer cosine similarity, and benign anchor distances.
- `HistGradientBoostingClassifier` task-drift detector with validation threshold selection.
- Trajectory-level score aggregation with `max` and `top2_mean`.
- Whole-context `rf_anchor` and `logistic_diff` baseline training.
- AgentDojo pipeline element for runtime task-drift detection.

The full AgentDojo benchmark runner, ASB evaluation runner, text-defense baselines, and summary-report scripts are still planned work. See `PLAN.md` for the current phase-by-phase status.

## Repository Layout

```text
configs/                    Model, data, training, and eval configuration
scripts/                    Dataset unification, activation extraction, detector training
src/secmcp/                 Main Python package
  activations/              Whole-context and task-drift activation datasets
  data/                     Unified schema, loaders, split logic
  detectors/                Drift detector and legacy baseline classifiers
  integrations/             AgentDojo runtime integration
  models/                   HF model loading, hooks, truncation
tests/                      Unit tests
outputs/                    Generated artifacts, ignored except .gitkeep files
data/                       Local datasets, ignored by git
```

## Requirements

Python 3.10 or newer is required. GPU execution is expected for real activation extraction.

Install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Or install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The model paths in `configs/models.yaml` point to local Hugging Face checkpoint locations such as `/hub/huggingface/models/...`. Update those paths if your checkpoints live elsewhere.

## Data Setup

Datasets are not committed to this repository. Place them under `data/` with this expected structure:

```text
data/
  AgentDojo/
  ASB/
  InjecAgent/
  AgentHarm/
  AgentTraj-L/
```

Generated outputs are also ignored by git. The important output roots are:

```text
outputs/splits/
outputs/activations/
outputs/drift_activations/
outputs/detectors/
outputs/eval/
```

## Reproduction

The current mainline experiment is task-drift detection. Start by generating fresh splits:

```bash
conda run -n taskdrift python scripts/01_unify_datasets.py
```

Before extracting task-drift activations, it is worth confirming that the splits contain tool steps. A zero step count means the split file came from an old schema or the source loader did not preserve tool messages.

### Extract Task-Drift Activations

For Mistral:

```bash
env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl-secmcp-mistral \
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split train --mode task_drift --shard-size 100 --log-every 1

env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl-secmcp-mistral \
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split val --mode task_drift --shard-size 100 --log-every 1

env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl-secmcp-mistral \
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split test --mode task_drift --shard-size 100 --log-every 1
```

For Gemma:

```bash
env CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl-secmcp-gemma \
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model gemma2_9b --split train --mode task_drift --shard-size 100 --log-every 1

env CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl-secmcp-gemma \
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model gemma2_9b --split val --mode task_drift --shard-size 100 --log-every 1

env CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl-secmcp-gemma \
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model gemma2_9b --split test --mode task_drift --shard-size 100 --log-every 1
```

These commands write:

```text
outputs/drift_activations/{model}/{split}/
  drift_00000.pt
  labels_00000.pt
  meta_00000.jsonl
```

### Train the Task-Drift Detector

```bash
conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift
```

For Gemma:

```bash
conda run -n taskdrift python scripts/03_train_detector.py \
  --model gemma2_9b --detector task_drift
```

The trained detector and metrics are written under:

```text
outputs/detectors/{model}/task_drift_best.pkl
outputs/detectors/{model}/task_drift_metrics.json
```

### Legacy Whole-Context Baselines

Whole-context activations are not the main method, but remain available for ablations:

```bash
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split train --mode whole_context

conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector rf_anchor

conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector logistic_diff
```

## Runtime Integration

After training a task-drift detector, it can be inserted into an AgentDojo pipeline:

```python
from secmcp.integrations.agentdojo_drift_detector import SecMCPTaskDriftDetector

detector = SecMCPTaskDriftDetector(
    detector_path="outputs/detectors/mistral_7b_v03/task_drift_best.pkl",
    model_name="mistral_7b_v03",
    raise_on_detection=True,
)
```

The detector records per-step scores in `extra_args["secmcp_task_drift_scores"]`, the latest step score in `extra_args["secmcp_task_drift_last_score"]`, and the aggregated trajectory score in `extra_args["secmcp_task_drift_trajectory_score"]`.

## Tests

Run the unit test suite with:

```bash
conda run -n taskdrift python -m pytest -q
```

The latest local verification before adding this README was:

```text
99 passed, 5 skipped
```

## Notes

- Do not commit `data/`, activation shards, detector pickles, logs, or split JSON files.
- If `outputs/drift_activations/` is empty, the task-drift detector cannot be trained yet.
- If existing split files produce zero tool steps, regenerate splits with the current loaders before activation extraction.
- Changing `configs/models.yaml`, layer selections, truncation settings, or drift feature configuration can invalidate previously trained detectors.
