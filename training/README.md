# Training the bundled task model

`src/llm_preclassifier/data/task_model.bin` is a hashed n-gram logistic regression
(32,768 buckets, int8 weights, ~230 KB). It is trained here, and at runtime it runs
in pure Python with no extra dependencies. The classifier only consults it when no
rule category matched (see `learned` in `policy.yaml`).

## Reproduce

```bash
pip install -e ".[train]"
python training/train_task_model.py --data-dir /tmp/lptm-data --evaluate-test
```

The script downloads each dataset at a pinned revision and trains one model per `--c` value.
It keeps the model with the best validation accuracy and writes it to `--output`.
Training is deterministic: re-running at the pinned revisions reproduces the bundled file byte for byte.

## Data

| Source | Revision | License | Used as |
|---|---|---|---|
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `bdd27f4d` | CC BY-SA 3.0 | `classification`, `information_extraction` → `extraction`, `summarization`, `creative_writing` → `writing`; `open_qa`, `general_qa`, `closed_qa`, `brainstorming` → `other` |
| [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | `152bb5e9` | CC BY 4.0 | `coding` |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | `e7f4b645` | Apache-2.0 | `agent_action` (first user turn whose first reply is a function call) |

Labels come from the datasets, not from us. `other` means "no opinion". When the model
predicts it, or predicts anything below `min_probability`, the rules' decision stands.

Exact duplicates, and texts that appear under more than one label, are dropped.
Each label is then capped at 4,000 rows. Rows are split 70/15/15 into train/val/test by
a CRC32 of their source id, so the split never changes between runs.

The weights are derived from the datasets above. Dolly is CC BY-SA 3.0: redistributing
the model file carries that attribution, and you should review the share-alike terms for
your use. This project's code remains Apache-2.0.

## Results (bundled model, C=32)

Validation accuracy was 86.0% across all 7 labels. At `min_probability`, "coverage" is
the share of non-`other` rows the model claims, and "precision" is how many of those it
gets right:

| Split | Threshold | Coverage | Precision |
|---|---:|---:|---:|
| val | 0.6 | 89.2% | 94.5% |
| test (run once, after tuning) | 0.6 | 87.7% | 96.0% |

Held-out test accuracy was 86.3% across all labels.

## Known limits

- There is no `chat` class. Short conversational remarks can be pushed into
  `summarization` or `classification`. Both route to the economy tier, as `chat` does.
- The `agent_action` data is API-style function calling ("what's the weather in Tokyo").
  It does not cover shell or DevOps commands ("remove /tmp/old-report", "list running
  containers"), and the model mostly defers on those.
- Some instructions that CodeAlpaca files under coding also read as plain language
  tasks. For example, "Translate hello into Spanish." is predicted as `coding`.
- English only.
