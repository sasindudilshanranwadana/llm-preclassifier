# Training the bundled task model

`src/llm_preclassifier/data/task_model.bin` is a hashed n-gram logistic regression
(32,768 buckets, int8 weights, ~300 KB). It is trained here, and at runtime it runs
in pure Python with no extra dependencies. The classifier only consults it when no
rule category matched and the semantic layer has no vote (see `learned` in `policy.yaml`).

## Reproduce

```bash
pip install -e ".[train]"
python training/train_task_model.py --data-dir /tmp/lptm-data --evaluate-test
```

The script downloads each source at a pinned revision and trains one model per `--c` value.
It keeps the model with the best validation accuracy and writes it to `--output`.
Training is deterministic: re-running at the pinned revisions reproduces the bundled file byte for byte.

## Data

| Source | Revision | License | Used as |
|---|---|---|---|
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `bdd27f4d` | CC BY-SA 3.0 | `classification`, `information_extraction` → `extraction`, `summarization`, `creative_writing` → `writing`; `open_qa`, `general_qa`, `closed_qa`, `brainstorming` → `other` |
| [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | `152bb5e9` | CC BY 4.0 | `coding` |
| [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) (Python) | `fc56fe33` | MIT | `coding` (commit subjects) |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | `e7f4b645` | Apache-2.0 | `agent_action` (first user turn whose first reply is a function call) |
| [tldr-pages/tldr](https://github.com/tldr-pages/tldr) (common, linux, osx) | `59f09394` | CC BY 4.0 | `agent_action` (shell example descriptions) |
| [allenai/soda](https://huggingface.co/datasets/allenai/soda) (validation split) | `fdc848ab` | CC BY 4.0 | `chat` (dialogue turns) |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (main, train) | `740312ad` | MIT | `reasoning` |

Labels come from the sources, not from us. `other` means "no opinion". When the model
predicts it, or predicts anything below `min_probability`, the rules' decision stands.

Several steps keep the data balanced and the benchmarks honest:
- Exact duplicates, texts that appear under more than one label, and any prompt from
  `eval/*.jsonl` are dropped, so the benchmarks stay blind. The script refuses to run
  if it finds no benchmark prompts.
- Each label is capped at 4,000 rows, shared evenly between its sources.
- Rows are split 70/15/15 into train/val/test by a CRC32 of their source id, so the split
  never changes between runs. A tldr page and a SODA dialogue are never split across sets.

The weights are derived from the sources above. Dolly is CC BY-SA 3.0: redistributing
the model file carries that attribution, and you should review the share-alike terms for
your use. The CC BY 4.0 sources require attribution, which this table provides. This
project's code remains Apache-2.0.

## Results (bundled model, C=32)

Validation accuracy was 88.7% across all 9 labels. At `min_probability`, "coverage" is
the share of non-`other` rows the model claims, and "precision" is how many of those it
gets right:

| Split | Threshold | Coverage | Precision |
|---|---:|---:|---:|
| val | 0.6 | 91.3% | 95.7% |
| test (run once, after tuning) | 0.6 | 91.1% | 95.9% |

Held-out test accuracy was 89.0% across all labels.

The model never answers before the semantic vote. I tried letting it answer first, because on the
validation split it was right on 98% of the prompts it routed, against 38% for the semantic vote. That
split comes from the same data the model was trained on, though, and the change lowered the
rules + semantic blind benchmarks (79.2% → 75.0% and 93.3% → 90.0%), so it was reverted.

## Known limits

- Follow-ups that only make sense with earlier turns ("Fix it and make them consistent")
  are classified from the latest message alone.
- Commit subjects and shell descriptions are both short imperatives. Short ops commands
  are sometimes read as `coding` ("Check whether nginx is running.") or left to the rules.
- Some instructions that CodeAlpaca files under coding also read as plain language
  tasks. For example, "Translate hello into Spanish." is predicted as `coding`.
- English only.
