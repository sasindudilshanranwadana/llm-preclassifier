"""Train the bundled task-type model from pinned public datasets.

Sources and labels (labels come from each dataset, not from us):
  - databricks/databricks-dolly-15k (CC BY-SA 3.0): classification,
    information_extraction -> extraction, summarization, creative_writing -> writing;
    open_qa, general_qa, closed_qa and brainstorming -> "other" (defer to the rules).
  - sahil2801/CodeAlpaca-20k (CC BY 4.0): coding.
  - glaiveai/glaive-function-calling-v2 (Apache-2.0): first user turn of conversations
    whose first assistant turn calls a function -> agent_action.
  - tldr-pages/tldr (CC BY 4.0): example descriptions of shell commands -> agent_action.
  - bigcode/commitpackft (MIT), Python subset: commit subjects -> coding.
  - allenai/soda (CC BY 4.0), validation split: dialogue turns -> chat.
  - openai/gsm8k (MIT): word problems -> reasoning.

Rows matching a prompt in eval/*.jsonl are dropped so the benchmarks stay blind.

Rows are split 70/15/15 into train/val/test by a stable hash of their source id.
Tune only on val. Pass --evaluate-test once, after tuning is frozen.

    pip install -e ".[train]"
    python training/train_task_model.py --data-dir DIR [--evaluate-test]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import urllib.request
import zlib
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from llm_preclassifier.task_model import DEFERRED_LABEL, TaskModel, features

SOURCES = {
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
              "databricks-dolly-15k.jsonl"),
    "codealpaca": ("sahil2801/CodeAlpaca-20k", "152bb5e9a29651266b018106053980070a0521a1",
                   "code_alpaca_20k.json"),
    "glaive": ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
               "glaive-function-calling-v2.json"),
    "commitpack": ("bigcode/commitpackft", "fc56fe33c030c6daa414c2b112c932b8eed085e6", "data/python/data.jsonl"),
    "soda": ("allenai/soda", "fdc848ab0183208ea7808206c91c724414d0a071", "valid.parquet"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/train-00000-of-00001.parquet"),
}
TLDR = ("tldr-pages/tldr", "59f09394d48f5ac3d354ec0d7e9aebafb6856155")
TLDR_PLATFORMS = ("common", "linux", "osx")
DOLLY_LABELS = {
    "classification": "classification", "information_extraction": "extraction",
    "summarization": "summarization", "creative_writing": "writing",
    "open_qa": DEFERRED_LABEL, "general_qa": DEFERRED_LABEL, "closed_qa": DEFERRED_LABEL,
    "brainstorming": DEFERRED_LABEL,
}
PER_LABEL_CAP = 4000
BUCKETS = 1 << 15
SEED = 20260924
_TLDR_EXAMPLE = re.compile(r"^- (.+?):?$", re.MULTILINE)
_TLDR_PAGE = re.compile(rf"/pages/(?:{'|'.join(TLDR_PLATFORMS)})/([^/]+)\.md$")
_FIRST_TURN = re.compile(r"USER:(.*?)\n\s*\n\s*(?:ASSISTANT|FUNCTION RESPONSE):(.*?)(?:<\|endoftext\|>|\n\s*\n\s*USER:|$)",
                         re.DOTALL)


def download(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        name: (f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}",
               f"{name}-{revision[:8]}{Path(filename).suffix}")
        for name, (repo, revision, filename) in SOURCES.items()
    }
    urls["tldr"] = (f"https://codeload.github.com/{TLDR[0]}/tar.gz/{TLDR[1]}", f"tldr-{TLDR[1][:8]}.tar.gz")
    paths = {}
    for name, (url, filename) in urls.items():
        path = data_dir / filename
        if not path.exists():
            print(f"downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def eval_prompts(eval_dir: Path) -> set[str]:
    prompts = set()
    for path in eval_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                case = json.loads(line)
                texts = [case.get("prompt", "")] + [
                    m["content"] for m in case.get("messages", []) if isinstance(m.get("content"), str)
                ]
                prompts.update(normalize(text) for text in texts if text)
    return prompts


def split_of(source_id: str) -> str:
    bucket = zlib.crc32(source_id.encode("utf-8")) % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def load_rows(paths: dict[str, Path]) -> list[dict]:
    rows = []
    for index, line in enumerate(paths["dolly"].open(encoding="utf-8")):
        record = json.loads(line)
        text = record["instruction"].strip()
        # Alternate between bare instructions and instructions with their passage, as users send both.
        if record["context"].strip() and index % 2:
            text = f"{text}\n\n{record['context'].strip()}"
        rows.append({"id": f"dolly:{index}", "text": text, "label": DOLLY_LABELS[record["category"]]})
    for index, record in enumerate(json.loads(paths["codealpaca"].read_text(encoding="utf-8"))):
        text = record["instruction"].strip()
        if record["input"].strip():
            text = f"{text}\n\n{record['input'].strip()}"
        rows.append({"id": f"codealpaca:{index}", "text": text, "label": "coding"})
    for index, record in enumerate(json.loads(paths["glaive"].read_text(encoding="utf-8"))):
        match = _FIRST_TURN.search(record["chat"])
        if match and "<functioncall>" in match.group(2):
            rows.append({"id": f"glaive:{index}", "text": match.group(1).strip(), "label": "agent_action"})
    for index, line in enumerate(paths["commitpack"].open(encoding="utf-8")):
        rows.append({"id": f"commitpack:{index}", "text": json.loads(line)["subject"].strip(), "label": "coding"})
    for index, dialogue in enumerate(pq.read_table(paths["soda"], columns=["dialogue"]).column(0).to_pylist()):
        # All turns, not just openers, so short replies ("Sounds good.") are covered; the dialogue is the split group.
        for turn, text in enumerate(dialogue or []):
            rows.append({"id": f"soda:{index}:{turn}", "group": f"soda:{index}", "text": text.strip(), "label": "chat"})
    for index, question in enumerate(pq.read_table(paths["gsm8k"], columns=["question"]).column(0).to_pylist()):
        rows.append({"id": f"gsm8k:{index}", "text": question.strip(), "label": "reasoning"})
    rows.extend(tldr_rows(paths["tldr"]))
    return rows


def tldr_rows(path: Path) -> list[dict]:
    """One row per example description; the page is the split group so a command never straddles splits."""
    rows = []
    with tarfile.open(path) as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            page = _TLDR_PAGE.search(member.name)
            if not page or not member.isfile():
                continue
            text = archive.extractfile(member).read().decode("utf-8")
            if "This command is an alias of" in text:
                continue
            for number, match in enumerate(_TLDR_EXAMPLE.finditer(text)):
                description = match.group(1).replace("`", "").strip()
                rows.append({"id": f"tldr:{page.group(1)}:{number}", "group": f"tldr:{page.group(1)}",
                             "text": description, "label": "agent_action"})
    return rows


def source_of(row: dict) -> str:
    return row["id"].split(":", 1)[0]


def prepare(rows: list[dict], blocked: set[str]) -> list[dict]:
    """Drop empty, conflicting, duplicate and benchmark rows, then cap each label, shared evenly by its sources."""
    labels_by_text: dict[str, set[str]] = {}
    for row in rows:
        labels_by_text.setdefault(normalize(row["text"]), set()).add(row["label"])
    seen: set[str] = set()
    kept = []
    for row in rows:
        key = normalize(row["text"])
        if not key or len(labels_by_text[key]) > 1 or key in seen or key in blocked:
            continue
        seen.add(key)
        kept.append({**row, "split": split_of(row.get("group", row["id"]))})
    sources_per_label: dict[str, set[str]] = {}
    for row in kept:
        sources_per_label.setdefault(row["label"], set()).add(source_of(row))
    rng = random.Random(SEED)
    rng.shuffle(kept)
    counts: Counter[tuple[str, str, str]] = Counter()
    capped = []
    for row in kept:
        share = PER_LABEL_CAP // len(sources_per_label[row["label"]])
        cap = share * {"train": 70, "val": 15, "test": 15}[row["split"]] // 100
        key = (row["split"], row["label"], source_of(row))
        if counts[key] < cap:
            counts[key] += 1
            capped.append(row)
    return capped


def matrix(rows: list[dict]) -> csr_matrix:
    data, cols, indptr = [], [], [0]
    for row in rows:
        for index, value in features(row["text"], BUCKETS).items():
            cols.append(index)
            data.append(value)
        indptr.append(len(cols))
    return csr_matrix((data, cols, indptr), shape=(len(rows), BUCKETS), dtype=np.float32)


def quantize(classifier: LogisticRegression, metadata: dict) -> TaskModel:
    labels = [str(label) for label in classifier.classes_]
    coef = classifier.coef_.astype(np.float64)
    scales = [float(max(np.abs(row).max(), 1e-12) / 127) for row in coef]
    quantized = np.vstack([np.clip(np.round(row / scale), -127, 127) for row, scale in zip(coef, scales)])
    weights = array("b", quantized.astype(np.int8).ravel().tobytes())
    return TaskModel(labels, BUCKETS, scales, [float(b) for b in classifier.intercept_], weights, metadata)


def coverage_curve(model: TaskModel, rows: list[dict]) -> list[str]:
    predictions = [(row["label"], model.predict(row["text"])) for row in rows]
    lines = ["threshold  coverage  precision  (routed = non-'other' prediction above threshold)"]
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        routed = [(gold, p) for gold, p in predictions if p.task_type != DEFERRED_LABEL and p.probability >= threshold]
        claimable = sum(gold != DEFERRED_LABEL for gold, _ in predictions)
        correct = sum(gold == p.task_type for gold, p in routed)
        lines.append(f"  {threshold:.1f}      {len(routed) / max(claimable, 1):6.1%}   "
                     f"{correct / max(len(routed), 1):6.1%}")
    return lines


def accuracy(model: TaskModel, rows: list[dict]) -> float:
    return sum(model.predict(row["text"]).task_type == row["label"] for row in rows) / len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=Path("eval"), help="benchmark prompts to exclude")
    parser.add_argument("--output", type=Path, default=Path("src/llm_preclassifier/data/task_model.bin"))
    parser.add_argument("--c", type=float, nargs="+", default=[0.5, 2.0, 8.0, 32.0])
    parser.add_argument("--evaluate-test", action="store_true", help="report held-out test metrics (run once)")
    args = parser.parse_args(argv)

    blocked = eval_prompts(args.eval_dir)
    if not blocked:
        parser.error(f"no benchmark prompts found in {args.eval_dir}; refusing to train without decontamination")
    rows = prepare(load_rows(download(args.data_dir)), blocked)
    by_split = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    for name, split_rows in by_split.items():
        print(f"{name}: {len(split_rows)} {dict(sorted(Counter(r['label'] for r in split_rows).items()))}")
    x_train = matrix(by_split["train"])
    y_train = [row["label"] for row in by_split["train"]]

    best = None
    for c in args.c:
        classifier = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        classifier.fit(x_train, y_train)
        model = quantize(classifier, {})
        val_accuracy = accuracy(model, by_split["val"])
        print(f"C={c:<6} val accuracy={val_accuracy:.1%}")
        if best is None or val_accuracy > best[0]:
            best = (val_accuracy, c, classifier)

    val_accuracy, c, classifier = best
    metadata = {
        "sources": {**{name: {"repo": repo, "revision": revision} for name, (repo, revision, _) in SOURCES.items()},
                    "tldr": {"repo": TLDR[0], "revision": TLDR[1]}},
        "per_label_cap": PER_LABEL_CAP, "seed": SEED, "c": c, "val_accuracy": round(val_accuracy, 4),
        "train_rows": len(by_split["train"]),
    }
    model = quantize(classifier, metadata)
    args.output.write_bytes(model.to_bytes())
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes), C={c}, val accuracy={val_accuracy:.1%}")
    print("\n".join(["val:"] + coverage_curve(model, by_split["val"])))
    if args.evaluate_test:
        print(f"TEST accuracy={accuracy(model, by_split['test']):.1%}")
        print("\n".join(["test:"] + coverage_curve(model, by_split["test"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
