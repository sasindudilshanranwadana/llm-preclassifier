.PHONY: test eval eval-blind eval-semantic build run verify

PYTHON ?= python3

test:
	$(PYTHON) -m pytest tests -vv

eval:
	$(PYTHON) -m llm_preclassifier.evaluation eval/dataset.jsonl --min-accuracy 0.95
	$(PYTHON) -m llm_preclassifier.evaluation eval/holdout.jsonl --min-accuracy 0.95

# Never tune rules, exemplars or thresholds against the blind sets; they measure generalisation.
eval-blind:
	$(PYTHON) -m llm_preclassifier.evaluation eval/blind.jsonl
	$(PYTHON) -m llm_preclassifier.evaluation eval/blind-v2.jsonl

SEMANTIC_MODEL ?= BAAI/bge-small-en-v1.5
eval-semantic:
	$(PYTHON) -m llm_preclassifier.evaluation eval/dataset.jsonl --semantic $(SEMANTIC_MODEL) --min-accuracy 0.95
	$(PYTHON) -m llm_preclassifier.evaluation eval/holdout.jsonl --semantic $(SEMANTIC_MODEL) --min-accuracy 0.95
	$(PYTHON) -m llm_preclassifier.evaluation eval/blind.jsonl --semantic $(SEMANTIC_MODEL)
	$(PYTHON) -m llm_preclassifier.evaluation eval/blind-v2.jsonl --semantic $(SEMANTIC_MODEL)

build:
	docker build --target runtime -t llm-preclassifier:local .

run:
	docker compose up --build

verify: test eval build
