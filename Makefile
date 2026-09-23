.PHONY: test eval build run verify

PYTHON ?= python3

test:
	$(PYTHON) -m pytest tests -vv

eval:
	$(PYTHON) -m llm_preclassifier.evaluation eval/dataset.jsonl --min-accuracy 0.8

build:
	docker build --target runtime -t llm-preclassifier:local .

run:
	docker compose up --build

verify: test eval build
