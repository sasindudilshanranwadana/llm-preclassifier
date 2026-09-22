.PHONY: test build run verify

test:
	python3 -m pytest tests -vv

build:
	docker build --target runtime -t llm-preclassifier:local .

run:
	docker compose up --build

verify: test build
