.PHONY: verify audit prepare evaluate-driving api web test

verify:
	.venv/bin/autodrive-fly verify --dataset malecns

audit:
	.venv/bin/autodrive-fly audit

prepare:
	.venv/bin/autodrive-fly build-graph
	.venv/bin/autodrive-fly build-overview
	.venv/bin/autodrive-fly build-pathways

evaluate-driving:
	.venv/bin/autodrive-fly evaluate-driving

api:
	./scripts/run-api.sh

web:
	npm_config_cache=.npm-cache npm --prefix apps/web run dev

test:
	.venv/bin/pytest -q
	.venv/bin/ruff check src apps/api tests
	npm_config_cache=.npm-cache npm --prefix apps/web test
	npm_config_cache=.npm-cache npm --prefix apps/web run build
