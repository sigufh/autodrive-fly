.PHONY: verify audit prepare calibrate-policy evaluate-constraints evaluate-driving evaluate-city-alpha evaluate-neural-decision train-neural-v6 evaluate-neural-transfer evaluate-neural-motor-adaptation evaluate-sensory-ablation evaluate-sensory-gains evaluate-body-motor-interaction train-sensory-pathway evaluate-panorama-release v7-init v7-evaluate-vision v7-evaluate-typed-vision v7-evaluate-optic-axis v7-audit-t4-sources v7-evaluate-branched-t4 v7-evaluate-t4-conductance v7-fit-t4-conductance v7-audit-retina-columns v7-audit-layer-mirror v7-audit-temporal-input v7-audit-local-input v7-audit-receptor-mask v7-audit-t4-input-coverage v7-audit-coverage-response v7-evaluate-mi9-sign v7-evaluate-conductance-order v7-audit-background-stability api web test

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

evaluate-city-alpha:
	.venv/bin/autodrive-fly evaluate-city-alpha

evaluate-neural-decision:
	.venv/bin/autodrive-fly evaluate-neural-decision

train-neural-v6:
	.venv/bin/autodrive-fly train-neural-v6

evaluate-neural-transfer:
	.venv/bin/autodrive-fly evaluate-neural-transfer

evaluate-neural-motor-adaptation:
	.venv/bin/autodrive-fly evaluate-neural-motor-adaptation

evaluate-sensory-ablation:
	.venv/bin/autodrive-fly evaluate-sensory-ablation

evaluate-sensory-gains:
	.venv/bin/autodrive-fly evaluate-sensory-gains

evaluate-body-motor-interaction:
	.venv/bin/autodrive-fly evaluate-body-motor-interaction

train-sensory-pathway:
	.venv/bin/autodrive-fly train-sensory-pathway

evaluate-panorama-release:
	.venv/bin/autodrive-fly evaluate-panorama-release

v7-init:
	.venv/bin/autodrive-fly v7-init

v7-evaluate-vision:
	.venv/bin/autodrive-fly v7-evaluate-vision

v7-evaluate-typed-vision:
	.venv/bin/autodrive-fly v7-evaluate-typed-vision

v7-evaluate-optic-axis:
	.venv/bin/autodrive-fly v7-evaluate-optic-axis

v7-audit-t4-sources:
	.venv/bin/autodrive-fly v7-audit-t4-sources

v7-evaluate-branched-t4:
	.venv/bin/autodrive-fly v7-evaluate-branched-t4

v7-evaluate-t4-conductance:
	.venv/bin/autodrive-fly v7-evaluate-t4-conductance

v7-fit-t4-conductance:
	.venv/bin/autodrive-fly v7-fit-t4-conductance

v7-audit-retina-columns:
	.venv/bin/autodrive-fly v7-audit-retina-columns

v7-audit-layer-mirror:
	.venv/bin/autodrive-fly v7-audit-layer-mirror

v7-audit-temporal-input:
	.venv/bin/autodrive-fly v7-audit-temporal-input

v7-audit-local-input:
	.venv/bin/autodrive-fly v7-audit-local-input

v7-audit-receptor-mask:
	.venv/bin/autodrive-fly v7-audit-receptor-mask

v7-audit-t4-input-coverage:
	.venv/bin/autodrive-fly v7-audit-t4-input-coverage

v7-audit-coverage-response:
	.venv/bin/autodrive-fly v7-audit-coverage-response

v7-evaluate-mi9-sign:
	.venv/bin/autodrive-fly v7-evaluate-mi9-sign

v7-evaluate-conductance-order:
	.venv/bin/autodrive-fly v7-evaluate-conductance-order

v7-audit-background-stability:
	.venv/bin/autodrive-fly v7-audit-background-stability

calibrate-policy:
	.venv/bin/autodrive-fly calibrate-policy

evaluate-constraints:
	.venv/bin/autodrive-fly evaluate-constraints \
		--checkpoint artifacts/checkpoints/driving-policy.calibrated-candidate.npz

api:
	./scripts/run-api.sh

web:
	npm_config_cache=.npm-cache npm --prefix apps/web run dev

test:
	.venv/bin/pytest -q
	.venv/bin/ruff check src apps/api tests
	npm_config_cache=.npm-cache npm --prefix apps/web test
	npm_config_cache=.npm-cache npm --prefix apps/web run build
