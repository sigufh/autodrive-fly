.PHONY: verify audit prepare calibrate-policy evaluate-constraints evaluate-driving evaluate-city-alpha evaluate-neural-decision train-neural-v6 evaluate-neural-transfer evaluate-neural-motor-adaptation evaluate-sensory-ablation evaluate-sensory-gains evaluate-body-motor-interaction train-sensory-pathway evaluate-panorama-release api web test

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
