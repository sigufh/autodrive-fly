.PHONY: verify audit prepare calibrate-policy evaluate-constraints evaluate-driving evaluate-city-alpha evaluate-neural-decision train-neural-v6 evaluate-neural-transfer evaluate-neural-motor-adaptation evaluate-sensory-ablation evaluate-sensory-gains evaluate-body-motor-interaction train-sensory-pathway evaluate-panorama-release v7-init v7-evaluate-vision v7-evaluate-typed-vision v7-evaluate-optic-axis v7-audit-t4-sources v7-evaluate-branched-t4 v7-evaluate-t4-conductance v7-fit-t4-conductance v7-audit-retina-columns v7-audit-layer-mirror v7-audit-temporal-input v7-audit-local-input v7-audit-receptor-mask v7-audit-t4-input-coverage v7-audit-coverage-response v7-evaluate-mi9-sign v7-evaluate-conductance-order v7-audit-background-stability v7-audit-feedback-cut v7-evaluate-synchronous-update v7-audit-normalization-gain v7-audit-full-update-perturbation v7-evaluate-phase-motion v7-evaluate-geometry-sign v7-evaluate-pixel-sampling v7-audit-spectral-controls v7-evaluate-neural-spectra v7-audit-electrophysiology v7-audit-timebase v7-validate-published-fig5 v7-build-ephys-interface v7-audit-t5-data v7-audit-t5-conductance v7-extract-t5-phenotype v7-audit-goal-coverage v7-audit-t5-labels v7-freeze-stage1-split v7-audit-t5-supplement v7-freeze-stage1-scoring v7-audit-stage1-input v7-evaluate-stage1-development v7-evaluate-stage1-geometry-ab v7-audit-visual-target-inputs v7-audit-looming-mechanisms v7-freeze-stage1-nested v7-freeze-target-fit v7-evaluate-closed-loop-tuning v7-evaluate-closed-loop-calibration v7-evaluate-closed-loop-controls v7-evaluate-closed-loop-multi v7-evaluate-r1r6-multi-tuning v7-evaluate-r1r6-multi-calibration v7-evaluate-r1r6-local-tuning v7-evaluate-r1r6-local-calibration v7-evaluate-lplc2-radial-opponency v7-evaluate-lplc2-position-coverage v7-evaluate-t4t5-local-edge-precheck v7-evaluate-t4-local-correlator-precheck api web test

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

v7-audit-feedback-cut:
	.venv/bin/autodrive-fly v7-audit-feedback-cut

v7-evaluate-synchronous-update:
	.venv/bin/autodrive-fly v7-evaluate-synchronous-update

v7-audit-normalization-gain:
	.venv/bin/autodrive-fly v7-audit-normalization-gain

v7-audit-full-update-perturbation:
	.venv/bin/autodrive-fly v7-audit-full-update-perturbation

v7-evaluate-phase-motion:
	.venv/bin/autodrive-fly v7-evaluate-phase-motion

v7-evaluate-geometry-sign:
	.venv/bin/autodrive-fly v7-evaluate-geometry-sign

v7-evaluate-pixel-sampling:
	.venv/bin/autodrive-fly v7-evaluate-pixel-sampling

v7-audit-spectral-controls:
	.venv/bin/autodrive-fly v7-audit-spectral-controls

v7-evaluate-neural-spectra:
	.venv/bin/autodrive-fly v7-evaluate-neural-spectra

v7-audit-electrophysiology:
	.venv/bin/autodrive-fly v7-audit-electrophysiology

v7-audit-timebase:
	.venv/bin/autodrive-fly v7-audit-timebase

v7-validate-published-fig5:
	.venv/bin/autodrive-fly v7-validate-published-fig5

v7-build-ephys-interface:
	.venv/bin/autodrive-fly v7-build-ephys-interface

v7-audit-t5-data:
	.venv/bin/autodrive-fly v7-audit-t5-data

v7-audit-t5-conductance:
	.venv/bin/autodrive-fly v7-audit-t5-conductance

v7-extract-t5-phenotype:
	.venv/bin/autodrive-fly v7-extract-t5-phenotype

v7-audit-goal-coverage:
	.venv/bin/autodrive-fly v7-audit-goal-coverage

v7-audit-t5-labels:
	.venv/bin/autodrive-fly v7-audit-t5-labels

v7-freeze-stage1-split:
	.venv/bin/autodrive-fly v7-freeze-stage1-split

v7-audit-t5-supplement:
	.venv/bin/autodrive-fly v7-audit-t5-supplement

v7-freeze-stage1-scoring:
	.venv/bin/autodrive-fly v7-freeze-stage1-scoring

v7-audit-stage1-input:
	.venv/bin/autodrive-fly v7-audit-stage1-input

v7-evaluate-stage1-development:
	.venv/bin/autodrive-fly v7-evaluate-stage1-development

v7-evaluate-stage1-geometry-ab:
	.venv/bin/autodrive-fly v7-evaluate-stage1-geometry-ab

v7-audit-visual-target-inputs:
	.venv/bin/autodrive-fly v7-audit-visual-target-inputs

v7-audit-looming-mechanisms:
	.venv/bin/autodrive-fly v7-audit-looming-mechanisms

v7-freeze-stage1-nested:
	.venv/bin/autodrive-fly v7-freeze-stage1-nested

v7-freeze-target-fit:
	.venv/bin/autodrive-fly v7-freeze-target-fit

v7-evaluate-closed-loop-tuning:
	.venv/bin/autodrive-fly v7-evaluate-closed-loop-tuning

v7-evaluate-closed-loop-calibration:
	.venv/bin/autodrive-fly v7-evaluate-closed-loop-calibration

v7-evaluate-closed-loop-controls:
	.venv/bin/autodrive-fly v7-evaluate-closed-loop-controls

v7-evaluate-closed-loop-multi:
	.venv/bin/autodrive-fly v7-evaluate-closed-loop-multi

v7-evaluate-r1r6-multi-tuning:
	.venv/bin/autodrive-fly v7-evaluate-r1r6-multi-tuning

v7-evaluate-r1r6-multi-calibration:
	.venv/bin/autodrive-fly v7-evaluate-r1r6-multi-calibration

v7-evaluate-r1r6-local-tuning:
	.venv/bin/autodrive-fly v7-evaluate-r1r6-local-tuning

v7-evaluate-r1r6-local-calibration:
	.venv/bin/autodrive-fly v7-evaluate-r1r6-local-calibration

v7-evaluate-r1r6-local-controls:
	.venv/bin/autodrive-fly v7-evaluate-r1r6-local-controls

v7-evaluate-neural-channels-tuning:
	.venv/bin/autodrive-fly v7-evaluate-neural-channels-tuning

v7-evaluate-neural-channels-calibration:
	.venv/bin/autodrive-fly v7-evaluate-neural-channels-calibration

v7-evaluate-neural-channel-controls:
	.venv/bin/autodrive-fly v7-evaluate-neural-channel-controls

v7-evaluate-neural-topology-controls:
	.venv/bin/autodrive-fly v7-evaluate-neural-topology-controls

v7-evaluate-nested-neural-screen:
	.venv/bin/autodrive-fly v7-evaluate-nested-neural-screen

v7-evaluate-t5-spatial-order:
	.venv/bin/autodrive-fly v7-evaluate-t5-spatial-order

v7-evaluate-lplc2-phenotype:
	.venv/bin/autodrive-fly v7-evaluate-lplc2-phenotype

v7-evaluate-t4-source-resolved:
	.venv/bin/autodrive-fly v7-evaluate-t4-source-resolved

v7-evaluate-heading-ring:
	.venv/bin/autodrive-fly v7-evaluate-heading-ring

v7-evaluate-neural-local-columns:
	.venv/bin/autodrive-fly v7-evaluate-neural-local-columns

v7-evaluate-neural-episode-cv:
	.venv/bin/autodrive-fly v7-evaluate-neural-episode-cv

v7-evaluate-neural-episode-controls:
	.venv/bin/autodrive-fly v7-evaluate-neural-episode-controls

v7-evaluate-fc2-pfl-dna:
	.venv/bin/autodrive-fly v7-evaluate-fc2-pfl-dna

v7-freeze-navigation-nested:
	.venv/bin/autodrive-fly v7-freeze-navigation-nested

v7-evaluate-navigation-nested:
	.venv/bin/autodrive-fly v7-evaluate-navigation-nested

v7-evaluate-danger-throttle:
	.venv/bin/autodrive-fly v7-evaluate-danger-throttle

v7-evaluate-visual-corridor:
	.venv/bin/autodrive-fly v7-evaluate-visual-corridor

v7-evaluate-neural-corridor:
	.venv/bin/autodrive-fly v7-evaluate-neural-corridor

v7-evaluate-local-column-corridor:
	.venv/bin/autodrive-fly v7-evaluate-local-column-corridor

v7-evaluate-neural-dynamics-local:
	.venv/bin/autodrive-fly v7-evaluate-neural-dynamics-local

v7-evaluate-neural-corridor-dagger:
	.venv/bin/autodrive-fly v7-evaluate-neural-corridor-dagger

v7-evaluate-visual-layer-locality:
	.venv/bin/autodrive-fly v7-evaluate-visual-layer-locality

v7-evaluate-lamina-goal:
	.venv/bin/autodrive-fly v7-evaluate-lamina-goal

v7-evaluate-lamina-goal-symmetry:
	.venv/bin/autodrive-fly v7-evaluate-lamina-goal-symmetry

v7-evaluate-t4-normalized-correlator:
	.venv/bin/autodrive-fly v7-evaluate-t4-normalized-correlator

v7-evaluate-fc2-goal-memory:
	.venv/bin/autodrive-fly v7-evaluate-fc2-goal-memory

v7-evaluate-lplc-typed-screen:
	.venv/bin/autodrive-fly v7-evaluate-lplc-typed-screen

v7-evaluate-lplc2-radial-opponency:
	.venv/bin/autodrive-fly v7-evaluate-lplc2-radial-opponency

v7-evaluate-lplc2-position-coverage:
	.venv/bin/autodrive-fly v7-evaluate-lplc2-position-coverage

v7-evaluate-t4t5-local-edge-precheck:
	.venv/bin/autodrive-fly v7-evaluate-t4t5-local-edge-precheck

v7-evaluate-t4t5-local-edge-backends:
	.venv/bin/autodrive-fly v7-evaluate-t4t5-local-edge-backends

v7-evaluate-t4-local-correlator-precheck:
	.venv/bin/autodrive-fly v7-evaluate-t4-local-correlator-precheck

v7-evaluate-t4-source-pool-local:
	.venv/bin/autodrive-fly v7-evaluate-t4-source-pool-local

.PHONY: v7-evaluate-t4-source-pool-local

v7-evaluate-three-hop-moment:
	.venv/bin/autodrive-fly v7-evaluate-three-hop-moment

.PHONY: v7-evaluate-three-hop-moment

v7-audit-three-hop-source-coverage:
	.venv/bin/autodrive-fly v7-audit-three-hop-source-coverage

.PHONY: v7-audit-three-hop-source-coverage

v7-evaluate-four-hop-scalar-precheck:
	.venv/bin/autodrive-fly v7-evaluate-four-hop-scalar-precheck

.PHONY: v7-evaluate-four-hop-scalar-precheck

v7-evaluate-lc4-position-speed-precheck:
	.venv/bin/autodrive-fly v7-evaluate-lc4-position-speed-precheck

.PHONY: v7-evaluate-lc4-position-speed-precheck

v7-evaluate-t5-lamina-split:
	.venv/bin/autodrive-fly v7-evaluate-t5-lamina-split

.PHONY: v7-evaluate-t5-lamina-split

v7-evaluate-t5-lamina-scalar-precheck:
	.venv/bin/autodrive-fly v7-evaluate-t5-lamina-scalar-precheck

.PHONY: v7-evaluate-t5-lamina-scalar-precheck

v7-audit-t5-source-axis:
	.venv/bin/autodrive-fly v7-audit-t5-source-axis

.PHONY: v7-audit-t5-source-axis

v7-evaluate-lplc1-near-collision-precheck:
	.venv/bin/autodrive-fly v7-evaluate-lplc1-near-collision-precheck

.PHONY: v7-evaluate-lplc1-near-collision-precheck

v7-audit-lplc1-input-structure:
	.venv/bin/autodrive-fly v7-audit-lplc1-input-structure

.PHONY: v7-audit-lplc1-input-structure

v7-evaluate-neural-goal-fusion:
	.venv/bin/autodrive-fly v7-evaluate-neural-goal-fusion

v7-freeze-fusion-nested:
	.venv/bin/autodrive-fly v7-freeze-fusion-nested

v7-evaluate-fusion-nested:
	.venv/bin/autodrive-fly v7-evaluate-fusion-nested

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
